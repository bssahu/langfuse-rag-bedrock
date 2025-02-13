import os
import logging
import functools
import asyncio
import json
from typing import Optional, Dict, Any
from langfuse import Langfuse
from langfuse.decorators import langfuse_context

logger = logging.getLogger(__name__)

def serialize_for_langfuse(obj: Any) -> Any:
    """Safely serialize objects for Langfuse."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return json.dumps([serialize_for_langfuse(item) for item in obj])
    elif isinstance(obj, dict):
        return json.dumps({
            str(k): serialize_for_langfuse(v) 
            for k, v in obj.items()
        })
    else:
        return str(obj)

# Initialize Langfuse client with local configuration
try:
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-lf-local"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-lf-local"),
        host=os.getenv("LANGFUSE_HOST", "http://langfuse:3000"),
        debug=True  # Enable debug mode for local development
    )
    # Test the connection
    langfuse.auth_check()
    logger.info("Langfuse client initialized and authenticated successfully")
except Exception as e:
    logger.error(f"Failed to initialize Langfuse client: {str(e)}")
    logger.warning("Continuing without Langfuse observability")
    langfuse = None

def observe_llm(name: Optional[str] = None, 
                metadata: Optional[Dict[str, Any]] = None):
    """
    Decorator for observing LLM calls with Langfuse.
    
    Args:
        name: Optional name for the generation
        metadata: Optional metadata to attach to the generation
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not langfuse:
                logger.warning("Langfuse client not initialized, skipping observability")
                return await func(*args, **kwargs)

            generation = None
            try:
                # Extract and serialize model info and parameters
                model = kwargs.get('model_id') or kwargs.get('model', 'unknown-model')
                clean_params = {
                    k: serialize_for_langfuse(v)
                    for k, v in kwargs.items()
                }

                # Clean and serialize input
                input_data = kwargs.get('messages', kwargs.get('prompt', args[0] if args else None))
                serialized_input = serialize_for_langfuse(input_data)

                # Start generation span
                generation = langfuse.generation(
                    name=name or func.__name__,
                    model=model,
                    model_parameters=clean_params,
                    metadata={
                        **(metadata or {}),
                        "environment": "local",
                        "service": "rag-chatbot"
                    },
                    input=serialized_input
                )

                # Execute the LLM call
                response = await func(*args, **kwargs)

                # Serialize output
                output_data = serialize_for_langfuse(response)

                # Format usage data correctly for Langfuse
                usage = {
                    'input': 0,
                    'output': 0,
                    'total': 0,
                    'unit': 'tokens'
                }

                # End generation with response
                if generation:
                    generation.end(
                        output=output_data,
                        usage=usage
                    )

                return response

            except Exception as e:
                if generation:
                    generation.end(error=str(e))
                logger.error(f"Error in LLM call: {str(e)}")
                raise
            finally:
                if generation:
                    try:
                        await langfuse.flush()
                    except Exception as e:
                        logger.error(f"Error flushing Langfuse data: {str(e)}")

        def sync_wrapper(*args, **kwargs):
            if not langfuse:
                logger.warning("Langfuse client not initialized, skipping observability")
                return func(*args, **kwargs)

            generation = None
            try:
                # Extract and serialize model info and parameters
                model = kwargs.get('model_id') or kwargs.get('model', 'unknown-model')
                clean_params = {
                    k: serialize_for_langfuse(v)
                    for k, v in kwargs.items()
                }

                # Clean and serialize input
                input_data = kwargs.get('messages', kwargs.get('prompt', args[0] if args else None))
                serialized_input = serialize_for_langfuse(input_data)

                # Start generation span
                generation = langfuse.generation(
                    name=name or func.__name__,
                    model=model,
                    model_parameters=clean_params,
                    metadata={
                        **(metadata or {}),
                        "environment": "local",
                        "service": "rag-chatbot"
                    },
                    input=serialized_input
                )

                # Execute the LLM call
                response = func(*args, **kwargs)

                # Serialize output
                output_data = serialize_for_langfuse(response)

                # Format usage data correctly for Langfuse
                usage = {
                    'input': 0,
                    'output': 0,
                    'total': 0,
                    'unit': 'tokens'
                }

                # End generation with response
                if generation:
                    generation.end(
                        output=output_data,
                        usage=usage
                    )

                return response

            except Exception as e:
                if generation:
                    generation.end(error=str(e))
                logger.error(f"Error in LLM call: {str(e)}")
                raise
            finally:
                if generation:
                    try:
                        langfuse.flush()
                    except Exception as e:
                        logger.error(f"Error flushing Langfuse data: {str(e)}")

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 