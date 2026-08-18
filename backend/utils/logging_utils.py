import time 

import logging 

from functools import wraps


logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger("app_logger")

def log_latency(func):

    @wraps(func)
    async def wrapper(*args,**kwargs):

        start_time = time.time()

        result = await func(*args, **kwargs)

        end_time = time.time()

        duration = end_time - start_time

        logger.info(
            f"{func.__name__} took {duration: .2f} seconds"
        )

        return result
    
    return wrapper