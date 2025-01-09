import os
import pickle
import hashlib
from functools import wraps


def get_cache_dir():
    """Get the cache directory path"""
    # Use a local cache directory in the project
    cache_dir = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache(key_fn):
    """Cache decorator that saves function outputs to disk"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache directory
            cache_dir = get_cache_dir()

            # Generate cache key
            if key_fn is None:
                key = str(args) + str(kwargs)
            else:
                key = str(key_fn(*args, **kwargs))

            # Create hash of key for filename
            key_hash = hashlib.md5(key.encode()).hexdigest()

            # Create cache folder for this function
            cache_folder = os.path.join(cache_dir, func.__name__)
            os.makedirs(cache_folder, exist_ok=True)

            cache_file = os.path.join(cache_folder, f"{key_hash}.pkl")

            # Return cached result if it exists
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            # Otherwise compute result and cache it
            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
