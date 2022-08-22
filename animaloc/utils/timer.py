import time
import datetime
import numpy as np
from functools import wraps

__all__ = ['timer']

def timer(text=None):
    ''' Decorator to print function's elapsed time '''
    def inner(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time.time()
            out = function(*args, **kwargs)
            end = time.time()
            elapsed = datetime.timedelta(seconds=int(np.round(end-start)))
            if text is not None:
                print(f"[{text.upper()}] - {function.__name__} - Elapsed time : {elapsed}")
            else:
                print(f"{function.__name__} - Elapsed time : {elapsed}")
            return out
        return wrapper
    return inner