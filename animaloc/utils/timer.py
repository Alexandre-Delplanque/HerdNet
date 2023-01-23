__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: November 23, 2022
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.1.0"


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