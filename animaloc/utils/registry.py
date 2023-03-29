__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 29, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.2.0"


import sys

from typing import Callable, Optional

class Registry:
    ''' Registry to map str to classes. Object can then be 
    called from the registry 

    Args:
        name (str): registry name.
        module_key (str, optional): module key, used for updating its __all__ variable. 
            Defaults to None.
    '''

    def __init__(self, name: str, module_key: Optional[str] = None) -> None:
        self.name = name
        self.module_key = module_key

        self._registered_objects = {}
    
    def register(self) -> Callable:
        ''' Register a class '''

        def _register(cls):
            self._registered_objects[cls.__name__] = cls
            if self.module_key is not None:
                sys.modules[self.module_key].__all__.append(cls.__name__)
            return cls

        return _register
    
    @property
    def registry_names(self) -> list:
        return list(self._registered_objects.keys())
    
    def __getitem__(self, key: str) -> Callable:
        return self._registered_objects[key]
    
    def __len__(self) -> int:
        return len(self._registered_objects)
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name=\'{self.name}\', ' \
            f'objects={self._registered_objects})'