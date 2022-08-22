from .register import DATASETS
from .csv import *
from .patched import *
from .folder import *

__all__ = ['DATASETS', *DATASETS.registry_names]