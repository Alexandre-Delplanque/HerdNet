from .register import LOSSES
from .ssim import *
from .focal import *

__all__ = ['LOSSES', *LOSSES.registry_names]