"""Backend-selected type configuration; native execution follows tensor dtype."""
from ._backend import BACKEND
if BACKEND == 'native':
    from .native._type_config import *
    from .native._type_config import __all__
else:
    from .wp_type_config import *
    from .wp_type_config import __all__
