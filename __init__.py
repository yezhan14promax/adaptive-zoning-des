import os

_inner = os.path.join(os.path.dirname(__file__), "arm_sim")
if os.path.isdir(_inner) and _inner not in __path__:
    __path__.append(_inner)
