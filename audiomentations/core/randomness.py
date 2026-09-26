"""
Utilities for reproducible randomness in audiomentations.

By default (no seed set), transforms draw random numbers from the global
``random`` and ``numpy.random`` states, exactly like older versions of
audiomentations. When a seed is assigned to a transform or a composition,
the transform instead draws from its own private random number generators,
which makes the results reproducible and independent of other transforms,
other threads and any global RNG state.
"""

import hashlib
import random

import numpy as np


class LegacyGlobalPythonRandom:
    """
    Facade that exposes the ``random.Random`` API while delegating to the
    global ``random`` module state. Used as the randomness source of
    unseeded transforms, so the historical behavior (including respect for
    ``random.seed()``) is preserved.
    """

    def __getattr__(self, name):
        return getattr(random, name)

    def __repr__(self):
        return "LegacyGlobalPythonRandom()"


class LegacyGlobalNumpyRandom:
    """
    Facade that exposes (a subset of) the ``numpy.random.Generator`` API
    while delegating to the legacy global numpy random state. Used as the
    randomness source of unseeded transforms, so the historical behavior
    (including respect for ``numpy.random.seed()``) is preserved.
    """

    def __getattr__(self, name):
        return getattr(np.random, name)

    def integers(self, low, high=None, size=None, dtype=int, endpoint=False):
        if endpoint:
            if high is None:
                low, high = 0, low + 1
            else:
                high = high + 1
        return np.random.randint(low, high=high, size=size, dtype=dtype)

    def __repr__(self):
        return "LegacyGlobalNumpyRandom()"


# Shared stateless instances. They carry no state of their own, so transforms
# that reference them remain picklable and deepcopy-able.
LEGACY_GLOBAL_PYTHON_RANDOM = LegacyGlobalPythonRandom()
LEGACY_GLOBAL_NUMPY_RANDOM = LegacyGlobalNumpyRandom()


def derive_seed(*parts) -> int:
    """
    Derive a deterministic non-negative integer seed from the given parts.

    Used to give each transform in a composition its own independent seed,
    derived from the composition's seed and the transform's identity (class
    name and occurrence index), so that adding or removing another transform
    does not change the random stream of the remaining transforms.

    Note: this intentionally does not use the built-in ``hash()``, whose
    result for strings varies between Python processes.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\x00")
    return int.from_bytes(hasher.digest(), byteorder="little", signed=False)
