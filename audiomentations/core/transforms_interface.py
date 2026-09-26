from __future__ import annotations

import inspect
import random
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.utils import is_waveform_multichannel
from audiomentations.core.randomness import (
    LEGACY_GLOBAL_NUMPY_RANDOM,
    LEGACY_GLOBAL_PYTHON_RANDOM,
)
from audiomentations.core.serialization import (
    Serializable,
    SerializableMeta,
    get_shortest_class_fullname,
)
from audiomentations.core.utils import format_args


class MultichannelAudioNotSupportedException(Exception):
    pass


class MonoAudioNotSupportedException(Exception):
    pass


class WrongMultichannelAudioShape(Exception):
    pass


class CombinedMeta(SerializableMeta):
    pass


class BaseTransform(Serializable, metaclass=CombinedMeta):
    supports_mono = True
    supports_multichannel = False

    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.are_parameters_frozen = False
        self._seed = None
        # Randomness sources. When unseeded (the default), these delegate to
        # the global `random` and `numpy.random` states, which preserves the
        # historical behavior. When a seed is set, they become private
        # `random.Random` and `numpy.random.Generator` instances instead.
        self.random = LEGACY_GLOBAL_PYTHON_RANDOM
        self.rng = LEGACY_GLOBAL_NUMPY_RANDOM

    @property
    def seed(self):
        """The seed of this transform, or None if it uses the global RNGs."""
        return self._seed

    @seed.setter
    def seed(self, value):
        self.set_seed(value)

    def set_seed(self, seed):
        """
        Set the seed of this transform's private random number generators.
        Given the same seed, the transform will draw the same sequence of
        random numbers, independently of other transforms and of the global
        RNG state. Set to None to revert to using the global RNGs.
        """
        self._seed = seed
        if seed is None:
            self.random = LEGACY_GLOBAL_PYTHON_RANDOM
            self.rng = LEGACY_GLOBAL_NUMPY_RANDOM
        else:
            self.random = random.Random(seed)
            self.rng = np.random.default_rng(seed)
        return self

    def __repr__(self) -> str:
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return f"{self.__class__.__name__}({format_args(state)})"

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        return self.parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    @classmethod
    def is_serializable(cls) -> bool:
        return True

    def get_base_init_args(self) -> dict[str, Any]:
        """Returns base init args - p"""
        return {"p": self.p}

    def get_transform_init_args(self) -> dict[str, Any]:
        """Exclude seed from init args during serialization"""
        init_signature = inspect.signature(self.__init__)
        args = {}
        for k, _ in init_signature.parameters.items():
            attr = getattr(self, k, None)
            if attr is not None:
                args[k] = attr
            else:
                warnings.warn(
                    f"Warning: attribute {k} is not found in the transform definition and it won't be printed."
                )
        args.pop("seed", None)  # Remove seed from args
        return args


class BaseWaveformTransform(BaseTransform):
    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        raise NotImplementedError

    def is_multichannel(self, samples):
        return is_waveform_multichannel(samples)

    def __call__(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        if samples.dtype == np.float64:
            warnings.warn(
                "Warning: input samples dtype is np.float64. Converting to np.float32"
            )
            samples = np.float32(samples)
        if not self.are_parameters_frozen or self.parameters["should_apply"] is None:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and len(samples) > 0:
            if self.is_multichannel(samples):
                # Note: We multiply by 8 here to allow big batches of very short audio
                if samples.shape[0] > samples.shape[1] * 8:
                    raise WrongMultichannelAudioShape(
                        "Multichannel audio must have channels first, not channels"
                        " last. In other words, the shape must be (channels, samples),"
                        " not (samples, channels). See"
                        " https://iver56.github.io/audiomentations/guides/multichannel_audio_array_shapes/"
                        " for more info."
                    )
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio. In other"
                        " words, a 1-dimensional input ndarray was expected, but the"
                        " input had more than 1 dimension.".format(
                            self.__class__.__name__
                        )
                    )
            elif not self.supports_mono:
                raise MonoAudioNotSupportedException(
                    "{} only supports multichannel audio, not mono audio".format(
                        self.__class__.__name__
                    )
                )
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        self.parameters["should_apply"] = self.random.random() < self.p

    @classmethod
    def get_class_fullname(cls) -> str:
        return get_shortest_class_fullname(cls)

    def to_dict_private(self) -> dict[str, Any]:
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state
