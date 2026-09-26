# Reproducibility

By default, audiomentations draws random numbers from the global `random` and
`numpy.random` states. This keeps full backward compatibility (you can still
use `random.seed(...)` and `numpy.random.seed(...)`), but it means results can
vary depending on what else consumes those global states, and it does not play
well with multithreading.

For fully reproducible augmentation, pass a `seed` to your composition:

```python
from audiomentations import Compose, AddGaussianNoise, Gain, Shift

augment = Compose(
    [
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        Shift(p=0.5),
    ],
    seed=42,
)
```

When a composition is seeded:

* The same seed gives the same sequence of augmented outputs, every time.
* Each transform gets its own private random number generators, derived from
  the composition's seed and the transform's identity. Transforms do not
  interfere with each other, and adding or removing a transform does not
  change the random numbers drawn by the remaining transforms.
* The results are independent of the global `random` and `numpy.random`
  states, so other code (or other threads, as long as each thread uses its
  own composition instance) cannot disturb the sequence.

`seed` is also supported by `SomeOf` and `OneOf`, and you can (re)seed any
transform or composition after construction:

```python
augment.set_seed(123)   # reseeds the composition and all its transforms
transform.set_seed(7)   # seeds a single transform
transform.seed = None   # back to using the global RNGs
```

Note that a seeded composition instance is stateful: each call advances its
random number generators, so consecutive calls give different augmentations,
but the whole sequence is reproducible when the same seed is set again. If you
share one instance between threads, calls from different threads will
interleave nondeterministically - give each thread/worker its own seeded
instance instead.
