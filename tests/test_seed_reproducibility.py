import random
import threading

import numpy as np
import pytest

from audiomentations import (
    AddGaussianNoise,
    Compose,
    Gain,
    OneOf,
    PolarityInversion,
    Shift,
    SomeOf,
    TimeMask,
)


def make_samples(num_samples=16000, num_channels=None):
    shape = (num_channels, num_samples) if num_channels else (num_samples,)
    return np.random.uniform(low=-0.2, high=0.2, size=shape).astype(np.float32)


def make_compose(seed):
    return Compose(
        [
            Gain(min_gain_db=-6, max_gain_db=6, p=0.8),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.8),
            Shift(min_shift=-0.5, max_shift=0.5, p=0.8),
            TimeMask(min_band_part=0.0, max_band_part=0.2, p=0.8),
        ],
        seed=seed,
    )


class TestSeedReproducibility:
    def test_same_seed_same_result(self):
        samples = make_samples()
        augment1 = make_compose(seed=42)
        augment2 = make_compose(seed=42)
        for _ in range(5):
            result1 = augment1(samples=samples, sample_rate=16000)
            result2 = augment2(samples=samples, sample_rate=16000)
            assert np.array_equal(result1, result2)

    def test_different_seed_different_result(self):
        samples = make_samples()
        result1 = make_compose(seed=42)(samples=samples, sample_rate=16000)
        result2 = make_compose(seed=123)(samples=samples, sample_rate=16000)
        assert not np.array_equal(result1, result2)

    def test_result_independent_of_global_rng_state(self):
        samples = make_samples()
        np.random.seed(999)
        random.seed(999)
        result1 = make_compose(seed=42)(samples=samples, sample_rate=16000)
        np.random.seed(1)
        random.seed(1)
        result2 = make_compose(seed=42)(samples=samples, sample_rate=16000)
        assert np.array_equal(result1, result2)

    def test_one_of_reproducible(self):
        samples = make_samples()
        transforms = [Gain(p=1.0), PolarityInversion(p=1.0), Shift(p=1.0)]
        augment1 = OneOf([t for t in transforms], seed=7)
        augment2 = OneOf([type(t)(**t.get_transform_init_args()) for t in transforms], seed=7)
        for _ in range(5):
            assert np.array_equal(
                augment1(samples=samples, sample_rate=16000),
                augment2(samples=samples, sample_rate=16000),
            )

    def test_some_of_reproducible(self):
        samples = make_samples()
        make_transforms = lambda: [Gain(p=1.0), PolarityInversion(p=1.0), Shift(p=1.0)]
        augment1 = SomeOf((1, 2), make_transforms(), seed=7)
        augment2 = SomeOf((1, 2), make_transforms(), seed=7)
        for _ in range(5):
            assert np.array_equal(
                augment1(samples=samples, sample_rate=16000),
                augment2(samples=samples, sample_rate=16000),
            )

    def test_shuffle_reproducible(self):
        samples = make_samples()
        augment1 = Compose([Gain(p=1.0), PolarityInversion(p=1.0)], shuffle=True, seed=5)
        augment2 = Compose([Gain(p=1.0), PolarityInversion(p=1.0)], shuffle=True, seed=5)
        for _ in range(5):
            assert np.array_equal(
                augment1(samples=samples, sample_rate=16000),
                augment2(samples=samples, sample_rate=16000),
            )

    def test_set_seed_after_construction(self):
        samples = make_samples()
        augment = make_compose(seed=None)
        augment.set_seed(42)
        result1 = augment(samples=samples, sample_rate=16000)
        augment.set_seed(42)
        result2 = augment(samples=samples, sample_rate=16000)
        assert np.array_equal(result1, result2)

    def test_seed_property(self):
        augment = make_compose(seed=42)
        assert augment.seed == 42
        assert all(t.seed is not None for t in augment.transforms)


class TestTransformIsolation:
    def test_removing_a_transform_does_not_affect_the_others(self):
        samples = make_samples()
        compose_full = Compose(
            [
                Gain(p=1.0),
                PolarityInversion(p=1.0),
                Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
            ],
            seed=7,
        )
        compose_reduced = Compose(
            [
                Gain(p=1.0),
                Shift(min_shift=-0.5, max_shift=0.5, p=1.0),
            ],
            seed=7,
        )
        compose_full(samples=samples, sample_rate=16000)
        compose_reduced(samples=samples, sample_rate=16000)
        assert (
            compose_full.transforms[0].parameters
            == compose_reduced.transforms[0].parameters
        )
        assert (
            compose_full.transforms[2].parameters
            == compose_reduced.transforms[1].parameters
        )

    def test_transforms_of_same_class_get_distinct_seeds(self):
        compose = Compose([Gain(p=1.0), Gain(p=1.0)], seed=7)
        assert compose.transforms[0].seed != compose.transforms[1].seed

    def test_nested_composition_isolated(self):
        samples = make_samples()
        augment1 = Compose(
            [OneOf([Gain(p=1.0), PolarityInversion(p=1.0)]), Shift(p=1.0)], seed=11
        )
        augment2 = Compose(
            [OneOf([Gain(p=1.0), PolarityInversion(p=1.0)]), Shift(p=1.0)], seed=11
        )
        for _ in range(3):
            assert np.array_equal(
                augment1(samples=samples, sample_rate=16000),
                augment2(samples=samples, sample_rate=16000),
            )


class TestEdgeCases:
    @pytest.mark.parametrize("num_samples", [1, 7, 100])
    def test_very_short_audio(self, num_samples):
        samples = make_samples(num_samples=num_samples)
        augment = make_compose(seed=42)
        result1 = augment(samples=samples, sample_rate=16000)
        result2 = make_compose(seed=42)(samples=samples, sample_rate=16000)
        assert np.array_equal(result1, result2)

    def test_mono_and_multichannel(self):
        for samples in (make_samples(), make_samples(num_channels=2)):
            augment = Compose(
                [Gain(p=1.0), AddGaussianNoise(p=1.0), PolarityInversion(p=1.0)],
                seed=42,
            )
            result1 = augment(samples=samples, sample_rate=16000)
            result2 = Compose(
                [Gain(p=1.0), AddGaussianNoise(p=1.0), PolarityInversion(p=1.0)],
                seed=42,
            )(samples=samples, sample_rate=16000)
            assert np.array_equal(result1, result2)

    @pytest.mark.parametrize("sample_rate", [1000, 8000, 44100, 192000])
    def test_extreme_sample_rates(self, sample_rate):
        samples = make_samples(num_samples=max(100, sample_rate // 10))
        augment = make_compose(seed=42)
        result1 = augment(samples=samples, sample_rate=sample_rate)
        result2 = make_compose(seed=42)(samples=samples, sample_rate=sample_rate)
        assert np.array_equal(result1, result2)

    def test_empty_audio_does_not_crash(self):
        samples = np.zeros(0, dtype=np.float32)
        augment = make_compose(seed=42)
        result = augment(samples=samples, sample_rate=16000)
        assert len(result) == 0


class TestMultithreading:
    def test_per_thread_instances_are_reproducible(self):
        samples = make_samples()
        num_threads = 8
        results = [None] * num_threads

        def worker(index):
            augment = make_compose(seed=2024)
            results[index] = [
                augment(samples=samples, sample_rate=16000) for _ in range(4)
            ]

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(num_threads)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        reference = results[0]
        for i in range(1, num_threads):
            for j in range(4):
                assert np.array_equal(results[i][j], reference[j])


class TestBackwardCompatibility:
    def test_unseeded_compose_works(self):
        samples = make_samples()
        augment = Compose([Gain(p=1.0), AddGaussianNoise(p=1.0)])
        result = augment(samples=samples, sample_rate=16000)
        assert result.shape == samples.shape

    def test_unseeded_transform_respects_global_seeds(self):
        samples = make_samples()
        np.random.seed(0)
        random.seed(0)
        result1 = Gain(p=1.0)(samples=samples, sample_rate=16000)
        np.random.seed(0)
        random.seed(0)
        result2 = Gain(p=1.0)(samples=samples, sample_rate=16000)
        assert np.array_equal(result1, result2)

    def test_seeded_transform_does_not_consume_global_randomness(self):
        samples = make_samples()
        np.random.seed(123)
        random.seed(123)
        expected_numpy = np.random.random()
        expected_python = random.random()
        np.random.seed(123)
        random.seed(123)
        Gain(p=1.0).set_seed(42)(samples=samples, sample_rate=16000)
        assert np.random.random() == expected_numpy
        assert random.random() == expected_python
