from pathlib import Path

import pytest

from src.training.build_fibre_finetune_set import split_real


def _images(n: int) -> list[Path]:
    return [Path(f"real_{i:03d}.jpg") for i in range(n)]


def test_split_real_holds_out_requested_fraction_without_overlap() -> None:
    train, test = split_real(_images(60), holdout_fraction=0.25, seed=0)

    assert len(test) == 15
    assert len(train) == 45
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(_images(60))


def test_split_real_is_deterministic_for_a_seed_and_input_order() -> None:
    images = _images(60)

    assert split_real(images, 0.25, seed=3) == split_real(list(reversed(images)), 0.25, seed=3)
    assert split_real(images, 0.25, seed=3) != split_real(images, 0.25, seed=4)


def test_split_real_keeps_at_least_one_test_image() -> None:
    _, test = split_real(_images(2), holdout_fraction=0.1, seed=0)

    assert len(test) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_split_real_rejects_out_of_range_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        split_real(_images(10), fraction, seed=0)
