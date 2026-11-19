from __future__ import annotations

from niche_pipeline.deconvolution import _device_from_devices, _normalise_device, _resolve_training_device_kwargs


def test_resolve_training_device_kwargs_uses_explicit_gpu_device() -> None:
    kwargs = _resolve_training_device_kwargs({"accelerator": "gpu", "device": 3})
    assert kwargs == {"accelerator": "gpu", "device": [3]}


def test_resolve_training_device_kwargs_defaults_to_cpu_when_use_gpu_false() -> None:
    kwargs = _resolve_training_device_kwargs({"use_gpu": False})
    assert kwargs == {"accelerator": "cpu", "device": "auto"}


def test_device_from_devices_accepts_legacy_list() -> None:
    assert _device_from_devices([2]) == [2]


def test_device_from_devices_accepts_comma_string_and_takes_first() -> None:
    assert _device_from_devices("0,1") == [0]


def test_normalise_device_accepts_auto() -> None:
    assert _normalise_device("auto") == "auto"


def test_normalise_device_returns_list_for_gpu_zero() -> None:
    assert _normalise_device(0, accelerator="gpu") == [0]
