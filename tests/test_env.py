import pytest


def test_kvtier_imports():
    import kvtier  # noqa: F401


@pytest.mark.gpu
def test_stack_versions():
    import lmcache
    import torch
    import vllm

    assert vllm.__version__.startswith("0.31.0")
    assert torch.__version__.startswith("2.13.0")
    assert lmcache.__version__ == "0.5.6" or lmcache.__version__.startswith("0.5.6")
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (7, 5)
