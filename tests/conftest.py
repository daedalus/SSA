"""Shared fixtures for the SparseAttention test suite."""
import pytest
import torch

from sparse_attention import SSAConfig


@pytest.fixture(autouse=True)
def _seed():
    """Every test gets a fresh, deterministic seed so failures are
    reproducible and tests don't leak randomness state into each other."""
    torch.manual_seed(42)
    yield


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def base_cfg():
    """Small, fast config used by most correctness tests. Mirrors the
    config used in the original __main__ test block."""
    return SSAConfig(
        d_model=128, num_heads=2, num_neighbors=32,
        max_num_hashes=3, window_size=4, num_global_tokens=2,
        dropout=0.0, causal=False,
    )


@pytest.fixture
def causal_cfg():
    return SSAConfig(
        d_model=64, num_heads=2, num_neighbors=16,
        max_num_hashes=2, window_size=3, num_global_tokens=1,
        causal=True,
    )


@pytest.fixture
def gqa_cfg():
    return SSAConfig(
        d_model=64, num_heads=4, num_kv_heads=2,
        num_neighbors=16, window_size=4, num_global_tokens=2,
        max_num_hashes=3,
    )


@pytest.fixture
def mqa_cfg():
    return SSAConfig(
        d_model=64, num_heads=4, num_kv_heads=1,
        num_neighbors=16, window_size=4, num_global_tokens=2,
        max_num_hashes=3,
    )
