"""Skeleton tests for ALVC — ensures package imports and configs are coherent.

Real unit tests added as halting + connector implementations land.
"""
from sgod.policies.alvc.connector import ALVCConfig
from sgod.policies.alvc.halting import HaltingConfig


def test_alvc_config_defaults_sane():
    cfg = ALVCConfig()
    assert cfg.k_min < cfg.k_max
    assert cfg.k_max <= cfg.n_visual_patches
    assert cfg.d_v > 0 and cfg.d_lm > 0


def test_halting_config_defaults_sane():
    cfg = HaltingConfig()
    assert 0 < cfg.lambda_p < 1
    assert cfg.beta > 0
    assert cfg.k_min < cfg.k_max


