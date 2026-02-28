import sys

import pytest

import infotheory_rs as ait


@pytest.mark.vm
def test_vm_symbol_exposure_and_config_smoke():
    if not ait.vm_enabled():
        assert not hasattr(ait, "NyxVmConfig")
        assert not hasattr(ait, "NyxVmEnvironment")
        pytest.skip("python extension built without vm feature")

    assert hasattr(ait, "NyxVmConfig")
    cfg = ait.NyxVmConfig()
    cfg.set_instance_id("pytest-vm")
    cfg.set_episode_steps(4)
    cfg.set_step_cost(-1)
    cfg.set_observation_bits(8)
    cfg.set_reward_bits(8)
    cfg.set_firecracker_config("/nonexistent/firecracker.json")

    with pytest.raises(RuntimeError):
        ait.NyxVmEnvironment(cfg)


def test_vm_enabled_returns_boolean():
    assert isinstance(ait.vm_enabled(), bool)
    if sys.platform != "linux":
        assert not ait.vm_enabled()

