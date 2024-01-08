"""System, hardware, and environment diagnostics for Kinesis."""

import platform
import sys
import torch
import numpy as np
import mujoco
from dm_control import suite


def run_doctor() -> None:
    """Probe hardware, precision, platform, and environment dependencies."""
    print("=== Kinesis Environment Diagnostics ===")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Platform: {platform.system()} ({platform.machine()})")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")
    print(f"MuJoCo Version: {mujoco.__version__}")

    # Check hardware accelerators
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name()})"
    elif torch.backends.mps.is_available():
        device = "mps (Apple Silicon Metal Performance Shaders)"
    print(f"Compute Device: {device}")

    # Check precision and numerical tolerances
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    print(f"Default Float Tensor: {x.dtype}, device={x.device}")

    # Verify dm_control suite tasks
    tasks = [("cartpole", "swingup"), ("walker", "walk"), ("finger", "spin")]
    print("\nProbing DeepMind Control Suite benchmark tasks:")
    for domain_name, task_name in tasks:
        try:
            env = suite.load(domain_name, task_name)
            time_step = env.reset()
            obs_keys = list(time_step.observation.keys())
            action_spec = env.action_spec()
            print(f"  [PASS] {domain_name}/{task_name}:")
            print(f"         Action shape: {action_spec.shape}, range: [{action_spec.minimum[0]:.1f}, {action_spec.maximum[0]:.1f}]")
            print(f"         Observation keys: {obs_keys}")
        except Exception as e:
            print(f"  [FAIL] {domain_name}/{task_name}: {e}")
            sys.exit(1)

    print("\nAll required core environments and dependencies verified successfully.")


if __name__ == "__main__":
    run_doctor()
