"""Pretrained-bfm semantics and inference contracts."""

from bfm_adapters.adapter import BFMAdapter
from bfm_adapters.batched_bfm import (
    BatchedBFM,
    DeviceBFMProvider,
    PreparedDeviceBFM,
    RobotStateView,
    SerialReferenceBFM,
)
from bfm_adapters.contracts import (
    DevicePDCommand,
    MOTION_COMMAND_ARRAY_FIELDS,
    MotionCommand,
    NamedTensors,
    PDCommand,
    RobotState,
    TensorSpec,
    BFMStateBatch,
)
from bfm_adapters.registry import (
    BFMInstanceConfig,
    available,
    get_family,
    instance_config_from_mapping,
    load_instance_config,
    register,
)

__all__ = [
    "DevicePDCommand",
    "DeviceBFMProvider",
    "MOTION_COMMAND_ARRAY_FIELDS",
    "MotionCommand",
    "NamedTensors",
    "PDCommand",
    "RobotState",
    "RobotStateView",
    "BatchedBFM",
    "PreparedDeviceBFM",
    "SerialReferenceBFM",
    "TensorSpec",
    "BFMAdapter",
    "BFMInstanceConfig",
    "BFMStateBatch",
    "available",
    "get_family",
    "instance_config_from_mapping",
    "load_instance_config",
    "register",
]

from bfm_adapters import families as families  # noqa: E402,F401  (registers built-ins)
