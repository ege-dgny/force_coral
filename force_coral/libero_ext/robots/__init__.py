"""Register custom robot models into robosuite's ROBOT_CLASS_MAPPING."""

from force_coral.libero_ext.robots.on_the_ground_panda import OnTheGroundPanda  # noqa: F401
from force_coral.libero_ext.robots.my_mounted_panda import MyMountedPanda  # noqa: F401

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.setdefault("OnTheGroundPanda", SingleArm)
ROBOT_CLASS_MAPPING.setdefault("MyMountedPanda", SingleArm)
