"""
Custom predicates: OnSide, OnSideWithRecentSupport.
"""

import numpy as np
import robosuite.utils.transform_utils as T
from libero.libero.envs.predicates.base_predicates import UnaryAtomic, BinaryAtomic


class OnSide(UnaryAtomic):
    """
    Returns True if the object's local z-axis is approximately horizontal,
    i.e., the object is flipped onto one of its side faces.

    Heuristic: |R[2,2]| < cos(theta_thresh), with theta_thresh ~ 88 deg.
    """

    def __init__(self, theta_deg: float = 88.0, require_floor_contact: bool = True):
        super().__init__()
        self.cos_thresh = np.cos(np.deg2rad(theta_deg))
        self.require_floor_contact = require_floor_contact

    def __call__(self, arg):
        quat = arg.get_geom_state()["quat"]
        R = T.quat2mat(T.convert_quat(quat, to="xyzw"))
        z_world = abs(R[2, 2])
        on_side = z_world < self.cos_thresh

        if not self.require_floor_contact:
            return bool(on_side)

        try:
            floor_state = arg.env.object_states_dict.get("floor", None)
            if floor_state is None:
                return bool(on_side)
            return bool(on_side and arg.check_contact(floor_state))
        except Exception:
            return bool(on_side)


class OnSideWithRecentSupport(BinaryAtomic):
    """
    True iff:
    - The object is currently on its side (orientation-only), AND
    - The environment has latched that the wall was used as support during the
      most recent flip onto its side.
    """

    def __init__(self, theta_deg: float = 88.0):
        super().__init__()
        self.cos_thresh = np.cos(np.deg2rad(theta_deg))

    def __call__(self, obj_state, wall_state):
        quat = obj_state.get_geom_state()["quat"]
        R = T.quat2mat(T.convert_quat(quat, to="xyzw"))
        z_world = abs(R[2, 2])
        on_side_now = z_world < self.cos_thresh

        env = getattr(obj_state, "env", None)
        if env is None:
            return False

        key = (obj_state.object_name, wall_state.object_name)
        try:
            support_latched = env._support_granted.get(key, False)
        except Exception:
            support_latched = None

        if support_latched is None:
            try:
                return bool(on_side_now and obj_state.check_contact(wall_state))
            except Exception:
                return bool(on_side_now)

        return bool(on_side_now and support_latched)
