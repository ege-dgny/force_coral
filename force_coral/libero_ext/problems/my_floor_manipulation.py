from robosuite.utils.mjcf_utils import new_site
import numpy as np
import robosuite.utils.transform_utils as T
from collections import deque

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class My_Floor_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "floor"
        self.visualization_sites_list = []
        self.floor_offset = (0, 0, -0.035)

        self.z_offset = -0.025
        # configurable N-step success hold to debounce transient contacts
        self._success_hold_steps = kwargs.pop("success_hold_steps", 5)
        self._success_hold_counter = 0
        # Control-step timing and temporal support windows (all in control steps)
        self._control_freq = kwargs.get("control_freq", 20)
        self._support_pre_steps = kwargs.pop("support_pre_steps", 20)
        self._support_post_steps = kwargs.pop("support_post_steps", 10)
        # Orientation threshold for OnSide (deg -> cos threshold)
        self._onside_cos_thresh = np.cos(np.deg2rad(88.0))
        # Temporal support tracking state (initialized here, finalized after super init)
        self._control_step_counter = 0
        self._contact_hist = {}
        self._on_side_prev = {}
        self._first_on_side_step = {}
        self._support_granted = {}
        self._support_pending_end = {}
        kwargs.update(
            {"robots": [f"OnTheGround{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.floor_offset})
        kwargs.update({"arena_type": "floor"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_floor_base_style.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "light-gray",
                        "wall_style": "light-gray-plaster",
                    }
                }
            )

        # Optional: per-object property overrides (e.g., size, density) supplied at env construction
        # Example: {"block_1": {"size": (0.05, 0.05, 0.05), "density": 500}}
        self._object_overrides = kwargs.pop("object_overrides", {})

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "floor":
                continue

            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                override_kwargs = self._object_overrides.get(object_name, {})
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name, **override_kwargs
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "floor" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.floor_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def _check_success(self):
        """
        Check if the goal is achieved with an N-step hold to avoid single-frame blips.
        """
        goal_state = self.parsed_problem["goal_state"]
        instant = True
        for state in goal_state:
            instant = self._eval_predicate(state) and instant

        if instant:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        return self._success_hold_counter >= self._success_hold_steps

    def _reset_internal(self):
        super()._reset_internal()
        # Reset debounce counter on each environment reset
        self._success_hold_counter = 0
        # Reset temporal support tracking state
        self._control_step_counter = 0
        self._contact_hist = {}
        self._on_side_prev = {}
        self._first_on_side_step = {}
        self._support_granted = {}
        self._support_pending_end = {}

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def _update_support_histories(self):
        """Update per control-step histories for wall-support flipping.

        We track:
        - Contact(obj, fixture) booleans in a fixed-length deque
        - Orientation-based OnSide status per object
        - First flip step (rising edge) per object
        - Latched support_granted per (obj, fixture), granted if contact occurs
          in a window around the first flip. Pending post-window is cleared after expiry.
        When an object leaves the side pose, we clear flip markers and latches so a
        subsequent flip must again show support.
        """
        # Count this control step
        self._control_step_counter += 1

        obj_names = list(getattr(self, "obj_of_interest", []))
        fix_names = list(getattr(self, "fixtures_dict", {}).keys())
        maxlen = max(self._support_pre_steps + self._support_post_steps, 1)

        # Orientation-only OnSide now for objects of interest
        onside_now = {}
        for obj in obj_names:
            try:
                quat = self.object_states_dict[obj].get_geom_state()["quat"]
                R = T.quat2mat(T.convert_quat(quat, to="xyzw"))
                z_world = abs(R[2, 2])
                onside_now[obj] = (z_world < self._onside_cos_thresh)
            except Exception:
                onside_now[obj] = False

        # Append contact history for each obj-fixture pair
        for obj in obj_names:
            for fix in fix_names:
                key = (obj, fix)
                if key not in self._contact_hist:
                    self._contact_hist[key] = deque(maxlen=maxlen)
                    self._support_granted[key] = False
                    self._support_pending_end[key] = None
                try:
                    c = self.object_states_dict[obj].check_contact(self.object_states_dict[fix])
                except Exception:
                    c = False
                self._contact_hist[key].append(bool(c))

        # Rising edge detection and support latching
        for obj in obj_names:
            prev = self._on_side_prev.get(obj, False)
            cur = onside_now[obj]
            if (not prev) and cur:
                # first flip moment
                self._first_on_side_step[obj] = self._control_step_counter
                for fix in fix_names:
                    key = (obj, fix)
                    hist = list(self._contact_hist.get(key, []))
                    pre_ok = any(hist[-self._support_pre_steps:]) if hist else False
                    if pre_ok:
                        self._support_granted[key] = True
                        self._support_pending_end[key] = None
                    else:
                        self._support_granted[key] = False
                        self._support_pending_end[key] = (
                            self._control_step_counter + self._support_post_steps
                        )
            elif prev and (not cur):
                # left side pose -> clear for a new attempt
                self._first_on_side_step[obj] = None
                for fix in fix_names:
                    key = (obj, fix)
                    self._support_granted[key] = False
                    self._support_pending_end[key] = None

            # While pending post-window, grant on contact or clear after expiry
            for fix in fix_names:
                key = (obj, fix)
                end_step = self._support_pending_end.get(key)
                if end_step is not None:
                    if self._control_step_counter <= end_step:
                        if self._contact_hist.get(key) and self._contact_hist[key][-1]:
                            self._support_granted[key] = True
                            self._support_pending_end[key] = None
                    else:
                        self._support_pending_end[key] = None

            self._on_side_prev[obj] = cur

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)
        # update temporal tracking exactly once per env.step()
        self._update_support_histories()
        return reward, done, info

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.8965773716836134, 5.216182733499864e-07, 0.65],
            quat=[
                0.6182166934013367,
                0.3432307541370392,
                0.3432314395904541,
                0.6182177066802979,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 0.65], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
