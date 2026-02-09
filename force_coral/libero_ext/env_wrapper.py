"""
SegmentationRenderEnv -- extends LIBERO's OffScreenRenderEnv with per-instance
segmentation mask support.
"""

import numpy as np
import cv2
import matplotlib.cm as cm

from libero.libero.envs.env_wrapper import OffScreenRenderEnv


class SegmentationRenderEnv(OffScreenRenderEnv):
    """
    This wrapper additionally generates the segmentation mask of objects,
    which is useful for comparing attention and for FoundationPose integration.
    """

    def __init__(
        self,
        camera_segmentations="instance",
        camera_heights=128,
        camera_widths=128,
        **kwargs,
    ):
        assert camera_segmentations is not None
        kwargs["camera_segmentations"] = camera_segmentations
        kwargs["camera_heights"] = camera_heights
        kwargs["camera_widths"] = camera_widths
        self.segmentation_id_mapping = {}
        self.instance_to_id = {}
        self.segmentation_robot_id = None
        super().__init__(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        self.segmentation_id_mapping = {}

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name == "Panda0":
                self.segmentation_robot_id = i

        for i, instance_name in enumerate(list(self.env.model.instances_to_ids.keys())):
            if instance_name not in ["Panda0", "RethinkMount0", "PandaGripper0"]:
                self.segmentation_id_mapping[i] = instance_name

        self.instance_to_id = {
            v: k + 1 for k, v in self.segmentation_id_mapping.items()
        }
        return obs

    def get_segmentation_instances(self, segmentation_image):
        seg_img_dict = {}
        segmentation_image[segmentation_image > self.segmentation_robot_id] = (
            self.segmentation_robot_id + 1
        )
        seg_img_dict["robot"] = segmentation_image * (
            segmentation_image == self.segmentation_robot_id + 1
        )
        for seg_id, instance_name in self.segmentation_id_mapping.items():
            seg_img_dict[instance_name] = segmentation_image * (
                segmentation_image == seg_id + 1
            )
        return seg_img_dict

    def get_segmentation_of_interest(self, segmentation_image):
        ret_seg = np.zeros_like(segmentation_image)
        for obj in self.obj_of_interest:
            ret_seg[segmentation_image == self.instance_to_id[obj]] = 1.0
        ret_seg[segmentation_image == 0] = -1.0
        return ret_seg

    # ------------------------------------------------------------------
    # FORTE: contact wrench access
    # ------------------------------------------------------------------

    def get_body_wrench(self, body_name: str) -> np.ndarray:
        """Return the 6D external contact wrench on a body in world frame.

        Uses MuJoCo's ``cfrc_ext`` which is computed automatically from the
        contact solver — no sensor XML modifications required.

        Parameters
        ----------
        body_name : str
            MuJoCo body name (e.g. ``"block_1_main"``).

        Returns
        -------
        np.ndarray, shape (6,)
            Wrench in **[Fx, Fy, Fz, τx, τy, τz]** order (force-first).
            MuJoCo stores ``cfrc_ext`` as [τ(3), F(3)]; we reorder here.
        """
        bid = self.env.sim.model.body_name2id(body_name)
        cfrc = self.env.sim.data.cfrc_ext[bid]  # MuJoCo: [τx,τy,τz, Fx,Fy,Fz]
        return np.concatenate([cfrc[3:6], cfrc[0:3]])  # → [F(3), τ(3)]

    def segmentation_to_rgb(self, seg_im, random_colors=False):
        seg_im = np.mod(seg_im, 256)
        if random_colors:
            from robosuite.utils.mjcf_utils import randomize_colors
            colors = randomize_colors(N=256, bright=True)
            return (255.0 * colors[seg_im]).astype(np.uint8)
        else:
            rstate = np.random.RandomState(seed=2)
            inds = np.arange(256)
            rstate.shuffle(inds)
            seg_img = (
                np.array(255.0 * cm.rainbow(inds[seg_im], 10))
                .astype(np.uint8)[..., :3]
                .astype(np.uint8)
                .squeeze(-2)
            )
            print(seg_img.shape)
            cv2.imshow("Seg Image", seg_img[::-1])
            cv2.waitKey(1)
            return seg_img
