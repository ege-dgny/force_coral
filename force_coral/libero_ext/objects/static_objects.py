"""
Custom static (fixture) objects for force_coral tasks.
"""

import os
import re
import pathlib

import numpy as np
from robosuite.models.objects import MujocoXMLObject

import force_coral
from libero.libero.envs.base_object import register_object


class GenericAssetObject(MujocoXMLObject):
    """
    Load a single XML from force_coral/data/assets/<obj_name>.xml with no
    articulation.  Falls back to LIBERO's own asset directory when the file
    is not found in force_coral (e.g. ``wall.xml`` ships with LIBERO).
    """

    def __init__(self, name, obj_name, joints=None):
        asset_dir = force_coral.get_data_path("assets")
        xml_path = os.path.join(asset_dir, f"{obj_name}.xml")

        # Fall back to LIBERO's assets for original objects (e.g. wall.xml)
        if not os.path.exists(xml_path):
            import libero.libero as _libero_pkg
            libero_assets = os.path.join(
                os.path.dirname(os.path.abspath(_libero_pkg.__file__)), "assets"
            )
            xml_path = os.path.join(libero_assets, f"{obj_name}.xml")

        super().__init__(
            xml_path,
            name=name,
            joints=joints or [],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}


@register_object
class Wall(GenericAssetObject):
    """Original wall fixture – asset lives in LIBERO's assets/wall.xml."""

    def __init__(self, name="wall", obj_name="wall", joints=None):
        super().__init__(name=name, obj_name=obj_name, joints=[])
        self.z_on_table = 0.0


@register_object
class Wall2(GenericAssetObject):
    def __init__(self, name="wall2", obj_name="wall2", joints=None):
        super().__init__(name=name, obj_name=obj_name, joints=[])
        self.z_on_table = 0.0
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"


# Ensure both Wall and Wall2 are in OBJECTS_DICT even if the decorator
# registration is insufficient.
try:
    from libero.libero.envs.base_object import OBJECTS_DICT as _OBJECTS_DICT
    _OBJECTS_DICT.setdefault("wall", Wall)
    _OBJECTS_DICT.setdefault("wall2", Wall2)
except Exception:
    pass
