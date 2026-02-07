"""
Basic primitive objects used in custom tasks.

Block defaults:
- size: half-extents in meters (sx, sy, sz)
- density: kg/m^3 (MuJoCo / SI units)
- friction: (sliding, torsional, rolling) coefficients.
"""

import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.models.objects import MujocoXMLObject

try:
    from robosuite.models.objects import BoxObject
except Exception:
    from robosuite.models.objects.primitive import BoxObject

import force_coral
from libero.libero.envs.base_object import register_object


def _resolve_texture_path(texture_file: str) -> str:
    """Resolve texture file path to an absolute path if possible."""
    if not texture_file:
        return None
    texture_file = str(texture_file)
    if os.path.isabs(texture_file) and os.path.exists(texture_file):
        return texture_file
    if os.path.exists(texture_file):
        return os.path.abspath(texture_file)
    # Try force_coral's bundled textures
    fc_bundled = Path(force_coral.get_data_path("assets")) / "textures" / texture_file
    if fc_bundled.exists():
        return str(fc_bundled)
    # Try LIBERO's bundled textures
    try:
        from libero.libero import get_libero_path
        libero_bundled = Path(get_libero_path("assets")) / "textures" / texture_file
        if libero_bundled.exists():
            return str(libero_bundled)
    except Exception:
        pass
    return texture_file


def _get_asset_element(model):
    """Return the object's <asset> element, creating it if missing."""
    asset_el = getattr(model, "asset", None)
    if asset_el is not None:
        return asset_el
    root = getattr(model, "root", None)
    if root is not None:
        asset_el = root.find("asset")
        if asset_el is None:
            asset_el = ET.SubElement(root, "asset")
        try:
            setattr(model, "asset", asset_el)
        except Exception:
            pass
        return asset_el
    return None


def _apply_texture_material(model, texture_file: str, texrepeat=(1, 1), texuniform=False, material_name: str = None):
    """Inject texture+material into asset and apply to the model's visual geoms."""
    if not texture_file:
        return
    texture_path = _resolve_texture_path(texture_file)

    mat_name = material_name or f"mat-{getattr(model, 'name', 'primitive')}"
    tex_name = f"tex-{getattr(model, 'name', 'primitive')}"

    asset_el = _get_asset_element(model)
    if asset_el is None:
        return

    for el in list(asset_el.findall("texture")):
        if el.get("name") == tex_name:
            asset_el.remove(el)
    for el in list(asset_el.findall("material")):
        if el.get("name") == mat_name:
            asset_el.remove(el)

    tex_el = ET.Element("texture")
    tex_el.set("name", tex_name)
    tex_el.set("type", "2d")
    tex_el.set("file", texture_path)

    u, v = texrepeat
    mat_el = ET.Element("material")
    mat_el.set("name", mat_name)
    mat_el.set("texture", tex_name)
    mat_el.set("texrepeat", f"{float(u)} {float(v)}")
    mat_el.set("texuniform", "true" if texuniform else "false")

    asset_el.append(tex_el)
    asset_el.append(mat_el)

    try:
        obj_el = model.get_obj() if hasattr(model, "get_obj") else None
        vis_names = set(getattr(model, "visual_geoms", []) or [])
        if obj_el is not None and vis_names:
            for g in list(obj_el.findall("geom")):
                if g.get("name") in vis_names:
                    g.set("material", mat_name)
                    if "rgba" in g.attrib:
                        try:
                            del g.attrib["rgba"]
                        except Exception:
                            pass
            return
    except Exception:
        pass

    if hasattr(model, "worldbody") and model.worldbody is not None:
        for g in model.worldbody.findall(".//geom"):
            grp = g.get("group")
            if grp == "1" or grp is None:
                g.set("material", mat_name)
                if "rgba" in g.attrib:
                    try:
                        del g.attrib["rgba"]
                    except Exception:
                        pass


# ── Resolve default texture path via LIBERO assets ─────────────────────
def _default_block_texture():
    """Return the default wood-plank texture path, checking LIBERO assets."""
    try:
        from libero.libero import get_libero_path
        p = os.path.join(get_libero_path("assets"), "textures", "seamless_wood_planks_floor.png")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return None


@register_object
class Block(BoxObject):
    """Solid cube-like object. Category name 'block'."""

    def __init__(
        self,
        name,
        size=(0.08, 0.08, 0.08),
        rgba=(0, 0, 1, 1.0),
        joints="default",
        density=100,
        friction=(1.0, 0.0, 0.0),
        texture_file=None,
        texrepeat=(16, 16),
        **kwargs,
    ):
        # Resolve default texture lazily
        if texture_file is None:
            texture_file = _default_block_texture()
        super().__init__(
            name=name,
            size=np.array(size, dtype=float),
            rgba=np.array(rgba, dtype=float),
            joints=joints,
            density=density,
            **kwargs,
        )
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "block"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class Card(BoxObject):
    """Thin, card-like rectangular object. Category name 'card'."""

    def __init__(
        self,
        name,
        size=(0.2, 0.1, 0.01),
        rgba=(0.85, 0.85, 0.85, 1.0),
        joints="default",
        density=50,
        friction=(1.0, 0.0, 0.0),
        texture_file=None,
        texrepeat=(1, 1),
        **kwargs,
    ):
        super().__init__(
            name=name,
            size=np.array(size, dtype=float),
            rgba=np.array(rgba, dtype=float),
            joints=joints,
            density=density,
            **kwargs,
        )
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "card"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class SmallGreenBox(BoxObject):
    """Small green box. Category name 'small_green_box'."""

    def __init__(self, name, size=(0.025, 0.025, 0.025), rgba=(0.0, 0.8, 0.0, 1.0),
                 joints="default", density=926, friction=(1.0, 0.0, 0.0),
                 texture_file=None, texrepeat=(1, 1), **kwargs):
        super().__init__(name=name, size=np.array(size, dtype=float),
                         rgba=np.array(rgba, dtype=float), joints=joints,
                         density=density, **kwargs)
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "small_green_box"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class SmallBlueBox(BoxObject):
    """Small blue box. Category name 'small_blue_box'."""

    def __init__(self, name, size=(0.03, 0.03, 0.03), rgba=(0.0, 0.4, 1.0, 1.0),
                 joints="default", density=926, friction=(1.0, 0.0, 0.0),
                 texture_file=None, texrepeat=(1, 1), **kwargs):
        super().__init__(name=name, size=np.array(size, dtype=float),
                         rgba=np.array(rgba, dtype=float), joints=joints,
                         density=density, **kwargs)
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "small_blue_box"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class SmallRedBox(BoxObject):
    def __init__(self, name, size=(0.03, 0.03, 0.03), rgba=(1.0, 0.1, 0.1, 1.0),
                 joints="default", density=926, friction=(1.0, 0.0, 0.0),
                 texture_file=None, texrepeat=(1, 1), **kwargs):
        super().__init__(name=name, size=np.array(size, dtype=float),
                         rgba=np.array(rgba, dtype=float), joints=joints,
                         density=density, **kwargs)
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "small_red_box"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class SmallYellowBox(BoxObject):
    def __init__(self, name, size=(0.03, 0.03, 0.03), rgba=(1.0, 0.9, 0.0, 1.0),
                 joints="default", density=926, friction=(1.0, 0.0, 0.0),
                 texture_file=None, texrepeat=(1, 1), **kwargs):
        super().__init__(name=name, size=np.array(size, dtype=float),
                         rgba=np.array(rgba, dtype=float), joints=joints,
                         density=density, **kwargs)
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "small_yellow_box"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class SmallPurpleBox(BoxObject):
    def __init__(self, name, size=(0.03, 0.03, 0.03), rgba=(0.6, 0.2, 0.8, 1.0),
                 joints="default", density=926, friction=(1.0, 0.0, 0.0),
                 texture_file=None, texrepeat=(1, 1), **kwargs):
        super().__init__(name=name, size=np.array(size, dtype=float),
                         rgba=np.array(rgba, dtype=float), joints=joints,
                         density=density, **kwargs)
        try:
            fric_str = array_to_string(np.array(friction, dtype=float))
            for g in self.worldbody.findall(".//geom"):
                g.set("friction", fric_str)
        except Exception:
            pass
        try:
            _apply_texture_material(self, texture_file, texrepeat=texrepeat)
        except Exception:
            pass
        self.category_name = "small_purple_box"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        if not hasattr(self, "object_properties"):
            self.object_properties = {}
        self.object_properties.setdefault("vis_site_names", {})


@register_object
class TexturedCube(MujocoXMLObject):
    """UV-mapped cube using a mesh for visuals. Category 'textured_cube'."""

    def __init__(self, name, size=(0.03, 0.03, 0.03), texture_file=None,
                 texrepeat=(1, 1), density=100, friction=(1.0, 0.0, 0.0),
                 joints="default", **kwargs):
        asset_dir = Path(force_coral.get_data_path("assets")) / "primitives" / "textured_cube"
        xml_path = str(asset_dir / "textured_cube.xml")
        super().__init__(
            xml_path,
            name=name,
            joints=[dict(type="free", damping="0.0005")] if joints == "default" else joints,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        try:
            if texture_file:
                texture_path = _resolve_texture_path(texture_file)
                for tex in self.asset.findall("texture"):
                    tex.set("file", texture_path)
            u, v = texrepeat
            for mat in self.asset.findall("material"):
                mat.set("texrepeat", f"{float(u)} {float(v)}")
                mat.set("texuniform", "false")
        except Exception:
            pass
        try:
            sx, sy, sz = [float(x) for x in size]
            for mesh in self.asset.findall("mesh"):
                mesh.set("scale", f"{2*sx} {2*sy} {2*sz}")
        except Exception:
            pass
        try:
            body = self.get_obj()
            for g in body.findall("geom"):
                if g.get("type") == "box" and g.get("group") == "0":
                    g.set("size", f"{sx} {sy} {sz}")
                    if density is not None:
                        g.set("density", str(float(density)))
                    fric_str = array_to_string(np.array(friction, dtype=float))
                    g.set("friction", fric_str)
        except Exception:
            pass
        self.category_name = "textured_cube"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class TexturedWall(MujocoXMLObject):
    """Textured wall with fixed geometry defined in XML. Category 'textured_wall'."""

    def __init__(self, name, joints=None, **kwargs):
        asset_dir = Path(force_coral.get_data_path("assets")) / "primitives" / "textured_wall"
        xml_path = str(asset_dir / "textured_wall.xml")
        super().__init__(
            xml_path,
            name=name,
            joints=joints if joints is not None else [],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "textured_wall"
        self.rotation = (0.0, 0.0)
        self.rotation_axis = "x"
        self.object_properties = {"vis_site_names": {}}
