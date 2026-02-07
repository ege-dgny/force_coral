"""Register custom region samplers into LIBERO's REGION_SAMPLERS dict."""

from libero.libero.envs.regions import REGION_SAMPLERS
from libero.libero.envs.regions.workspace_region_sampler import TableRegionSampler

REGION_SAMPLERS.setdefault("my_floor_manipulation", {"floor": TableRegionSampler})
REGION_SAMPLERS.setdefault("my_tabletop_manipulation", {"table": TableRegionSampler})
REGION_SAMPLERS.setdefault("my_floor_manipulation_shifted_robot", {"floor": TableRegionSampler})
