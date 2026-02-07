import force_coral  # noqa: F401  bootstrap
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
print("Has libero_minicube key? ->", "libero_minicube" in libero_task_map)
if "libero_minicube" in libero_task_map:
    val = libero_task_map["libero_minicube"]
    print("Type:", type(val).__name__, "| length:", len(val))


from libero.libero.envs.bddl_base_domain import TASK_MAPPING
print("my_floor_manipulation" in TASK_MAPPING)  # should be True
