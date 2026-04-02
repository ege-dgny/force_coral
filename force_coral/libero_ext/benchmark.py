"""
Custom benchmark suite (``my_suite``) and its task map.

Importing this module registers the ``my_suite`` benchmark into LIBERO's
``BENCHMARK_MAPPING`` so that ``get_benchmark("my_suite")`` works without
modifying any LIBERO source file.
"""

import os
import torch

from libero.libero.benchmark import (
    Benchmark,
    Task,
    register_benchmark,
    grab_language_from_filename,
    task_maps,
)
import force_coral

# ── Task list for my_suite ────────────────────────────────────────────────
MY_SUITE_TASKS = [
    "push_the_blue_box_to_the_front_of_the_wall",
    "push_the_box_up_along_the_wall_while_maintaining_contact",
    "flip_the_blue_box_onto_its_side",
    "push_the_box_to_the_wall_and_use_the_wall_as_a_support_to_flip_the_box_onto_its_side",
    "push_the_card_to_the_edge_of_the_table_and_pick_the_card",
    "pick_the_green_box_and_place_it_in_the_basket",
    "pick_the_blue_box_and_place_it_in_the_basket",
]


def _register_task_map():
    """Inject my_suite tasks into LIBERO's ``task_maps`` dict."""
    if "my_suite" in task_maps:
        return  # already registered
    task_maps["my_suite"] = {}
    for task_name in MY_SUITE_TASKS:
        language = grab_language_from_filename(task_name + ".bddl")
        task_maps["my_suite"][task_name] = Task(
            name=task_name,
            language=language,
            problem="Libero",
            problem_folder="my_suite",
            bddl_file=f"{task_name}.bddl",
            init_states_file=f"{task_name}.pruned_init",
        )


# Run task map injection on import
_register_task_map()


# ── Benchmark class ───────────────────────────────────────────────────────
@register_benchmark
class my_suite(Benchmark):  # noqa: N801  (lowercase matches LIBERO convention)
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "my_suite"
        # Use all tasks in declaration order (no 10-task reordering)
        tasks = list(task_maps[self.name].values())
        self.tasks = tasks
        self.n_tasks = len(self.tasks)

    # Override path methods to point at force_coral's data directory
    def get_task_bddl_file_path(self, i):
        return os.path.join(
            force_coral.get_data_path("bddl_files"),
            "my_suite",
            self.tasks[i].bddl_file,
        )

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            force_coral.get_data_path("init_states"),
            "my_suite",
            self.tasks[i].init_states_file,
        )
        return torch.load(init_states_path, weights_only=False, map_location="cpu")
