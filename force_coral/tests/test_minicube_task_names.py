import force_coral  # noqa: F401  bootstrap
from libero.libero import benchmark, get_libero_path

d = benchmark.get_benchmark_dict()
suite = d["libero_minicube"]()

print("n_tasks:", suite.n_tasks)
print("task names:", suite.get_task_names())

# Show where LIBERO expects your files (they may not exist yet — that’s OK for now)
bddl_path = suite.get_task_bddl_file_path(0)
init_states_path = f"{get_libero_path('init_states')}/libero_minicube/lift_the_red_cube.pruned_init"
print("expected BDDL path:", bddl_path)
print("expected init-states path:", init_states_path)
