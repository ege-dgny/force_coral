import force_coral  # noqa: F401  bootstrap
from libero.libero import benchmark

d = benchmark.get_benchmark_dict()
print("Has libero_minicube in registry? ->", "libero_minicube" in d)

SuiteClass = d.get("libero_minicube")
if SuiteClass:
    suite = SuiteClass()
    print("Constructed:", type(suite).__name__)
    print("n_tasks:", suite.n_tasks)
    print("task names:", getattr(suite, "get_task_names", lambda: [])())
