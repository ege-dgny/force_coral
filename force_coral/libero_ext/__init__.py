"""
force_coral LIBERO extensions.

Importing this package registers all custom objects, robots, predicates,
regions, problems, and benchmarks into LIBERO's plugin registries.

Registration order matters:
1. Objects  (so problem classes can look them up via get_object_fn)
2. Robots   (so problem classes can reference custom robot names)
3. Predicates (so predicate evaluation works for custom goals)
4. Regions  (so region sampling works for custom domains)
5. Problems (triggers @register_problem -> TASK_MAPPING)
6. Benchmark (triggers @register_benchmark -> BENCHMARK_MAPPING)
"""

# 1. Custom objects
import force_coral.libero_ext.objects  # noqa: F401

# 2. Custom robots
import force_coral.libero_ext.robots  # noqa: F401

# 3. Custom predicates
import force_coral.libero_ext.predicates  # noqa: F401

# 4. Custom regions
import force_coral.libero_ext.regions  # noqa: F401

# 5. Custom problem domains
import force_coral.libero_ext.problems  # noqa: F401

# 6. Custom benchmark suite (my_suite)
import force_coral.libero_ext.benchmark  # noqa: F401
