"""FORTE: Force Optimization via Riemannian Trajectory Estimation.

Self-contained package. Only external dependency: force_coral (LIBERO env construction).
"""

import force_coral

force_coral.bootstrap_libero_extensions(require=True)
