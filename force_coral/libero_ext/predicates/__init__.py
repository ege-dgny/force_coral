"""Register custom predicates into LIBERO's predicate registry."""

from force_coral.libero_ext.predicates.custom_predicates import (  # noqa: F401
    OnSide,
    OnSideWithRecentSupport,
)
from libero.libero.envs.predicates import VALIDATE_PREDICATE_FN_DICT
from libero.libero.envs.predicates.base_predicates import InContactPredicateFn

# InContact is defined in LIBERO but commented out in its predicate dict;
# our BDDL tasks need it, so register it here.
VALIDATE_PREDICATE_FN_DICT.setdefault("incontact", InContactPredicateFn())

VALIDATE_PREDICATE_FN_DICT.setdefault("onside", OnSide())
VALIDATE_PREDICATE_FN_DICT.setdefault("onsidewithrecentsupport", OnSideWithRecentSupport())
