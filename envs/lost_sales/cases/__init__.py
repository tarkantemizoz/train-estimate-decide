"""The paper's evaluation cases, shared by every environment.

Each module lists one case's instances in the legacy loop order.
Each environment says which cases its model can represent, and how.
"""

from . import case1, case2, case3, online
from .instance import Instance

CASES = {
    "case1": case1.instances,
    "case2": case2.instances,
    "case3": case3.instances,
    "online": online.instances,
}


def case_of(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def find(instance_id: str) -> Instance:
    """The instance with this ID, such as case1-000 or online-003."""
    for item in CASES.get(case_of(instance_id), list)():
        if item.id == instance_id:
            return item
    raise KeyError(f"unknown instance {instance_id!r}; IDs look like case1-000 or online-003")


__all__ = ["CASES", "Instance", "case_of", "find"]
