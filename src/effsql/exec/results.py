"""Result-set hashing that agrees with Python set equality.

The official EX check is `set(predicted_res) == set(ground_truth_res)`. To compare against a
cached gold result we store a hash instead of the rows, so the hash must treat rows exactly as
`set()` does: order and duplicates are ignored, and values that compare equal in Python hash
equal (1 == 1.0, so an integral float is canonicalized to int). Strings and numbers stay distinct
("1" != 1), as in Python.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable


def _canon_value(v: Any) -> Any:
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    return v


def canonical_rows(rows: Iterable[tuple]) -> list[str]:
    """Sorted, de-duplicated text form of a row set."""
    return sorted({repr(tuple(_canon_value(v) for v in row)) for row in rows})


def result_hash(rows: Iterable[tuple]) -> str:
    h = hashlib.sha256()
    for line in canonical_rows(rows):
        h.update(line.encode("utf-8", "surrogatepass"))
        h.update(b"\n")
    return h.hexdigest()
