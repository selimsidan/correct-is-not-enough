"""Execution accuracy (EX).

Ported from the official BIRD mini-dev evaluation:
  repository bird-bench/mini_dev, commit 1c49a375e76c471e218564da3f4617513d33e371,
  file evaluation/evaluation_ex.py, function calculate_ex:

      def calculate_ex(predicted_res, ground_truth_res):
          res = 0
          if set(predicted_res) == set(ground_truth_res):
              res = 1
          return res

Errors and timeouts (30 s, meta_time_out) score 0 in the official script, and here too.
For cached gold results, `hashes_match` compares result_hash values, which are defined to agree
with set equality (see exec/results.py).
"""

from __future__ import annotations


def calculate_ex(predicted_res: list[tuple], ground_truth_res: list[tuple]) -> int:
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res


def hashes_match(pred_hash: str | None, gold_hash: str | None) -> bool:
    return pred_hash is not None and gold_hash is not None and pred_hash == gold_hash
