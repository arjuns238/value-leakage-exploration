from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.types import BinaryPreferenceMeasurement


@dataclass
class PairwiseActivationData:
    activations: dict[int, np.ndarray]
    # Aggregated unique pairs: each row is (task_i_idx, task_j_idx)
    pairs: np.ndarray  # (n_unique_pairs, 2) — task_i always has lower index
    wins_i: np.ndarray  # (n_unique_pairs,) — number of times task_i won
    total: np.ndarray  # (n_unique_pairs,) — total comparisons per pair
    n_measurements: int  # total raw measurements before aggregation

    def filter_by_indices(self, idx_set: set[int]) -> PairwiseActivationData:
        """Keep only pairs where both task indices are in idx_set."""
        mask = np.array([
            int(i) in idx_set and int(j) in idx_set
            for i, j in self.pairs
        ])
        if mask.sum() == 0:
            return PairwiseActivationData(
                activations=self.activations,
                pairs=np.empty((0, 2), dtype=int),
                wins_i=np.empty(0, dtype=float),
                total=np.empty(0, dtype=float),
                n_measurements=0,
            )
        return PairwiseActivationData(
            activations=self.activations,
            pairs=self.pairs[mask],
            wins_i=self.wins_i[mask],
            total=self.total[mask],
            n_measurements=int(np.sum(self.total[mask])),
        )

    def split_by_groups(
        self,
        task_ids: np.ndarray,
        task_groups: dict[str, str],
        held_out_set: set[str],
    ) -> tuple[PairwiseActivationData, PairwiseActivationData]:
        """Split into train/eval by task group.

        Train: pairs where both tasks are in train groups.
        Eval: pairs where both tasks are in held-out groups.
        Cross-group pairs are dropped.
        """
        idx_to_tid = {i: tid for i, tid in enumerate(task_ids)}

        train_mask = np.zeros(len(self.pairs), dtype=bool)
        eval_mask = np.zeros(len(self.pairs), dtype=bool)

        for k, (i, j) in enumerate(self.pairs):
            tid_i = idx_to_tid.get(i)
            tid_j = idx_to_tid.get(j)
            if tid_i is None or tid_j is None:
                continue
            g_i = task_groups.get(tid_i)
            g_j = task_groups.get(tid_j)
            if g_i is None or g_j is None:
                continue
            i_held = g_i in held_out_set
            j_held = g_j in held_out_set
            if not i_held and not j_held:
                train_mask[k] = True
            elif i_held and j_held:
                eval_mask[k] = True

        def _subset(mask: np.ndarray) -> PairwiseActivationData:
            return PairwiseActivationData(
                activations=self.activations,
                pairs=self.pairs[mask],
                wins_i=self.wins_i[mask],
                total=self.total[mask],
                n_measurements=int(np.sum(self.total[mask])),
            )

        return _subset(train_mask), _subset(eval_mask)

    @classmethod
    def from_measurements(
        cls,
        measurements: list[BinaryPreferenceMeasurement],
        task_ids: np.ndarray,
        activations: dict[int, np.ndarray],
    ) -> PairwiseActivationData:
        id_to_idx = {tid: i for i, tid in enumerate(task_ids)}

        # Count wins per ordered pair (lower_idx, higher_idx)
        pair_wins: Counter[tuple[int, int], int] = Counter()
        pair_total: Counter[tuple[int, int], int] = Counter()
        n_measurements = 0

        for m in measurements:
            if m.choice == "refusal":
                continue
            a_id, b_id = m.task_a.id, m.task_b.id
            if a_id not in id_to_idx or b_id not in id_to_idx:
                continue
            a_idx, b_idx = id_to_idx[a_id], id_to_idx[b_id]
            winner_idx = a_idx if m.choice == "a" else b_idx

            # Canonical ordering: lower index is always task_i
            i, j = min(a_idx, b_idx), max(a_idx, b_idx)
            pair_total[(i, j)] += 1
            if winner_idx == i:
                pair_wins[(i, j)] += 1
            n_measurements += 1

        if not pair_total:
            empty = np.empty((0, 2), dtype=int)
            return cls(
                activations=activations, pairs=empty,
                wins_i=np.empty(0, dtype=float), total=np.empty(0, dtype=float),
                n_measurements=0,
            )

        unique_pairs = sorted(pair_total.keys())
        pairs = np.array(unique_pairs, dtype=int)
        wins_i = np.array([pair_wins[p] for p in unique_pairs], dtype=float)
        total = np.array([pair_total[p] for p in unique_pairs], dtype=float)

        return cls(
            activations=activations, pairs=pairs,
            wins_i=wins_i, total=total, n_measurements=n_measurements,
        )
