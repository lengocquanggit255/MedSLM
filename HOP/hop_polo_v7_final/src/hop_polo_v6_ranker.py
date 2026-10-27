"""V6 pool ranker with a fixed zero old-score gate and fixed-score negatives."""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

from hop_polo_v2_ranker import (PoolRankerV2, cold_numeric_score,
                                collect_pool as collect_pool_v2)


@torch.inference_mode()
def collect_pool(*args, **kwargs):
    row = collect_pool_v2(*args, **kwargs)
    # The untrained beam head is never a feature or a residual in v6.
    row.old_scores.fill(0)
    row.numeric[:, 0] = 0
    return row


class PoolRankerV6(PoolRankerV2):
    def __init__(self, node_text, relation_text, node_types):
        super().__init__(node_text, relation_text, node_types, old_gate=0.0)
        self.old_scale.requires_grad_(False)

    def forward(self, nodes, old_scores, numeric, relation_paths,
                question_text):
        assert not torch.any(old_scores != 0)
        return super().forward(nodes, old_scores, numeric,
                               relation_paths, question_text)


def training_labels(rows, questions, graph):
    """Positive answers and hard negatives from the fixed label-free score."""
    labels = []
    stats = Counter()
    for row, q in zip(rows, questions):
        assert not np.any(row.old_scores)
        mask = np.isin(row.nodes, q.answers)
        positives = np.flatnonzero(mask)
        wrong = np.flatnonzero(~mask)
        stats["questions"] += 1
        stats["questions_with_positive"] += int(len(positives) > 0)
        stats["gold_in_pool"] += len(positives)
        stats["gold_total"] += len(q.answers)
        stats["candidate_nodes"] += len(row.nodes)
        if not len(positives) or not len(wrong):
            labels.append((positives, np.empty(0, dtype=np.int64)))
            continue
        base = cold_numeric_score(row.numeric)
        ordered = wrong[np.argsort(-base[wrong], kind="stable")]
        types = graph.types[row.nodes.astype(np.int64)]
        positive_types = set(int(x) for x in types[positives])
        same_type = [int(j) for j in ordered
                     if int(types[j]) in positive_types]
        hard = list(dict.fromkeys(ordered[:32].tolist() + same_type[:32]))
        labels.append((positives.astype(np.int64),
                       np.asarray(hard, dtype=np.int64)))
        stats["hard_wrong"] += len(hard)
        stats["same_type_hard_wrong"] += len(same_type[:32])
    return labels, {**stats, "hard_negative_source": "cold_numeric_score",
                    "old_scores_used": False}
