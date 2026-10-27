from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class MVPConfig:
    """Single source of truth for the reproducible MVP configuration."""

    data_dir: Path
    gnn_checkpoint: Path
    output_dir: Path
    device: str = "cuda"
    seed: int = 42

    n_clusters: int = 64
    use_semantic_ids: bool = True
    semantic_id_source: str = "kmeans"
    codebook_sample: int = 50_000
    codebook_batch_size: int = 4_096
    codebook_max_iter: int = 100
    codebook_n_init: int = 3

    budgets: tuple[int, ...] = (256, 512, 1_024, 2_048, 4_096, 8_192)
    max_steps: int = 6
    max_edges_per_action: int = 2_048
    max_candidate_actions: int = 512
    edge_semantic_weight: float = 0.7
    edge_ppr_weight: float = 0.3

    hidden_dim: int = 128
    sft_epochs: int = 10
    sft_batch_size: int = 64
    sft_lr: float = 3e-4
    sft_patience: int = 2
    sft_demo_repeats: int = 1

    ppo_epochs: int = 3
    ppo_rollouts_per_query: int = 2
    ppo_minibatch_size: int = 64
    ppo_minibatch_passes: int = 4
    ppo_lr: float = 1e-4
    ppo_gamma: float = 1.0
    ppo_gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_entropy: float = 0.01
    ppo_value_coef: float = 0.5
    grad_clip: float = 1.0

    bootstrap_samples: int = 1_000

    limit_train: int = 0
    limit_val: int = 0
    limit_test: int = 0
    eval_dense: bool = True

    def as_json(self) -> dict:
        out = asdict(self)
        out["data_dir"] = str(self.data_dir)
        out["gnn_checkpoint"] = str(self.gnn_checkpoint)
        out["output_dir"] = str(self.output_dir)
        out["budgets"] = list(self.budgets)
        return out

    @classmethod
    def quick(
        cls,
        data_dir: Path,
        gnn_checkpoint: Path,
        output_dir: Path,
        device: str = "cpu",
    ) -> "MVPConfig":
        """Tiny end-to-end configuration used by CI and Kaggle smoke runs."""
        return cls(
            data_dir=data_dir,
            gnn_checkpoint=gnn_checkpoint,
            output_dir=output_dir,
            device=device,
            budgets=(32, 64),
            max_steps=3,
            max_edges_per_action=32,
            n_clusters=8,
            codebook_sample=2_000,
            codebook_batch_size=256,
            codebook_max_iter=20,
            codebook_n_init=1,
            hidden_dim=64,
            sft_epochs=1,
            sft_batch_size=8,
            sft_patience=1,
            ppo_epochs=1,
            ppo_rollouts_per_query=1,
            ppo_minibatch_size=8,
            ppo_minibatch_passes=1,
            bootstrap_samples=100,
            limit_train=8,
            limit_val=4,
            limit_test=4,
            eval_dense=False,
        )


def parse_budgets(value: str | Sequence[int]) -> tuple[int, ...]:
    if not isinstance(value, str):
        return tuple(int(x) for x in value)
    result = tuple(sorted({int(x.strip()) for x in value.split(",") if x.strip()}))
    if not result or any(x <= 0 for x in result):
        raise ValueError("budgets must be a comma-separated list of positive integers")
    return result
