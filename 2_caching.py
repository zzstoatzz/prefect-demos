# /// script
# dependencies = [
#     "numpy",
#     "prefect",
#     "torch",
# ]
# ///

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
from prefect import flow, task
from prefect.cache_policies import CacheKeyFnPolicy


@dataclass
class TrainConfig:
    version: str
    dataset_path: str
    batch_size: int = 32  # trades off speed vs. stability

    # Architecture config
    d_model: int = 512  # embedding dimension, powers of 2 for efficiency
    n_layers: int = 4  # deeper -> more capacity but harder to train

    # Training config
    lr: float = 3e-4  # standard Adam learning rate from Transformer paper
    n_epochs: int = 100
    grad_clip: float = 1.0  # prevents exploding gradients

    # Where to save model checkpoints
    save_dir: str = "checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def make_cache_key(context, params) -> str:
    """Cache dataset preprocessing based on model architecture and data version."""
    config = params["config"]
    return f"v{config.version}_{config.d_model}_{config.n_layers}"


PREPROCESSING_CACHE_POLICY = CacheKeyFnPolicy(cache_key_fn=make_cache_key).configure(
    key_storage=Path("cache")
)


@task(
    cache_policy=PREPROCESSING_CACHE_POLICY,
    cache_expiration=timedelta(hours=12),
    persist_result=True,
    result_storage_key="data_v{parameters[config].version}.pt",
)
def prepare_dataset(config: TrainConfig) -> torch.utils.data.DataLoader:
    """Load and prepare dataset. Cached since preprocessing is expensive."""
    print(f"Loading dataset from {config.dataset_path}")
    # In real code, this would load and process your data
    return torch.utils.data.DataLoader([], batch_size=config.batch_size)


@task(
    cache_expiration=timedelta(minutes=30),
    persist_result=True,
    result_storage_key="ckpt_epoch_{parameters[epoch]}.pt",
    cache_result_in_memory=False,
)
def train_epoch(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    grad_clip: float,
) -> float:
    """Train for one epoch and return avg loss (checkpointed every epoch)."""
    if not dataloader or len(dataloader) == 0:
        print("No data to train on!")
        return float("inf")

    model.train()
    total_loss = 0.0
    n_batches = 0

    try:
        for x, y in dataloader:  # Explicit data unpacking
            # Forward pass and loss computation
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch}: loss {avg_loss:.4f}")
        return avg_loss
    except Exception as e:
        print(f"Error during training: {e}")
        return float("inf")


@flow
def training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: torch.utils.data.DataLoader,
    config: TrainConfig,
) -> dict:
    """Main training loop, checkpointing model after each epoch."""
    state = {}
    model = model.to(config.device)
    best_loss = float("inf")

    for epoch in range(config.n_epochs):
        loss = train_epoch(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            grad_clip=config.grad_clip,
        )
        if loss < best_loss:
            best_loss = loss
            # Only save model and optimizer state we need to resume training
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
    return state


@task(
    persist_result=True,
    result_storage_key="model_v{parameters[config].version}.pt",
)
def save_checkpoint(state: dict, config: TrainConfig) -> None:
    """Save model checkpoint."""
    path = Path(config.save_dir) / f"model_v{config.version}.pt"
    path.parent.mkdir(exist_ok=True)
    torch.save(state, path)
    print(f"Saved best model to {path} (loss: {state['loss']:.4f})")


@flow(log_prints=True)
def train(config: TrainConfig) -> None:
    """Train a model with checkpointing."""
    torch.manual_seed(42)  # reproducibility

    dataloader = prepare_dataset(config)
    model = torch.nn.Linear(10, 10)  # placeholder for demo

    final_state = training_loop(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,  # 3e-4: good default for Adam + transformers
        ),
        dataloader=dataloader,
        config=config,
    )
    save_checkpoint(final_state, config)


if __name__ == "__main__":
    config = dict(
        version="1",
        dataset_path="data/train",
        d_model=512,  # try 768 for larger model
        n_layers=4,  # try 8 for deeper model
        n_epochs=3,  # increase for better results
    )
    train(config)
