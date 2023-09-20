import os
from pathlib import Path

import wandb

if not os.environ.get("WANDB_API_KEY"):
    raise ValueError("You must set WANDB_API_KEY environment variable")

wandb_team = "harrisonyu"
wandb_project = "model-registry"
wandb_model = "foodformer:v1"
wandb_model_path = f"{wandb_team}/{wandb_project}/{wandb_model}"

wandb.init()

current_folder = Path(__file__).parent

path = wandb.use_artifact(wandb_model_path, type="model").download(root=current_folder)
