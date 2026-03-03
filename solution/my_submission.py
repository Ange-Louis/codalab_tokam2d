import torch
import copy
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def train_model(training_dir):
    train_dataset = TokamDataset(training_dir, include_unlabeled=False)

    """
    Creating a validating set from the training data to evince too much 
    validation on CodaBench which one is confronted to a lot of Server Error. 
    """
    total_size = len(train_dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_subset, batch_size=2, collate_fn=collate_fn, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_subset, batch_size=2, collate_fn=collate_fn, shuffle=False
    )

    if torch.cuda.is_available():
        print("Using GPU")
    device = "mps" if torch.cuda.is_available() else "cpu"

    model = fasterrcnn_resnet50_fpn()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters())

    max_epochs = 10

    best_loss = float('inf')
    best_model_weights = None

    for i in range(max_epochs):
        print(f"Epoch {i+1}/{max_epochs}")
        for images, targets in train_dataloader:
            images = [im.to(device) for im in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets
            ]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            full_loss = sum(loss for loss in loss_dict.values())
            full_loss.backward()
            optimizer.step()

        val_loss_total = 0.0
        with torch.no_grad():
            for val_images, val_targets in val_dataloader:
                val_images = [im.to(device) for im in val_images]
                val_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in val_targets]
                val_loss_dict = model(val_images, val_targets)
                val_loss = sum(loss for loss in val_loss_dict.values())
                val_loss_total += val_loss.item()
    
        moyenne_val_loss = val_loss_total / len(val_dataloader)
        print(f"Validation Loss: {moyenne_val_loss}")

        if moyenne_val_loss < best_loss:
            best_loss = moyenne_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)

    model.eval().to("cpu")

    return model
