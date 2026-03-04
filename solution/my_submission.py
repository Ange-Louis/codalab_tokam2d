import torch
import copy
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2

from tokam2d_utils import TokamDataset


def collate_fn(batch: torch.Tensor) -> torch.Tensor:
    return tuple(zip(*batch))


def train_model(training_dir):
    """
    Increasing the dataset by modifying the existing dataset with rotation, zoom of the pictures
    """
    transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5),
                             v2.RandomVerticalFlip(p=0.5),
                             v2.RandomAffine(degrees=(-45, 45), scale=(0.8, 1.2)),
                             v2.SanitizeBoundingBoxes()])

    complete_train_dataset = TokamDataset(training_dir, include_unlabeled=False, transforms= transforms)
    complete_val_dataset = TokamDataset(training_dir, include_unlabeled=False)

    """
    Creating a validating set from the training data to evince too much 
    validation on CodaBench which one is confronted to a lot of Server Error. 
    """
    total_size = len(complete_train_dataset)
    train_size = int(total_size * 0.8)

    random_index = torch.randperm(total_size).tolist()
    train_index = random_index[:train_size]
    val_index = random_index[train_size:]

    train_subset = Subset(complete_train_dataset, train_index)
    val_subset = Subset(complete_val_dataset, val_index)

    train_dataloader = DataLoader(
        train_subset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    val_dataloader = DataLoader(
        val_subset, batch_size=4, collate_fn=collate_fn, shuffle=False
    )

    if torch.cuda.is_available():
        print("Using GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = fasterrcnn_resnet50_fpn()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=5e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    max_epochs = 60

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
    
        val_loss_mean = val_loss_total / len(val_dataloader)
        print(f"Validation Loss: {val_loss_mean}")

        scheduler.step(val_loss_mean)

        if val_loss_mean < best_loss:
            best_loss = val_loss_mean
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)

    model.eval().to("cpu")

    return model
