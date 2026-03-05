import torch
import copy
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import v2

from tokam2d_utils import TokamDataset

class ModelFilter(torch.nn.Module):
    """
    Creating a filter for preserving only best boxe predictions
    """
    def __init__(self, model, threshold= 0.5):
        super().__init__()

        self.model= model
        self.threshold= threshold

    def forward(self, images, targets= None):
        results= self.model(images, targets)

        if self.training:
            return results
        
        else:
            good_results= []

            for pred in results:
                mask= pred['scores'] > self.threshold

                good_pred= {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }

                good_results.append(good_pred)

            return good_results

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


    """
    Creating a validating set from the training data to evince too much 
    validation on CodaBench which one is confronted to a lot of Server Error. 
    """
    complete_val_dataset = TokamDataset(training_dir, include_unlabeled=False)


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

    max_epochs = 20

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


    """
    Auto-teachering
    """
    # 1. On donne le dataset directement, SANS utiliser Subset
    unlabeled_train_dataset = TokamDataset(training_dir, include_unlabeled=True)

    unlabeled_dataloader = DataLoader(
        unlabeled_train_dataset, batch_size=2, collate_fn=collate_fn, shuffle=False
    )

    model.eval()

    teacher_model = ModelFilter(model, threshold=0.85)
    pseudo_labeled_data = []

    # 2. Notre sécurité anti-crash et anti-timeout
    images_traitees = 0
    limite_images = 50 

    with torch.no_grad():
        for unlabeled_images, _ in unlabeled_dataloader:
            
            # Si on a atteint la limite, on explose la boucle !
            if images_traitees >= limite_images:
                break
                
            gpu_images= [im.to(device) for im in unlabeled_images]

            filtered_predictions = teacher_model(gpu_images)

            for i, pred in enumerate(filtered_predictions):
                if len(pred['boxes']) > 0:
                    
                    pseudo_target = {
                        'boxes': pred['boxes'].cpu(),
                        'labels': pred['labels'].to(torch.int64).cpu()
                    }
    
                    pseudo_labeled_data.append((unlabeled_images[i], pseudo_target))
            
            # On ajoute le nombre d'images de ce batch (ex: 2) à notre compteur
            images_traitees += len(unlabeled_images)

    
    mixed_train_dataset= list(train_subset) + pseudo_labeled_data

    mixed_train_dataloader= DataLoader(
        mixed_train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    """
    Second training loop, this time using the generated pseudo-labels
    """
    model.train()

    max_epochs = 10

    best_loss = float('inf')
    best_model_weights = None

    for i in range(max_epochs):
        print(f"Epoch {i+1}/{max_epochs}")
        for images, targets in mixed_train_dataloader:
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

    filtered_model= ModelFilter(model, threshold= 0.05)

    return filtered_model
