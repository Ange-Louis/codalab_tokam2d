import torch
import copy
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
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
    transforms = v2.Compose([v2.RandomVerticalFlip(p=0.5),
                             v2.RandomAffine(degrees=(-45, 45), scale=(0.8, 1.2)),
                             v2.SanitizeBoundingBoxes()])

    transform_train_dataset = TokamDataset(training_dir, include_unlabeled=False, transforms= transforms)

    train_dataloader = DataLoader(
        transform_train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    # train_dataset = TokamDataset(training_dir, include_unlabeled=False)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    # )

    if torch.cuda.is_available():
        print("Using GPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = fasterrcnn_resnet50_fpn()
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    model.to(device)
    model.train()

    max_epochs = 5
    
    optimizer = torch.optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=5e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

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

        scheduler.step()


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

    images_traitees = 0
    limite_images = 50 

    # Utilisation d'un itérateur manuel pour capturer l'IndexError silencieusement
    data_iterator = iter(unlabeled_dataloader)

    with torch.no_grad():
        while True:
            if images_traitees >= limite_images:
                break
                
            try:
                # On tente de charger le prochain batch
                unlabeled_images, _ = next(data_iterator)
            except StopIteration:
                # Fin normale du DataLoader
                break
            except IndexError:
                # Bug du dataset CodaBench atteint, on arrête proprement
                print("IndexError interceptée : données corrompues ignorées.")
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
            
            images_traitees += len(unlabeled_images)

    if len(pseudo_labeled_data) == 0:
        raise RuntimeError("Erreur : Aucun pseudo-label n'a été généré. Le modèle initial n'est pas assez confiant ou le threshold (0.85) est trop élevé.")
    
    mixed_train_dataset = ConcatDataset([transform_train_dataset, pseudo_labeled_data])

    # mixed_train_dataset= list(train_dataset) + pseudo_labeled_data

    mixed_train_dataloader= DataLoader(
        mixed_train_dataset, batch_size=4, collate_fn=collate_fn, shuffle=True
    )

    """
    Second training loop, this time using the generated pseudo-labels
    """
    model.train()

    max_epochs = 5
    
    optimizer = torch.optim.AdamW(model.parameters(), lr= 5e-5, weight_decay=5e-5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

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

        scheduler.step()

    model.eval().to("cpu")

    filtered_model= ModelFilter(model, threshold= 0.05)

    return filtered_model
