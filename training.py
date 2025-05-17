import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from ultralytics import YOLO
from tqdm import tqdm
from DataAugmentation import DatasetAugmentation
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from ultralytics.utils.loss import v8DetectionLoss
from torchvision.transforms.v2 import ConvertBoundingBoxFormat
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

'''def collate_fn(batch):
    images, boxes_list, labels_list = zip(*batch)
    images = torch.stack(images, dim=0)
    B, C, H, W = images.shape
    targets = []
    for batch_idx, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)  # CPU only
        bboxes = BoundingBoxes(boxes, format=BoundingBoxFormat.XYXY, canvas_size=(H, W))
        bboxes = ConvertBoundingBoxFormat(BoundingBoxFormat.CXCYWH)(bboxes)
        boxes = torch.as_tensor(bboxes)
        boxes = boxes / torch.tensor([W, H, W, H])
        cls    = labels.view(-1, 1).float()
        idxs   = torch.full((boxes.size(0), 1), batch_idx, dtype=torch.float32)
        targets.append(torch.cat([idxs, cls, boxes], dim=1))
    targets = torch.cat(targets, dim=0)
    return images, {
        "batch_idx": targets[:,0].long(),
        "cls":       targets[:,1],
        "bboxes":    targets[:,2:]
    }'''
def collate_fn(batch):
    images, boxes_list, labels_list = zip(*batch) # Unzip the batch, * is used to unpack the list of tuples
    images = torch.stack(images, dim=0) 
    targets = []
    for batch_idx, (bbs, labels) in enumerate(zip(boxes_list, labels_list)):
        # bbs.data is (num_objs, 4) in CXCYWH absolute pixels
        coords = bbs.data  # torch.Tensor
        H, W = bbs.canvas_size
        # normalize [cx,cy,w,h]
        coords = coords / torch.tensor([W, H, W, H], dtype=torch.float32, device=coords.device)
        cls   = labels.to(torch.float32).view(-1,1)
        idxs  = torch.full((coords.size(0),1), batch_idx, dtype=torch.float32, device=coords.device)
        targets.append(torch.cat([idxs, cls, coords], dim=1))
    targets = torch.cat(targets, dim=0)
    return images, {
        "batch_idx": targets[:,0].long(),
        "cls":       targets[:,1],
        "bboxes":    targets[:,2:]
    }

class FixedDict(dict):
    def __init__(self, d):
        self.__dict__.update(d)

def main():
    MODEL_PATH      = "yolo12.yaml"
    OUT_DIR        = "output_dir"
    device         = "cuda" if torch.cuda.is_available() else "mps"
    print(f"Using device: {device}")
    IMG_SIZE       = 640 #yolo default input size
    BATCH_SIZE     = 6
    N_EPOCHS       = 30
    LR             = 1e-3
    WEIGHT_DECAY   = 5e-4
    PCT_START      = 0.3   # OneCycle warm-up

    os.makedirs(OUT_DIR, exist_ok=True)


    resize_transform = v2.Compose([
        v2.Resize((640, 640)),
        v2.SanitizeBoundingBoxes(),
        v2.ToImage(),
    ])

    train_dataset = DatasetAugmentation(
        training_path="augmented_data/train",#"train/split/train",#
        split_images=False,
        perform_transformations=True 
    )
    train_dataset.transforms = resize_transform

    val_dataset = DatasetAugmentation(
        training_path="augmented_data/val",#"train/split/val",#
        split_images=False,
        perform_transformations=True
    )
    val_dataset.transforms = resize_transform
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=8,pin_memory=True, collate_fn=collate_fn,persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=collate_fn,persistent_workers=True)

    # model and weights initialization
    model_yolo = YOLO(MODEL_PATH)
    model = model_yolo.model
    num_params_all = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in the model: {num_params_all}")
    model=model.to(device)
    '''def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    model.apply(init_weights)'''



    # some more initialization
    compute_loss = v8DetectionLoss(model)  # box, obj, cls losses
    compute_loss.hyp = FixedDict(compute_loss.hyp)
    '''optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = N_EPOCHS * len(train_loader)
    scheduler    = OneCycleLR(optimizer,
                            max_lr=LR * 2,
                            total_steps=total_steps,
                            pct_start=PCT_START,
                            anneal_strategy="cos")'''
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4, nesterov=True)
    total_steps = N_EPOCHS * len(train_loader)
    warmup_steps = int(0.3 * total_steps)
    sched1 = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
    sched2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [sched1, sched2], [warmup_steps])
    

    #TRAINING LOOP + validation
    best_val_loss = float("inf")
    best_epoch    = 0
    loss_history_train = []
    loss_history_val   = []

    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Epoch Loop", unit="epoch"):
        #Train
        #model.train() not needed bcs this is the ultralytics training and we want our own
        train_loss = 0.0
        model.train()
        for imgs, targets in tqdm(train_loader, desc=f"Training", unit="batch", leave=False):
            imgs   = imgs.to(device).float() / 255.0
            for k, v in targets.items():
                targets[k] = v.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss, _ = compute_loss(preds, targets)
            loss = loss.sum() / imgs.shape[0] # average loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        #Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Evaluation", unit="batch", leave=False):
                imgs    = imgs.to(device).float() / 255.0
                for k, v in targets.items():
                    targets[k] = v.to(device)
                preds   = model(imgs)
                loss, _ = compute_loss(preds, targets)
                loss = loss.sum() / imgs.shape[0] # average loss
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        loss_history_train.append(avg_train_loss)
        loss_history_val.append(avg_val_loss)

        # Keep best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            best_model_state = model.state_dict()

    # save the best model and training history
    torch.save(best_model_state, os.path.join(OUT_DIR, "best.pt"))
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "last.pt"))
    plot_data = {
        "train_loss": loss_history_train,
        "val_loss":   loss_history_val,
        "best_epoch": best_epoch,
        "best_model": best_model_state,   # optional
    }
    torch.save(plot_data, os.path.join(OUT_DIR, "plot_data.pt"))

    # plotting the loss curves
    epochs = range(1, N_EPOCHS + 1)

    plt.figure()
    plt.plot(epochs, loss_history_train, label="Train Loss")
    plt.plot(epochs, loss_history_val,   label="Val Loss")
    plt.axvline(best_epoch, linestyle="--", color="red",
                label=f"Best Epoch = {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()