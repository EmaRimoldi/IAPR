import os
import glob
from torch.utils.data import DataLoader
from PIL import Image
import torch
from ultralytics import YOLO
import pandas as pd
import torchvision.transforms.v2 as v2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.ops import nms

def xywh_to_xyxy(x):
    # go from yolo format to torch format
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2 
    y[:, 1] = x[:, 1] - x[:, 3] / 2 
    y[:, 2] = x[:, 0] + x[:, 2] / 2 
    y[:, 3] = x[:, 1] + x[:, 3] / 2 
    return y

def count_classes(pred,conf_thres=0.25, iou_thres=0.5):
    pred = pred.permute(1,0)
    boxes_xywh = pred[:, 0:4]          # (N,4)
    cls_prob   = pred[:, 4:]           # (N, nc)

    # Get confidence
    cls_conf, cls_ids = cls_prob.max(dim=1)  

    # keep only the boxes with a score above the threshold
    detections   = cls_conf > conf_thres
    if detections.sum() == 0: # no detections
        final_cls = torch.empty((0,), dtype=torch.int64)
    else:
        boxes_xywh = boxes_xywh[detections]
        boxes = xywh_to_xyxy(boxes_xywh)
        cls_conf = cls_conf[detections]
        cls_ids = cls_ids[detections]

        # run NMS per class (keep non overlapping boxes)
        keep = []
        for c in cls_ids.unique():
            idxs = (cls_ids == c).nonzero().flatten() #get indicies
            b    = boxes[idxs]
            scores    = cls_conf[idxs]
            k    = nms(b, scores, iou_thres)
            keep.append(idxs[k])
        keep = torch.cat(keep)
        final_cls = cls_ids[keep].cpu().numpy().astype(int)
    return final_cls


class FixedDict(dict):
    def __init__(self, d):
        self.__dict__.update(d)
def collate_fn(batch):
        imgs, ids = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, list(ids)

class TestDataset(Dataset):
    """Loads all L*.jpg images from a folder and applied resizing transform"""
    def __init__(self, image_dir, transform=None):
        self.paths = sorted(glob.glob(os.path.join(image_dir, 'L*.jpg')))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # get the id LXXXXXX.jpg : id XXXXXX
        img_id = int(os.path.splitext(os.path.basename(path))[0][1:])
        return img, img_id
    
def main():
    MODEL_PATH = "project\yolo12s.yaml"
    test_dir = "test"
    weights_path = "output_dir/best.pt"
    output_csv = "submission.csv"
    BATCH_SIZE = 6
    num_classes = 13
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    #go from the model classes to the submission classes
    model2idx = {
        0:  3,   # Amandina : 3
        1: 12,   # Arabia   : 12
        2:  7,   # Comtesse : 7
        3:  4,   # Creme_brulee : 4
        4:  2,   # Jelly_Black : 2
        5:  1,   # Jelly_Milk  : 1
        6:  0,   # Jelly_White : 0
        7:  8,   # Noblesse : 8
        8:  9,   # Noir_authentique : 9
        9: 10,   # Passion_au_lait : 10
        10: 11,  # Stracciatella : 11
        11:  6,  # Tentation_noir : 6
        12:  5,  # Triangolo : 5
    }
    cols = [
        "Jelly White", "Jelly Milk", "Jelly Black",
        "Amandina", "Crème brulée", "Triangolo", "Tentation noir",
        "Comtesse", "Noblesse", "Noir authentique",
        "Passion au lait", "Stracciatella", "Arabia"
    ]
    resize_transform = v2.Compose([
        v2.Resize((640, 640)),
        v2.ToImage(),
    ])
    test_dataset = TestDataset(test_dir, transform=resize_transform)
    # load model architecture + weights
    model_yolo = YOLO(MODEL_PATH)
    model = model_yolo.model
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    
    test_loader   = DataLoader(test_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=6, collate_fn=collate_fn)
    records = []

    for imgs, ids in tqdm(test_loader, desc=f"Evaluation", unit="batch", leave=False):
        imgs    = imgs.to(device).float() / 255.0
        with torch.no_grad():
            results = model(imgs)
        '''print("=== DEBUG: raw results ===")
        print(f"Type of results: {type(results)}")
        print(f"Number of outputs in results: {len(results)}")

        for i, out in enumerate(results):
            if isinstance(out, torch.Tensor):
                print(f"  results[{i}] is Tensor with shape {out.shape}")
            elif isinstance(out, (list, tuple)):
                print(f"  results[{i}] is {type(out).__name__} of length {len(out)}")
                for j, sub in enumerate(out):
                    if isinstance(sub, torch.Tensor):
                        print(f"    results[{i}][{j}] is Tensor with shape {sub.shape}")
                    else:
                        print(f"    results[{i}][{j}] is {type(sub).__name__}")
            else:
                print(f"  results[{i}] is {type(out).__name__}")'''

        for pred, img_id in zip(results[0], ids):
            # extract predicted class IDs
            cls_ids = count_classes(pred)

            counts = np.zeros(num_classes, dtype=int)
            for cid in cls_ids:
                cidx = model2idx.get(int(cid))
                counts[cidx] += 1

            row = {'id': img_id}
            for col_name, cnt in zip(cols, counts):
                row[col_name] = int(cnt)
            records.append(row)

    df = pd.DataFrame(records)

    df = df[["id",
        "Jelly White", "Jelly Milk", "Jelly Black",
        "Amandina", "Crème brulée", "Triangolo", "Tentation noir",
        "Comtesse", "Noblesse", "Noir authentique",
        "Passion au lait", "Stracciatella", "Arabia"
    ]].sort_values('id')
    df.to_csv(output_csv, index=False)
    print(f"Saved in {output_csv}")

if __name__ == '__main__':
    main()