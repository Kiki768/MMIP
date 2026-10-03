import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_base_tf(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

class PetDataset(Dataset):
    def __init__(self, frame, transform):
        self.paths = frame["path"].str.replace("\\", "/", regex=False).tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]

def load_split(csv="split.csv"):
    df = pd.read_csv(csv)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    class_names = df.drop_duplicates("label").sort_values("label")["breed"].tolist()
    return df, class_names

def make_loaders(df, batch_size=32, img_size=224, train_tf=None):
    base = get_base_tf(img_size)
    train_tf = train_tf or base
    def mk(split, tf, shuffle):
        sub = df[df["split"] == split]
        return DataLoader(PetDataset(sub, tf), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0, pin_memory=True)
    return mk("train", train_tf, True), mk("val", base, False), mk("test", base, False)