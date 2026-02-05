#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from torchinfo import summary

from sklearn.preprocessing import LabelEncoder

# coral
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss
from coral_pytorch.dataset import levels_from_labelbatch

from tqdm import tqdm

import seaborn as sns

import random
import torchvision.transforms.functional as TF


sns.set_theme(style="ticks", palette="pastel", rc={'lines.linewidth': 2.5})


# In[2]:


import os

folder_path = 'models/'

# Create the folder path and checkpoints directory if they don't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
if not os.path.exists(os.path.join(folder_path, 'checkpoints')):
    os.makedirs(os.path.join(folder_path, 'checkpoints'))


# In[3]:


# set device to mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# In[4]:


# modify so indices is the index of the dataframe
def read_hdf5_to_dataframe_with_index(h5_path="unified_parallel.h5"):
    with h5py.File(h5_path, "r") as f:
        viirs_start = f["viirs_start"][:]
        viirs_end = f["viirs_end"][:]
        rgb = f["rgb"][:]
        figures = f["figures"][:]
        indices = f["indices"][:]
        iso3 = f["iso3"][:]

    # Decode bytes to strings for iso3
    iso3_decoded = [x.decode("utf-8") if isinstance(x, bytes) else x for x in iso3]

    # Create a DataFrame with indices as the index
    df = pd.DataFrame(
        {
            "viirs_start": list(viirs_start),
            "viirs_end": list(viirs_end),
            "rgb": list(rgb),
            "figures": figures,
            "iso3": iso3_decoded,
        },
        index=indices,
    )

    df.sort_index(inplace=True)  # Ensure indices are sorted

    return df


# In[5]:


# path = "../src/data/processed/flood.h5"
path = '../src/data/processed/disaster.h5'
df = read_hdf5_to_dataframe_with_index(path)


# In[6]:


def replace_iso3_with_embedding(df, embedding_dim=64):
    le = LabelEncoder()
    df["iso3_id"] = le.fit_transform(df["iso3"])
    emb = nn.Embedding(
        num_embeddings=df["iso3_id"].nunique(), embedding_dim=embedding_dim
    )
    with torch.no_grad():
        embeddings = emb(torch.tensor(df["iso3_id"].values)).numpy()
    df["iso3_encoded"] = embeddings.tolist()
    # drop iso3_id
    df.drop(columns=["iso3_id"], inplace=True)
    return df, le, emb

df, le, emb = replace_iso3_with_embedding(df)


# In[7]:


def create_viirs_diff_column(df):
    df["viirs_diff"] = df["viirs_end"] - df["viirs_start"]
    df.drop(columns=["viirs_start", "viirs_end"], inplace=True)
    # transpose the rgb column to have the shape (3, 224, 224) and viirs_diff to (3, 224, 224)
    df["rgb"] = df["rgb"].apply(
        lambda x: np.transpose(x, (2, 0, 1)) if len(x.shape) == 3 else x
    )
    df["viirs_diff"] = df["viirs_diff"].apply(
        lambda x: np.transpose(x, (2, 0, 1)) if len(x.shape) == 3 else x
    )
    return df
df = create_viirs_diff_column(df)


# In[9]:


normalization_stats = {}


def log_and_normalize_column(df, column_name):
    df[column_name] = np.log1p(df[column_name])
    mean = df[column_name].mean()
    std = df[column_name].std()
    normalization_stats[column_name] = {"mean": mean, "std": std}
    df[column_name] = (df[column_name] - mean) / std
    return df

df = log_and_normalize_column(df, "figures")


# In[10]:


log_figures = df["figures"].explode().dropna()

quantiles = log_figures.quantile([0.2, 0.4, 0.6, 0.8])
q1, q2, q3, q4 = quantiles[0.2], quantiles[0.4], quantiles[0.6], quantiles[0.8]
print(f"Log1p Quantiles: {q1}, {q2}, {q3}, {q4}")


def label_figures(row):
    val = row["figures"]
    if val <= q1:
        return 0
    elif val <= q2:
        return 1
    elif val <= q3:
        return 2
    elif val <= q4:
        return 3
    else:
        return 4


df["label"] = df.apply(label_figures, axis=1)


# In[ ]:


df.sample(20)


# - Neural network

# In[11]:


class MultiBranchCNNOrdinal(nn.Module):
    def __init__(self, output_dim=5):  
        super().__init__()

        self.rgb_model = models.resnet50(pretrained=True)
        self.rgb_model.fc = nn.Identity() 
        rgb_output_dim = 2048

        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.viirs_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # Output shape: (batch_size, 64)
        )

        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),  # RGB + VIIRS + country_embedding
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.coral = CoralLayer(64, num_classes=output_dim)

    def forward(
        self, rgb_img, viirs_img, country_embedding
    ): 
        rgb_feat = self.rgb_model(rgb_img)
        rgb_proj = self.rgb_proj(rgb_feat)

        viirs_feat = self.viirs_cnn(viirs_img)

        fused = torch.cat(
            [rgb_proj, viirs_feat, country_embedding], dim=1
        ) 
        features = self.mlp(fused)

        return self.coral(features)


# - data loader

# In[12]:


target_size = (224, 224) 


class JointTransform:
    def __init__(self, target_size=target_size):
        self.target_size = target_size
        self.resize_rgb = transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=False,
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        # self.normalize_rgb = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        self.normalize_viirs = transforms.Normalize(mean=[0], std=[1])

    def __call__(self, rgb, viirs):
        # Resize only RGB (PIL)
        rgb = self.resize_rgb(rgb)
        # rgb = self.color_jitter(rgb)

        # Random horizontal flip
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            viirs = TF.hflip(viirs)

        # Random vertical flip
        if random.random() > 0.5:
            rgb = TF.vflip(rgb)
            viirs = TF.vflip(viirs)

        # Normalize
        # rgb = self.normalize_rgb(rgb)
        viirs = self.normalize_viirs(viirs)

        return rgb, viirs


# In[13]:


class IDPDatasetOrdinal(Dataset):
    def __init__(self, df, joint_transform=None, device="cpu"):
        self.df = df
        self.joint_transform = joint_transform
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load tensors
        viirs = torch.tensor(self.df.iloc[idx]["viirs_diff"]).float()
        if viirs.ndim == 2:
            viirs = viirs.unsqueeze(0)  # (1, H, W)

        rgb = torch.tensor(self.df.iloc[idx]["rgb"]).float()
        if rgb.ndim == 3 and rgb.shape[0] != 3:
            rgb = rgb.permute(2, 0, 1)  # Ensure (3, H, W)

        label = torch.tensor(self.df.iloc[idx]["label"]).long()
        iso3 = self.df.iloc[idx]["iso3"]
        iso3_emb = torch.tensor(self.df.iloc[idx]["iso3_encoded"]).float()

        # Apply joint transforms
        if self.joint_transform:
            rgb, viirs = self.joint_transform(rgb, viirs)

        return (
            rgb.to(self.device),
            viirs.to(self.device),
            label.to(self.device),
            iso3,
            iso3_emb.to(self.device),
        )


# In[14]:


batch_size = 32

joint_transform = JointTransform()
dataset = IDPDatasetOrdinal(df, joint_transform=joint_transform, device="mps")
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[15]:


train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MultiBranchCNNOrdinal(output_dim=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

summary(
    model,
    input_size=((batch_size, 3, 224, 224), (batch_size, 1, 20, 20), 
                (batch_size, 64)), 
    device=device.type,
)


# In[16]:


def coral_criterion(logits, labels, num_classes):
    levels = levels_from_labelbatch(labels, num_classes=num_classes).to(logits.device)
    return coral_loss(logits, levels)


def train_one_epoch(model, dataloader, optimizer, num_classes, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for rgb_imgs, viirs_imgs, labels, _, iso3_emb in tqdm(dataloader, desc="Training"):
        rgb_imgs, viirs_imgs, labels, iso3_emb = (
            rgb_imgs.to(device),
            viirs_imgs.to(device),
            labels.to(device),
            iso3_emb.to(device)
        )

        optimizer.zero_grad()
        logits = model(rgb_imgs, viirs_imgs, iso3_emb)
        loss = coral_criterion(logits, labels, num_classes)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        # Prediction from logits (ordinal decode)
        pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def validate(model, dataloader, num_classes, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for rgb_imgs, viirs_imgs, labels, _, iso3_emb in tqdm(dataloader, desc="Validating"):
            rgb_imgs, viirs_imgs, labels, iso3_emb = (
                rgb_imgs.to(device),
                viirs_imgs.to(device),
                labels.to(device),
                iso3_emb.to(device)
            )
            logits = model(rgb_imgs, viirs_imgs, iso3_emb)
            loss = coral_criterion(logits, labels, num_classes)

            running_loss += loss.item() * labels.size(0)

            pred = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


# In[17]:


num_classes = 5  # Adjust based on your dataset
num_epochs = 400

model = MultiBranchCNNOrdinal(output_dim=num_classes).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)

best_val_acc = 0.0


# In[ ]:


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, num_classes, device
    )
    val_loss, val_acc = validate(model, val_loader, num_classes, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "./models/checkpoints/best_model_ordinal.pth")
        print("✅ Best model saved.")


# In[ ]:


model = MultiBranchCNNOrdinal(output_dim=num_classes).to(device)

# Load saved weights
model.load_state_dict(torch.load("./models/checkpoints/best_model_ordinal.pth", map_location=device))
model.eval()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

all_preds = []
all_labels = []

with torch.no_grad():
    for rgb_imgs, viirs_imgs, labels, _, iso3_emb in test_loader:
        rgb_imgs = rgb_imgs.to(device)
        viirs_imgs = viirs_imgs.to(device)
        labels = labels.to(device)
        iso3_emb = iso3_emb.to(device)

        logits = model(rgb_imgs, viirs_imgs, iso3_emb)

        # CORAL prediction: count how many sigmoid outputs > 0.5
        preds = torch.sum(torch.sigmoid(logits) > 0.5, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# In[ ]:


# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds, normalize='true')
class_names = ["0", "1", "2", "3", "4"]

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
# save to pdf
plt.savefig("cm.pdf")
plt.show()


# In[ ]:


# classification report
print(classification_report(all_labels, all_preds, target_names=class_names))

