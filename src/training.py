#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import h5py

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
from torchinfo import summary
from sklearn.metrics import r2_score
import plotly.express as px


from sklearn.preprocessing import LabelEncoder

import random
import torchvision.transforms.functional as TF

import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="ticks", palette="pastel", rc={"lines.linewidth": 2.5})

# Set the random seed for reproducibility
torch.manual_seed(42)
generator = torch.Generator().manual_seed(42)

# Set the device to use for training
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# # Data

# In[2]:


folder_path = "models/"

# Create the folder path and checkpoints directory if they don't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if not os.path.exists(os.path.join(folder_path, "checkpoints")):
    os.makedirs(os.path.join(folder_path, "checkpoints"))


# - Reading files

# In[3]:


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


# In[4]:


path = "../src/data/processed/disaster.h5"
df = read_hdf5_to_dataframe_with_index(path)


# - encode the iso3

# In[5]:


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


# - Process to create virrs (end - start)

# In[6]:


def create_viirs_diff_column(df):
    df["viirs_diff"] = df["viirs_end"] - df["viirs_start"]
    df.drop(columns=["viirs_start", "viirs_end"], inplace=True)
    # transpose the rgb column to have the shape (3, 224, 224) and viirs_diff to (3, 224, 224)
    df["rgb"] = df["rgb"].apply(
        lambda x: np.transpose(x, (2, 0, 1)) if len(x.shape) == 3 else x
    )
    return df


# - Normalization

# In[7]:


normalization_stats = {}

def log_and_normalize_column(df, column_name):
    df[column_name] = np.log1p(df[column_name])
    mean = df[column_name].mean()
    std = df[column_name].std()
    normalization_stats[column_name] = {"mean": mean, "std": std}
    df[column_name] = (df[column_name] - mean) / std
    return df


# In[8]:


df = create_viirs_diff_column(df)
df = log_and_normalize_column(df, "figures")


# In[9]:


df


# - Fitlering data that has those zeroes...

# In[10]:


def calculate_variance(array):
    return np.var(array)

def remove_low_variance_images(df, threshold=0.0005):
    def has_low_variance(image):
        return calculate_variance(image) < threshold
    mask = df.apply(
        lambda row: has_low_variance(row["rgb"]),  # or has_low_variance(row["viirs_diff"]),
        axis=1,
    )
    return df[~mask]

# threshold_var = 0.00001
# threshold_var = 0.001

df_old = df.copy()

print(f"Dataset size before variance filtering: {len(df_old)}")
# df = remove_low_variance_images(df_old, threshold=threshold_var)
print(f"Dataset size after variance filtering: {len(df)}")


# # NN

# - Architecture

# In[11]:


class MultiBranchCNN(nn.Module):
    def __init__(self, output_dim=1):
        super(MultiBranchCNN, self).__init__()

        # === ResNet-50 Feature Extractor ===
        self.rgb_model = models.resnet50(pretrained=True)
        for param in self.rgb_model.parameters():
            param.requires_grad = True
        self.rgb_model.fc = nn.Identity()
        rgb_output_dim = 2048

        # === RGB Projection Head ===
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  # final output is 64
            nn.ReLU(),
        )

        # === VIIRS CNN Branch ===
        self.viirs_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (32, H, W)
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: (64, 1, 1)
            nn.Flatten(),  # Final output: (64,)
        )

        # === Fusion MLP ===
        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(
        self, rgb_img, viirs_img, country_embedding
    ):  # country_embedding is (batch_size, 64)
        rgb_feat = self.rgb_model(rgb_img)
        rgb_proj = self.rgb_proj(rgb_feat)  # (batch_size, 64)

        viirs_feat = self.viirs_cnn(viirs_img)  # (batch_size, 64)

        fused = torch.cat(
            [rgb_proj, viirs_feat, country_embedding], dim=1
        )  # (batch_size, 192)
        output = self.mlp(fused)
        return output.squeeze(1)  # for regression; remove this for classification


# - Hyperparameters

# In[12]:


batch_size = 64  # we reached .7 training on this .15 for test (64, and 1e-5 works too)
learning_rate = 1e-4 # e-5 was fine , batch size 32 (.37 train, .09 test)
weight_decay = 1e-5 
num_epochs = 400


# - Transform

# In[13]:


target_size = (224, 224)  # Define the target size for resizing images

class JointTransform:
    def __init__(self, target_size=target_size):
        self.target_size = target_size
        self.resize_rgb = transforms.Resize(
            target_size, interpolation=transforms.InterpolationMode.BILINEAR, antialias=False
        )
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        self.normalize_rgb = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.normalize_viirs = transforms.Normalize(mean=[0], std=[1])

    def __call__(self, rgb, viirs):
        # Resize only RGB (PIL)
        rgb = self.resize_rgb(rgb)
        rgb = self.color_jitter(rgb)
        
        # Random horizontal flip
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            viirs = TF.hflip(viirs)

        # Random vertical flip
        if random.random() > 0.5:
            rgb = TF.vflip(rgb)
            viirs = TF.vflip(viirs)

        # Normalize
        rgb = self.normalize_rgb(rgb)
        # viirs = self.normalize_viirs(viirs)

        return rgb, viirs


# - Dataset and DataLoader

# In[14]:


class IDPDataset(Dataset):
    def __init__(self, df, joint_transform=None):
        self.df = df
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        viirs = (
            torch.tensor(self.df.iloc[idx]["viirs_diff"]).float().squeeze().to(device)
        )

        if viirs.ndim == 2:
            viirs = viirs.unsqueeze(0)  # (1, H, W)

        rgb = torch.tensor(self.df.iloc[idx]["rgb"]).float().squeeze().to(device)

        target = torch.tensor(self.df.iloc[idx]["figures"]).float().to(device)
        iso3 = self.df.iloc[idx]["iso3"]
        iso3_emb = torch.tensor(self.df.iloc[idx]["iso3_encoded"]).to(device)

        if self.joint_transform:
            rgb, viirs = self.joint_transform(rgb, viirs)

        return rgb, viirs, target, iso3, iso3_emb


# In[15]:


# Create the transform
joint_transform = JointTransform(target_size=(224, 224))

# Create the dataset
dataset = IDPDataset(df=df, joint_transform=joint_transform)


# In[16]:


dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# - Data split

# In[17]:


def create_data_loaders(dataset, batch_size, generator, split_ratio=(0.8, 0.1, 0.1)):
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1.0"

    total = len(dataset)
    train_size = int(split_ratio[0] * total)
    val_size = int(split_ratio[1] * total)
    test_size = total - train_size - val_size  # Ensure full coverage

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# In[18]:


train_loader, val_loader, test_loader = create_data_loaders(
    dataset, batch_size, generator
)


# - Initiate model, loss and optimizer

# In[19]:


model = MultiBranchCNN(output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scheduler = ReduceLROnPlateau(
#     optimizer, mode="min", factor=0.5, patience=5, verbose=True
# )


# In[20]:


summary(
    model,
    input_size=(
        ( batch_size, 3, 224, 224),
        ( batch_size, 1, 20, 20),
        ( batch_size, 64)
    ),
    device=device.type,
)


# # Training

# In[21]:


train_size = len(train_loader.dataset)
val_size = len(val_loader.dataset)


# In[ ]:


best_loss = float("inf")
patience = 10
patience_counter = 0
train_losses = []
val_losses = []

print(f"Training on {train_size} examples, validating on {val_size} examples")

for epoch in range(num_epochs):

    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")
        
    # Training phase
    model.train()
    epoch_train_loss = 0.0

    for rgb, viirs, targets, _ , iso3_emb in train_loader:
        # Move data to device
        rgb = rgb.to(device)
        viirs = viirs.to(device)
        targets = targets.to(device)
        iso3_emb = iso3_emb.to(device)

        # Forward pass
        outputs = model(rgb, viirs, iso3_emb)
        loss = criterion(outputs, targets)
        epoch_train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for rgb, viirs, targets, _, iso3_emb in val_loader:
            rgb = rgb.to(device)
            viirs = viirs.to(device)
            targets = targets.to(device)
            iso3_emb = iso3_emb.to(device)

            outputs = model(rgb, viirs, iso3_emb)
            val_loss = criterion(outputs, targets)
            epoch_val_loss += val_loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    # scheduler.step(epoch_val_loss)

    # Print epoch statistics
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
    )

    # Early stopping
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), folder_path + "checkpoints/best_model.pth")
        print(f" ✅ Saved with validation loss: {best_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


# In[ ]:


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


# if using ReduceLROnPlateau, restore lr
if "scheduler" in locals():
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate  # Reset learning rate to initial value


# - Inference using best model

# In[ ]:


model.load_state_dict(torch.load(folder_path + "checkpoints/best_model.pth"))


# In[ ]:


def predict_batch(model, batch, device):
    """Processes a single batch and returns predictions and targets."""
    rgb, viirs, target, _, iso3_emb = batch
    rgb = rgb.to(device)
    viirs = viirs.to(device)
    iso3_emb = iso3_emb.to(device)

    output = model(rgb, viirs, iso3_emb)
    return output.cpu().numpy(), target.cpu().numpy()


def get_predictions_and_targets(model, dataloader, device):
    """Gets predictions and targets for the entire dataloader."""
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch_preds, batch_targets = predict_batch(model, batch, device)
            predictions.extend(batch_preds)
            targets.extend(batch_targets)

    return np.array(predictions), np.array(targets)


def plot_scatter(ax, targets, predictions, title):
    """Plots a scatter plot with predictions vs targets."""
    r2 = r2_score(targets, predictions)
    ax.scatter(targets, predictions, alpha=0.3)
    ax.plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "r--",
        lw=2,
    )
    ax.set_title(f"{title}\n$R^2$ = {r2:.4f}")
    ax.set_xlabel("True IDP")
    ax.set_ylabel("Predicted IDP")


def plot_predictions_vs_targets(
    model, device, *data_loaders, save=False, save_path="predictions_vs_targets.pdf"
):

    num_sets = len(data_loaders)
    fig, axes = plt.subplots(
        1, num_sets, figsize=(7 * num_sets, 6), sharex=True, sharey=True
    )

    if num_sets == 1:
        axes = [axes] 

    for ax, (label, loader) in zip(axes, data_loaders):
        preds, targets = get_predictions_and_targets(model, loader, device)
        plot_scatter(ax, targets, preds, f"{label} Set")

    # plt.suptitle("Predictions vs Targets")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, format=save_path.split(".")[-1], bbox_inches="tight")
        print(f"Plot saved to: {os.path.abspath(save_path)}")
        plt.show(block=False)  # Show the plot without blocking
        plt.close()
    else:
        plt.show()


# In[ ]:


plot_predictions_vs_targets(
    model, device, ("Train", train_loader), ("Test", test_loader), save=True, save_path="predictions_vs_targets.pdf"
)


# In[ ]:


plot_predictions_vs_targets(model, device, ("Test", test_loader), save=True, save_path="scatter_idmc.pdf")


# # IDU

# In[ ]:


path_idu = "../src/data/processed/testing.h5"
df_idu = read_hdf5_to_dataframe_with_index(path_idu)


# In[ ]:


df_idu, le_idu, emb_idu = replace_iso3_with_embedding(df_idu)
df_idu = create_viirs_diff_column(df_idu)
df_idu = log_and_normalize_column(df_idu, "figures")


# In[ ]:


dataset_idu = IDPDataset(df=df_idu, joint_transform=joint_transform)


# In[ ]:


# split the dataset into train, val, test
train_loader_idu, val_loader_idu, test_loader_idu = create_data_loaders(dataset_idu, batch_size, generator)


# In[ ]:


train_size_idu = len(train_loader_idu.dataset)
val_size_idu = len(val_loader_idu.dataset)
test_size_idu = len(test_loader_idu.dataset)


# - Fine-tuning

# In[ ]:


# finetune model on idu
best_loss = float("inf")
patience = 10
patience_counter = 0
train_losses = []
val_losses = []

print(f"Training on {train_size_idu} examples, validating on {val_size_idu} examples")

for epoch in range(num_epochs):
    for param_group in optimizer.param_groups:
        print(f"Current learning rate: {param_group['lr']}")

    # Training phase
    model.train()
    epoch_train_loss = 0.0

    for rgb, viirs, targets, _, iso3_emb in train_loader_idu:
        # Move data to device
        rgb = rgb.to(device)
        viirs = viirs.to(device)
        targets = targets.to(device)
        iso3_emb = iso3_emb.to(device)

        # Forward pass
        outputs = model(rgb, viirs, iso3_emb)
        loss = criterion(outputs, targets)
        epoch_train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0

    with torch.no_grad():
        for rgb, viirs, targets, _, iso3_emb in val_loader_idu:
            rgb = rgb.to(device)
            viirs = viirs.to(device)
            targets = targets.to(device)
            iso3_emb = iso3_emb.to(device)

            outputs = model(rgb, viirs, iso3_emb)
            val_loss = criterion(outputs, targets)
            epoch_val_loss += val_loss.item()

    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)
    # scheduler.step(epoch_val_loss)

    # Print epoch statistics
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}"
    )

    # Early stopping
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), folder_path + "checkpoints/best_model.pth")
        print(f" ✅ Saved with validation loss: {best_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


# In[ ]:


plot_predictions_vs_targets(model, device, ("Test", test_loader_idu), save=True, save_path="scatter_idu.pdf")


# - Per country error

# In[ ]:


def evaluate_per_country(model, dataloader):
    model.eval()
    country_predictions = {}
    country_targets = {}

    with torch.no_grad():
        for rgb, viirs, target, iso, iso3_emb in dataloader:
            rgb = rgb.to(device)
            viirs = viirs.to(device)
            output = model(rgb, viirs, iso3_emb)

            # Move tensors to CPU before converting to numpy
            output_cpu = output.cpu()
            target_cpu = target.cpu()

            for i in range(len(iso)):
                code = iso[i]
                if code not in country_predictions:
                    country_predictions[code] = []
                    country_targets[code] = []
                country_predictions[code].append(output_cpu[i].item())
                country_targets[code].append(target_cpu[i].item())

    return country_predictions, country_targets


# Run evaluation
country_predictions, country_targets = evaluate_per_country(model, dataloader)

# Compute Mean Absolute Error (MAE) per country
country_mae = {
    iso3: np.mean(
        np.abs(
            np.array(country_predictions[iso3], dtype=np.float32)
            - np.array(country_targets[iso3], dtype=np.float32)
        )
    )
    for iso3 in country_predictions
}

# Create a dataframe with country codes and their MAE values
mae_df = pd.DataFrame(
    {"iso3": list(country_mae.keys()), "mae": list(country_mae.values())}
)

# Plot choropleth map using Plotly
fig = px.choropleth(
    mae_df,
    locations="iso3",  # ISO 3-letter codes
    color="mae",  # Color scale based on MAE
    hover_name="iso3",
    color_continuous_scale="Reds",
    # title="Mean Absolute Error by Country",
)

fig.update_geos(
    showframe=False, showcoastlines=False, projection_type="equirectangular"
)
fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
fig.show()

