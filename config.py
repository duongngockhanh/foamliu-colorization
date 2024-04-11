import os

import scipy.io
import numpy as np
import torch

img_rows, img_cols = 256, 256
channel = 3
num_classes = 313
epsilon = 1e-8
epsilon_sqr = epsilon ** 2
nb_neighbors = 5
T = 0.38 # temperature parameter T

default_pretrained = "eccv16_pretrained.pth"
use_default_pretrained = True
freeze_default_pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
q_ab = np.load("data/pts_in_hull.npy")
mat = scipy.io.loadmat('human_colormap.mat')
color_map = (mat['colormap'] * 256).astype(np.int32)

# Load the color prior factor that encourages rare colors
prior_factor = torch.from_numpy(np.load("data/prior_factor.npy")).to(device)
weights_1 = prior_factor.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

prior_probs = torch.from_numpy(np.load("data/prior_prob.npy")).to(device)
weights_2 = 1 / (0.5 * prior_probs + 0.5 / 313)
weights_2 = weights_2 / sum(prior_probs * weights_2)

# Hyperparameters
epochs = 100
lr = 1e-4
train_num_max = 60
val_num_max = 60
pretrained = None
save_dir = "exp_Zhang_Cla_Lab"
loss_type = 2

train_root = "/kaggle/input/small-coco-stuff/small-coco-stuff/train2017/train2017"
val_root = "/kaggle/input/small-coco-stuff/small-coco-stuff/train2017/train2017"
train_batch_size = 32
val_batch_size = 8

# Save weight
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
saved_weights = sorted(os.listdir(save_dir))
if len(saved_weights) == 0:
    saved_weight_file = "exp01.pt"
    saved_weight_path = os.path.join(save_dir, saved_weight_file)
else:
    saved_weight_file = f"exp{int(saved_weights[-1][3:-3]) + 1:02d}.pt"
    saved_weight_path = os.path.join(save_dir, saved_weight_file)

# Use WanDB
use_wandb = True 
wandb_proj_name = "Zhang_Cla_Lab_0411"
wandb_config = {
    "dataset": "coco-stuff",
    "model": "Zhang_Cla_Lab",
    "epochs": epochs,
    "lr": lr,
    "criterion": "categorical_crossentropy",
    "optimizer": "Adam",
    "train_num_max": train_num_max,
    "val_num_max": val_num_max,
    "pretrained": pretrained,
    "saved_weight_path": saved_weight_path
}
