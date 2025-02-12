from model_zoo.vqvae import VQVAE
from model_zoo.diffusion_model_unet import DiffusionModelUNet
from model_zoo.inferers.inferer import LatentDiffusionInferer
from model_zoo.schedulers.ddpm import DDPMScheduler
import torch
from torch.utils.data import DataLoader
import nibabel as nib
from data.loaders.atlas_3d_loader import Atlas3DDataset, Atlas3DLoader
import matplotlib.pyplot as plt
import torchvision

# Load the model
model = VQVAE(
    spatial_dims=3,
    in_channels=1,
    out_channels=1, 
    num_channels=[128, 256],
    num_res_layers=2,
    num_res_channels=[128, 256],
    downsample_parameters=[(2,4,1,1), (2,4,1,1)],
    upsample_parameters=[(2,4,1,1,0), (2,4,1,1,0)],
    num_embeddings=512,
    embedding_dim=32,
    output_act="tanh",
)
model.load_state_dict(torch.load("projects/3dmededit/weights/runs/2025_02_04_11_07_16_544669/best_epoch_990.pt")["model_weights"])
model.to("cuda")
model.eval()

# Load the data
dataset_path = "data/atlas_skull_stripped/atlas_skull_stripped_test.csv"
target_size = (128, 128, 128)

ds = Atlas3DDataset(data_dir=dataset_path, target_size=target_size)
dl_args = {
    "target_size": target_size,
    "batch_size": 1,
    "dataset_module": {
        "module_name": "data.loaders.atlas_3d_loader",
        "class_name": "Atlas3DDataset",
    },
    "data_dir": {
        "train": [dataset_path], 
    }
}
dl = Atlas3DLoader(dl_args).train_dataloader()

# Get the first sample
sample = next(iter(dl))[0]
print(sample.shape)
sample = sample.to("cuda")


# Get the model output
recon, _ = model(sample)

# add noise to the image with varying steps
for i in range(1, 1001, 100):
    noisy = sample
    for j in range(i):
        noise = torch.Tensor(sample.shape).normal_(0, 0.03).to("cuda")
        noisy = noisy + noise
    noisy_recon, _ = model(noisy)
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(sample[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
    axes[1].imshow(noisy[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
    axes[2].imshow(noisy_recon[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
    plt.savefig(f"projects/3dmededit/noisy_recon_{i}.png")

# get latent and add noise to that
latent_sample = model.encode_stage_2_inputs(sample) # shape: [1, 32, 32, 32, 32]

fig, axes = plt.subplots(5, 3)
for i in range(5):
    axes[i, 0].imshow(latent_sample[0, i, 16, :, :].detach().cpu().numpy(), cmap="gray")
    axes[i, 1].imshow(latent_sample[0, i, :, 16, :].detach().cpu().numpy(), cmap="gray")
    axes[i, 2].imshow(latent_sample[0, i, :, :, 16].detach().cpu().numpy(), cmap="gray")
plt.savefig(f"projects/3dmededit/latent_sample.png")

# add noise to the latent
for i in range(1, 1001, 100):
    noisy = latent_sample
    for j in range(i):
        noise = torch.Tensor(latent_sample.shape).normal_(0, 0.03).to("cuda")
        noisy = noisy + noise
    noisy_recon = model.decode_stage_2_outputs(noisy)
    fig, axes = plt.subplots(4, 4)
    for row in range(4):
        axes[row, 0].imshow(sample[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 1].imshow(latent_sample[0, row, 16, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 2].imshow(noisy[0, row, 16, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 3].imshow(noisy_recon[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
    plt.savefig(f"projects/3dmededit/latent_recon_{i}.png")

# # noise using MONAI scheduler
scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    schedule="linear_beta",
    beta_start=0.0001,
    beta_end=0.02,
)

# noise = torch.randn_like(latent_sample)
# fig, axes = plt.subplots(1, 10, figsize=(5, 1))
# for i in range(0, 1000, 100):
#     noisy = scheduler.add_noise(original_samples=latent_sample, noise=noise, timesteps=torch.tensor([i]))
#     axes[i//100].imshow(noisy[0, 3, 16, :, :].detach().cpu().numpy(), cmap="gray")
#     axes[i//100].axis("off")
# plt.savefig("projects/3dmededit/scheduler_latent.png")

for i in range(0, 1000, 100):
    noise = torch.randn_like(noisy)
    noisy = scheduler.add_noise(original_samples=latent_sample, noise=noise, timesteps=torch.tensor([i]))
    noisy_recon = model.decode_stage_2_outputs(noisy)
    fig, axes = plt.subplots(4, 4)
    for row in range(4):
        axes[row, 0].imshow(sample[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 1].imshow(latent_sample[0, row, 16, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 2].imshow(noisy[0, row, 16, :, :].detach().cpu().numpy(), cmap="gray")
        axes[row, 3].imshow(noisy_recon[0, 0, 64, :, :].detach().cpu().numpy(), cmap="gray")
    plt.savefig(f"projects/3dmededit/scheduler_latent_recon_{i}.png")

# import cv2
# cat =cv2.imread("projects/3dmededit/test_cat.jpg")
# # cat = torch.tensor(cat).permute(2, 0, 1).unsqueeze(0).float()
# cat = torchvision.transforms.ToTensor()(cat)
# print(cat.shape)
# noise = torch.randn_like(cat)
# fig, axes = plt.subplots(1, 10)
# for i in range(0, 1000, 100):
#     noisy_cat = scheduler.add_noise(original_samples=cat, noise=noise, timesteps=torch.tensor([i]))
#     axes[i//100].imshow(noisy_cat.permute(1, 2, 0).numpy(), cmap="gray")
#     axes[i//100].axis("off")
# plt.savefig("projects/3dmededit/scheduler_cat.png")
