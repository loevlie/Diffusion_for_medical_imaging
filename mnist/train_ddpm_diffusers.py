import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
).to(device)

# Scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.shape[0]

        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(images)
        noisy_images = scheduler.add_noise(images, noise, timesteps)

        pred = model(noisy_images, timesteps).sample
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Generation
model.eval()
scheduler.set_timesteps(1000)

samples = torch.randn(16, 1, 28, 28, device=device)
for t in scheduler.timesteps:
    with torch.no_grad():
        residual = model(samples, t).sample
    samples = scheduler.step(residual, t, samples).prev_sample

samples = (samples / 2 + 0.5).clamp(0, 1).cpu()

# Save
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("generated_diffusers.png", dpi=150)
print("Saved generated_diffusers.png")
