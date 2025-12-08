import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMPipeline, DDPMScheduler
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model (CIFAR-10: 32x32, 3 channels)
pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
model = pipeline.unet.to(device)
scheduler = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")

# Data: resize to 32x32 and convert grayscale to RGB
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
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

samples = torch.randn(16, 3, 32, 32, device=device)
for t in scheduler.timesteps:
    with torch.no_grad():
        residual = model(samples, t).sample
    samples = scheduler.step(residual, t, samples).prev_sample

samples = (samples / 2 + 0.5).clamp(0, 1).cpu()
samples_gray = samples.mean(dim=1)  # Convert back to grayscale for display

# Save
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples_gray[i], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("generated_diffusers_pretrained.png", dpi=150)
print("Saved generated_diffusers_pretrained.png")
