import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import math

device = "cuda" if torch.cuda.is_available() else "cpu"


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.gelu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = F.gelu(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class Attention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(c), dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(b, c, h, w)
        return x + self.proj(out)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4)):
        super().__init__()
        time_dim = ch * 4
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(ch),
            nn.Linear(ch, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = ch
        for mult in ch_mult:
            cur_ch = ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(prev_ch, cur_ch, time_dim),
                ResBlock(cur_ch, cur_ch, time_dim),
                Attention(cur_ch) if mult == ch_mult[-1] else nn.Identity(),
            ]))
            self.down_samples.append(nn.Conv2d(cur_ch, cur_ch, 3, stride=2, padding=1))
            prev_ch = cur_ch

        # Middle
        self.mid_block1 = ResBlock(prev_ch, prev_ch, time_dim)
        self.mid_attn = Attention(prev_ch)
        self.mid_block2 = ResBlock(prev_ch, prev_ch, time_dim)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(ch_mult):
            cur_ch = ch * mult
            self.up_samples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.up_blocks.append(nn.ModuleList([
                ResBlock(prev_ch + cur_ch, cur_ch, time_dim),
                ResBlock(cur_ch, cur_ch, time_dim),
                Attention(cur_ch) if mult == ch_mult[-1] else nn.Identity(),
            ]))
            prev_ch = cur_ch

        self.final = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.init_conv(x)

        skips = []
        skip_sizes = []
        for (res1, res2, attn), down in zip(self.down_blocks, self.down_samples):
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)
            skips.append(h)
            skip_sizes.append(h.shape[2:])
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for (res1, res2, attn), up in zip(self.up_blocks, self.up_samples):
            h = up(h)
            skip = skips.pop()
            target_size = skip.shape[2:]
            h = F.interpolate(h, size=target_size, mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb)
            h = res2(h, t_emb)
            h = attn(h)

        return self.final(h)


class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x, noise, t):
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x + sqrt_one_minus * noise

    @torch.no_grad()
    def sample_step(self, model, x, t):
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = model(x, t_tensor)

        alpha = self.alphas[t]
        beta = self.betas[t]

        mean = (1 / torch.sqrt(alpha)) * (x - (beta / self.sqrt_one_minus_alphas_cumprod[t]) * pred_noise)

        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta)
            return mean + variance * noise
        return mean


# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Model and scheduler
model = UNet(in_ch=1, out_ch=1, ch=64, ch_mult=(1, 2, 4)).to(device)
scheduler = NoiseScheduler(timesteps=1000, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training
num_epochs = 20
for epoch in range(num_epochs):
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)
        batch_size = images.shape[0]

        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
        noise = torch.randn_like(images)
        noisy = scheduler.add_noise(images, noise, t)

        pred = model(noisy, t)
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Generation
model.eval()
samples = torch.randn(16, 1, 28, 28, device=device)

for t in reversed(range(scheduler.timesteps)):
    samples = scheduler.sample_step(model, samples, t)

samples = (samples / 2 + 0.5).clamp(0, 1).cpu()

# Save
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("generated_pytorch.png", dpi=150)
print("Saved generated_pytorch.png")
