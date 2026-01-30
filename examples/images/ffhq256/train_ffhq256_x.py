import copy
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import torch
from absl import app, flags
from tqdm import trange
from diffusers import AutoencoderKL
from tqdm import tqdm
from utils_cifar import ema, infiniteloop

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

from torchcfm.models.unet.unet import UNetModelWrapper
from models.EDM import DhariwalUNet
from torchdyn.core import NeuralODE
from torchvision.utils import save_image

# 🔧 MOD: 8-bit optimizer
import bitsandbytes as bnb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



device = "cuda"

transform = T.Compose([
    T.Resize(256),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

@torch.no_grad()
def encode_ffhq_to_latents(
    image_dir,
    latent_dir,
    vae,
    batch_size=16,
    scaling_factor=0.18215,
):
    os.makedirs(latent_dir, exist_ok=True)

    image_paths = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".png", ".jpg"))
    ])

    vae.eval()

    batch_imgs = []
    batch_names = []
    os.makedirs(latent_dir, exist_ok=True)

    for name in tqdm(image_paths):
        latent_path = os.path.join(
            latent_dir, name.replace(".png", ".pt").replace(".jpg", ".pt")
        )

        if os.path.exists(latent_path):
            continue

        img = Image.open(os.path.join(image_dir, name)).convert("RGB")
        img = transform(img)
        batch_imgs.append(img)
        batch_names.append(name)

        if len(batch_imgs) == batch_size:
            _encode_and_save(
                batch_imgs, batch_names, latent_dir, vae, scaling_factor
            )
            batch_imgs, batch_names = [], []

    if len(batch_imgs) > 0:
        _encode_and_save(
            batch_imgs, batch_names, latent_dir, vae, scaling_factor
        )


def _encode_and_save(imgs, names, latent_dir, vae, scaling_factor):
    x = torch.stack(imgs).to(device)

    with torch.autocast("cuda"):
        z = vae.encode(x).latent_dist.sample()
        z = z * scaling_factor

    for zi, name in zip(z, names):
        torch.save(
            zi.cpu(),
            os.path.join(
                latent_dir,
                name.replace(".png", ".pt").replace(".jpg", ".pt"),
            ),
        )

# =====================================================
# Sampling (VAE + Latent FM)  ——  FP32 for stability
from torchdiffeq import odeint_adjoint as odeint
from functools import partial
def sample_from_model(model, x_0):
    t = torch.tensor([0.0, 1.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    return fake_image
# =====================================================
def generate_samples_vae_x0(model, vae, parallel, savedir, step, x0, net_="normal"):
    model.eval()
    model_ = copy.deepcopy(model)

    if parallel:
        model_ = model_.module.to(device)

    B = 16
    x0 = x0.to(device)
    y = None  # 🔧 MOD: unconditional
    sample_model = partial(model, y=y)

    with torch.no_grad():
        with torch.autocast("cuda"):
            z = sample_from_model(sample_model, x0)[-1]
        #print(z.shape)
        z = z / 0.18215
        x = vae.decode(z).sample
        x = (x.clamp(-1, 1) + 1) / 2.0

    save_image(
        x,
        f"{savedir}/{net_}_generated_step_{step}.png",
        nrow=4,
    )

    model.train()


FLAGS = flags.FLAGS

# =====================================================
# Args
# =====================================================
flags.DEFINE_string("model", "otcfm", help="['otcfm', 'icfm', 'fm', 'si']")
flags.DEFINE_string("output_dir", "./resultsx/", help="output directory")
flags.DEFINE_string("ffhq_root", "./ffhq256", help="FFHQ-256 root")
flags.DEFINE_string(
    "pretrained_autoencoder_ckpt",
    "stabilityai/sd-vae-ft-mse",
    help="VAE checkpoint",
)

# UNet
flags.DEFINE_integer("num_channel", 256, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient clip")
flags.DEFINE_integer("total_steps", 400001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="lr warmup steps")
flags.DEFINE_integer("batch_size", 32, help="batch size (4090: 32~64)")
flags.DEFINE_integer("num_workers", 8, help="dataloader workers")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay")
flags.DEFINE_bool("parallel", False, help="use DataParallel")

# Saving
flags.DEFINE_integer("save_step", 20000, help="save frequency")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Dataset
# =====================================================
class FFHQLatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.paths = sorted([
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir)
            if f.endswith(".pt")
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.load(self.paths[i])

import random
import torchvision.transforms.functional as TF
def random_gaussian_blur_tensor(x, p=0.5, kernel_size=5, sigma=(0.1, 2.0)):
    if random.random() < p:
        s = random.uniform(*sigma)
        return TF.gaussian_blur(x, kernel_size, sigma=s)
    return x
def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup
import torch.nn.functional as F
# =====================================================
# Train
# =====================================================
def train(argv):

    # ---------------------
    # VAE (frozen)
    # ---------------------
    vae = AutoencoderKL.from_pretrained(
        FLAGS.pretrained_autoencoder_ckpt
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    encode_ffhq_to_latents(
    image_dir="./ffhq256",
    latent_dir="./ffhq256_latents",
    vae=vae,
    batch_size=16,   # 4090 可设 16~32
)
    encode_ffhq_to_latents(
    image_dir="./fid_samples/ffhq256_fm_unet",
    latent_dir="./ffhq256_x",
    vae=vae,
    batch_size=16,   # 4090 可设 16~32
)
    dataset = FFHQLatentDataset("./ffhq256_latents")
    dataset_x = FFHQLatentDataset("./ffhq256_x")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    loader = torch.utils.data.DataLoader(
    dataset_x,
    batch_size=256,
    shuffle=False,
)   
    latents = []
    for batch in loader:
        if isinstance(batch, torch.Tensor):
            latents.append(batch)
        else:
            latents.append(torch.from_numpy(batch))

    dataset0 = torch.cat(latents, dim=0).to(device)
    print(dataset0.shape)

    datalooper = infiniteloop(dataloader)
    # ---------------------
    # DiT latent model
    # ---------------------
    net_model = UNetModelWrapper(
        dim=(4, 32, 32),
        num_res_blocks=2,
        num_channels=256,
        channel_mult=[1, 1, 2, 2, 4, 4],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.0,
    ).to(device)
    ema_model = copy.deepcopy(net_model)

    if FLAGS.parallel:
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))
    # ---------------------
    # 🔧 MOD: AdamW 8-bit
    # ---------------------
    optim = bnb.optim.AdamW8bit(
        net_model.parameters(),
        lr=FLAGS.lr
    )

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    # 🔧 MOD: AMP scaler
    scaler = torch.GradScaler()

    # ---------------------
    # Flow Matcher
    # ---------------------
    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError

    savedir = os.path.join(FLAGS.output_dir, FLAGS.model)
    os.makedirs(savedir, exist_ok=True)
    dtype = torch.float32
    x0_init = dataset0[:16].to(device, dtype=dtype)
    # ---------------------
    # Training loop
    # ---------------------
    with trange(FLAGS.total_steps) as pbar:
        for step in pbar:
            batch = next(iter(dataloader))
            x = batch.to(device, dtype=dtype, non_blocking=True)
            x1 = x
            x0 = dataset0[torch.randint(0, dataset0.shape[0], (x1.shape[0],), device=device)]
            #mask = torch.rand(x0.shape[0], device=device) < 0.5
            #x0_tile = x0 + (mask.float().view(-1, 1, 1, 1) * torch.randn_like(x0) * 0.5)
            x0_tile = random_gaussian_blur_tensor(x0, p=0.5, kernel_size=5, sigma=(0.01, 0.02))

            # 🔧 MOD: AMP forward
            with torch.autocast("cuda", dtype=torch.bfloat16):
                '''
                t = torch.rand(x.size(0), device=device).view(-1, 1, 1, 1)
                z_1 = torch.randn_like(x)
                z_t = (1 - t) * x + (1e-5 + (1 - 1e-5) * t) * z_1
                u = (1 - 1e-5) * z_1 - x
                v = net_model(t.squeeze(), z_t)
                loss = F.mse_loss(v, u)
                '''
                t, xt, ut = FM.sample_location_and_conditional_flow(x0_tile, x1)
                vt = net_model(t.squeeze(), xt)
                loss = torch.mean((vt - ut) ** 2)
                
                #loss = F.mse_loss(vt, ut)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip
            )
            scaler.unscale_(optim)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            sched.step()

            ema(net_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            if step % FLAGS.save_step == 0:
                generate_samples_vae_x0(
                    net_model, vae, FLAGS.parallel, savedir, step, x0_init, "normal"
                )
                generate_samples_vae_x0(
                    ema_model, vae, FLAGS.parallel, savedir, step, x0_init, "ema"
                )

                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    f"{savedir}/ckpt_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
