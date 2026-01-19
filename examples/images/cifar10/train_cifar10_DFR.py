# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os

import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from tqdm import trange
from utils_cifar import ema, generate_samples, infiniteloop,generate_samples_ext

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
import numpy as np
import bitsandbytes as bnb
FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/x", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 128, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

def sample_z_per_label_random(x, y, samples, labels):
    """
    For each y[i], randomly select one sample from `samples`
    that has the same label.
    
    The returned tensor z has shape:
        [batch_size, *sample_shape]
    """
    device = samples.device
    x = x.to(device)
    y = y.to(device)
    labels = labels.to(device)

    batch_size = x.shape[0]
    z = torch.empty((batch_size, *samples.shape[1:]),
                    dtype=samples.dtype, device=device)

    # Collect indices of samples corresponding to each label
    unique_labels = y.unique()
    label_to_idx = {
        lbl.item(): torch.nonzero(labels == lbl, as_tuple=True)[0]
        for lbl in unique_labels
    }

    for lbl in unique_labels:
        # Indices of batch elements with the current label
        batch_idxs = torch.nonzero(y == lbl, as_tuple=True)[0]

        # Candidate sample indices with the same label
        candidate_idxs = label_to_idx[lbl.item()]

        # Randomly select one candidate sample for each batch index
        random_idx = torch.randint(
            0, len(candidate_idxs),
            (len(batch_idxs),),
            device=device
        )
        selected_idxs = candidate_idxs[random_idx]

        z[batch_idxs] = samples[selected_idxs]

    return z

import random
import torchvision.transforms.functional as TF
def random_gaussian_blur_tensor(x, p=0.5, kernel_size=5, sigma=(0.1, 2.0)):
    if random.random() < p:
        s = random.uniform(*sigma)
        return TF.gaussian_blur(x, kernel_size, sigma=s)
    return x

import torch
import numpy as np

def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    # DATASETS/DATALOADER
    data_9 = np.load("generated_images_y.npz")
    labels0 = data_9['labels']
    dataset0 = data_9['images']
    labels0 = torch.tensor(labels0, dtype=torch.int64, device=device)
    train_data2 = np.load("cifar10_train.npz")["data"]
    labels2 = np.load("cifar10_train.npz")["labels"]
    labels2 = torch.tensor(labels2, dtype=torch.int64, device=device)
    dataset0 = torch.from_numpy(dataset0).to(device)
    train_data2 = torch.from_numpy(train_data2).to(device)
    #print("Dataset loaded, shape:", dataset0.shape, "labels shape:", labels0.shape)
    train_set = TensorDataset(train_data2, labels2)
    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
    )

    #datalooper = infiniteloop(dataloader)

    # MODELS
    net_model = UNetModelWrapper(
        dim=(3, 32, 32),
        num_res_blocks=2,
        num_channels=FLAGS.num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.0,
        class_cond=True,
        num_classes=10,
    ).to(
        device
    )  # new dropout + bs of 128

    ema_model = copy.deepcopy(net_model)
    optim = bnb.optim.AdamW8bit(
        net_model.parameters(),
        lr=FLAGS.lr,
        betas=(0.9, 0.99),
        weight_decay=1e-4,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    #################################
    #            OT-CFM
    #################################

    sigma = 0.1
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "extfm":
        from torchcfm.conditional_flow_matching import EXTConditionalFlowMatcher

        FM = EXTConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + "/"
    os.makedirs(savedir, exist_ok=True)
    scaler = torch.GradScaler()
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            #x1 = next(datalooper).to(device)
            #batch = next(dataloader)
            batch = next(iter(dataloader))
            x1 = batch[0].to(device)
            y = batch[1].to(device)
            #x0_tile = x0 + torch.randn_like(x0) * 0.1  # add noise to x0
            #print("x0 shape:", x0.shape, "labels shape:", labels.shape)
            #print("trandata2 shape:", train_data2.shape, "labels2 shape:", labels2.shape)
            #x0 = torch.randn_like(x1)
            #x1 = sample_z_per_label_nearest(x0, y, train_data2, labels2)
            x0 = sample_z_per_label_random(x1, y, dataset0, labels0)
            #x0_tile = random_gaussian_blur_tensor(x0, p=0.3, kernel_size=5, sigma=(0.01, 0.02))  # add noise to x0
            # x0 
            mask = torch.rand(x0.shape[0], device=device) < 0.3
            x0_tile = x0 + (mask.float().view(-1, 1, 1, 1) * torch.randn_like(x0) * 0.01)
            #t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t, xt, ut, _, y1 = FM.guided_sample_location_and_conditional_flow(x0_tile, x1, y1=y)
                #t = sample_t(FLAGS.batch_size, weighting="lognormal", path_type="linear", device=device, dtype=x0.dtype)
                vt = net_model(t, xt, y1)
                loss = torch.mean((vt - ut) ** 2)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            scaler.step(optim)
            scaler.update()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                generate_samples_ext(x0[:100],y1[:100],net_model, FLAGS.parallel, savedir, step, net_="normal", guidance=True)
                generate_samples_ext(x0[:100],y1[:100],ema_model, FLAGS.parallel, savedir, step, net_="ema", guidance=True)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_cifar10_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)
