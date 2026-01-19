# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import os
import sys
import numpy as np
import torch
from absl import app, flags
from torchdiffeq import odeint
from torchdyn.core import NeuralODE
from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

# UNet model parameters
flags.DEFINE_integer("num_channel", 128, help="Base channel size of UNet")

# Sampling parameters
flags.DEFINE_string("input_dir", "./results", help="Directory containing the model checkpoint")
flags.DEFINE_string("model", "otcfm", help="Flow matching model type")
flags.DEFINE_integer("integration_steps", 100, help="Number of inference steps")
flags.DEFINE_string("integration_method", "dopri5", help="ODE integration method (euler, rk4, dopri5, etc.)")
flags.DEFINE_integer("step", 400000, help="Training step of the checkpoint to load")
flags.DEFINE_integer("num_gen", 50000, help="Total number of images to generate")
flags.DEFINE_float("tol", 1e-5, help="ODE solver tolerance (absolute and relative)")
flags.DEFINE_integer("batch_size", 1024, help="Batch size for sampling")
flags.DEFINE_string("save_path", "./generated_images_y_xT.npz", help="Path to save the generated npz file")

FLAGS(sys.argv)
# base FID:4.75
# Set device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the UNet model
net = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=FLAGS.num_channel,
    channel_mult=[1, 2, 2, 2],
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
    class_cond=True,
    num_classes=10,
).to(device)

# Load checkpoint
PATH = f"{FLAGS.input_dir}/otcfm/otcfm_cifar10_weights_step_400000.pt"
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]

# Handle potential key mismatch in checkpoint
try:
    net.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v  # remove 'module.' prefix if present
    net.load_state_dict(new_state_dict)

net.eval()  # Set model to evaluation mode


data_9 = np.load("cifar10_train.npz")
labels0 = data_9['labels']
dataset0 = data_9["data"]
x0 = torch.from_numpy(dataset0).to(device)
label0 = torch.from_numpy(labels0).to(device)


x0_index = 0  

def gen_1_img(unused_latent):
    global x0_index
    
    with torch.no_grad():
        # 从 x0 里依次取 batch
        start = x0_index
        end = start + FLAGS.batch_size
        x = x0[start:end].to(device)
        y = label0[start:end].to(device)

        # 更新索引，如果超过长度则循环回到 0
        x0_index = end % len(x0)
        model_ = lambda t, x, args=None: net(t, x, y)
        if FLAGS.integration_method == "euler":
            node = NeuralODE(model_, solver=FLAGS.integration_method)
        if FLAGS.integration_method == "euler":
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(1, 0, FLAGS.integration_steps + 1, device=device)
            traj = node.trajectory(x, t_span=t_span)
        else:
            print("Use method: ", FLAGS.integration_method)
            t_span = torch.linspace(1, 0, 2, device=device)
            traj = odeint(
                model_, x, t_span, rtol=FLAGS.tol, atol=FLAGS.tol, method=FLAGS.integration_method
            )

    x_final = traj[-1, :]  # .view([-1, 3, 32, 32]).clip(-1, 1)
    img_batch = x_final.cpu().numpy()
    return img_batch, y.cpu().numpy()


def main(_):
    # Compute number of batches
    num_batches = (FLAGS.num_gen + FLAGS.batch_size - 1) // FLAGS.batch_size
    all_imgs = []
    labels = []

    for i in range(num_batches):
        current_batch_size = min(FLAGS.batch_size, FLAGS.num_gen - i * FLAGS.batch_size)
        imgs,y = gen_1_img(current_batch_size)
        all_imgs.append(imgs)
        labels.append(y)
        print(f"Generated batch {i+1}/{num_batches}")

    # Concatenate all batches and save as npz
    all_imgs = np.concatenate(all_imgs, axis=0)
    labels = np.concatenate(labels, axis=0)
    print(f"Saving {all_imgs.shape[0]} images to {FLAGS.save_path}")
    np.savez_compressed(FLAGS.save_path, images=all_imgs, labels=labels)
    print("Done!")


if __name__ == "__main__":
    app.run(main)
