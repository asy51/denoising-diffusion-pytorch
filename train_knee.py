import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import nibabel

nibabel.imageglobals.logger.setLevel(40)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels=1,
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

trainer = Trainer(
    diffusion,
    '',
    train_batch_size = 12,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                      # turn on mixed precision
    convert_image_to='L',
    results_folder='./results_comet'
)

trainer.train()