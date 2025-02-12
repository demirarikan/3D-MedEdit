import math
import logging

import torch
import wandb
import lpips 
import tqdm
import matplotlib.pyplot as plt 
from skimage.metrics import structural_similarity as ssim
import json

from core.DownstreamEvaluator import DownstreamEvaluator
from core.DataLoader import DefaultDataLoader
from model_zoo.vqvae import VQVAE
from model_zoo.inferers.inferer import LatentDiffusionInferer
from model_zoo.schedulers.ddpm import DDPMScheduler
from core.DataLoader import DefaultDataLoader

class DDPMEvaluator(DownstreamEvaluator):
    def __init__(self, task_name, model, device, data: dict[str, DefaultDataLoader], checkpoint_path, **kwargs):
        super().__init__(model, device, checkpoint_path)
        self.data = data
        self.autoencoder = self.setup_autoencoder(kwargs.get("autoencoder_params", None), device)

        scheduler_params = kwargs.get("scheduler_params", None)
        self.scheduler = DDPMScheduler(num_train_timesteps=scheduler_params["num_train_timesteps"],
                                       schedule=scheduler_params["schedule"],
                                       **{"beta_start": scheduler_params["beta_start"],
                                          "beta_end": scheduler_params["beta_end"]}
                                        )
        self.inferer = LatentDiffusionInferer(scheduler=self.scheduler,
                                              scale_factor=self.calculate_scale_factor())
        

    def setup_autoencoder(self, autoencoder_params, device):
        checkpoint_path = autoencoder_params.pop("checkpoint_path")
        autoencoder = VQVAE(**autoencoder_params).to(device)
        autoencoder.load_state_dict(torch.load(checkpoint_path)['model_weights'])
        return autoencoder        
    
    def calculate_scale_factor(self):
        first_ds = next(iter(self.data.values()))
        first_batch = next(iter(first_ds))[0].to(self.device)
        z = self.autoencoder.encode_stage_2_inputs(first_batch)
        self.example_latent = z
        scale_factor = 1/torch.std(z)
        return scale_factor

    def start_task(self, global_model):
        model_weights = None
        if "model_weights" in global_model.keys():
            model_weights = global_model["model_weights"]
            logging.info("[Configurator::train::INFO]: Model weights loaded!")
        
        self.model.load_state_dict(model_weights)
        self.model.eval()

        for dataloader_name, dataloader in self.data.items():
            curr_dataloader = dataloader

            progress_bar = tqdm.tqdm(
                enumerate(curr_dataloader), total=len(curr_dataloader)
            )
            progress_bar.set_description(f"DDPM Evaluation {dataloader_name}")

            l1_losses = {}

            with torch.no_grad():
                for batch_idx, (volume, _) in progress_bar:
                    volume = volume.to(self.device)

                    batch_size = volume.shape[0]

                    noise = torch.randn_like(self.example_latent).to(self.device)

                    min_timestep = 30
                    max_timestep = 500
                    step_size = 50 
                    num_steps = math.ceil((max_timestep - min_timestep) / step_size)

                    num_rows = 3
                    fig, axs = plt.subplots(num_rows, num_steps+1, figsize=(num_steps*2, batch_size))
                    fig.suptitle(f"DDPM Evaluation {dataloader_name} - Batch: {batch_idx}")

                    for i in range(num_rows):
                        try:
                            axs[i, 0].imshow(volume[i][0][80, :, :].cpu().numpy(), cmap="gray")
                        except IndexError:
                            continue
                        axs[i, 0].axis("off")
                        axis_title = f"GT"
                        axs[i, 0].set_title(axis_title)




                    clean_latents = self.autoencoder.encode_stage_2_inputs(volume)

                    # noisy_latents = torch.randn_like(clean_latents).to(self.device)

                    # reconstructed_volumes = self.inferer.sample(
                    #     input_noise=noisy_latents,
                    #     autoencoder_model=self.autoencoder,
                    #     diffusion_model=self.model,
                    #     scheduler=self.scheduler,
                    # )

                    # fig, axs = plt.subplots(num_rows, 2, figsize=(num_steps*2, batch_size))
                    # fig.suptitle(f"DDPM Evaluation {dataloader_name} - Batch: {batch_idx}")
                    # for i in range(num_rows):
                    #     axs[i, 0].imshow(volume[i][0][80, :, :].cpu().numpy(), cmap="gray")
                    #     axs[i, 0].axis("off")
                    #     axs[i, 0].set_title("GT")
                    #     axs[i, 1].imshow(reconstructed_volumes[i][0][80, :, :].cpu().numpy(), cmap="gray")
                    #     axs[i, 1].axis("off")
                    #     axs[i, 1].set_title("Reconstructed")
                    
                    # fig.tight_layout()
                    # fig.savefig(f"projects/3dmededit/results/1000steps_{dataloader_name}_Batch_{batch_idx}.png")




                    for idx, curr_timestep in enumerate(range(min_timestep, max_timestep, step_size)):
                        # self.scheduler.num_train_timesteps = curr_timestep
                        self.scheduler.set_timesteps(curr_timestep)

                        timesteps = torch.full((batch_size,), curr_timestep).to(self.device)

                        noisy_latents = self.scheduler.add_noise(clean_latents, noise, timesteps)

                        reconstructed_volumes = self.inferer.sample(
                            input_noise=noisy_latents,
                            autoencoder_model=self.autoencoder,
                            diffusion_model=self.model,
                            scheduler=self.scheduler,
                        )
                        
                        l1_loss_batch = torch.nn.functional.l1_loss(reconstructed_volumes, volume)
                        if curr_timestep not in l1_losses.keys():
                            l1_losses[curr_timestep] = l1_loss_batch
                        else:
                            l1_losses[curr_timestep] += l1_loss_batch

                        for i in range(num_rows):
                            try:
                                axs[i, idx+1].imshow(reconstructed_volumes[i][0][80, :, :].cpu().numpy(), cmap="gray")
                            except IndexError:
                                continue
                            axs[i, idx+1].axis("off")
                            axis_title = f"timestep: {curr_timestep}"
                            axs[i, idx+1].set_title(axis_title)
                        
                    fig.tight_layout()
                    fig.savefig(f"projects/3dmededit/results/DDPM_Evaluation_{dataloader_name}_Batch_{batch_idx}.png")

                l1_losses = {k: (v / len(curr_dataloader)).tolist() for k, v in l1_losses.items()}
                with open(f"projects/3dmededit/results/DDPM_Evaluation_{dataloader_name}.txt", "w") as f:
                    f.write(json.dumps(l1_losses))
                wandb.log(l1_losses)            
                
                