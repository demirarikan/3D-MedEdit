from core.DownstreamEvaluator import DownstreamEvaluator
import os
import logging
import tqdm
import lpips
import torch
from skimage.metrics import structural_similarity as ssim
from core.DataLoader import DefaultDataLoader
import wandb
import math

class AutoencodeEvaluator(DownstreamEvaluator):
    def __init__(self, dst_name, model, device, data: dict[str, DefaultDataLoader], checkpoint_path):
        super().__init__(model, device, checkpoint_path)
        self.data = data
        self.sagittal_lpips, self.coronal_lpips, self.axial_lpips = [], [], []
        self.ssim = []
        self.l1_loss = 0
        self.mse_loss = 0
        self.lpips_model = lpips.LPIPS(net='alex')

        self.recon_intensities = {
            'max': -math.inf,
            'min': math.inf,
            'mean': 0
        }
        self.volume_intensities = {
            'max': -math.inf,
            'min': math.inf,
            'mean': 0
        }

    def start_task(self, global_model):
        model_weights = None
        if "model_weights" in global_model.keys():
            model_weights = global_model["model_weights"]
            logging.info("[Configurator::train::INFO]: Model weights loaded!")

        self.model.load_state_dict(model_weights)
        self.model.eval()

        for dataloader_name, dataloader in self.data.items():
            curr_dataloader = dataloader

            progress_bar = tqdm.tqdm(enumerate(curr_dataloader), total=len(curr_dataloader))
            progress_bar.set_description(f"Autoencoder Evaluation")

            for batch_idx, (volume, _) in progress_bar:
                volume = volume.to(self.device)
                reconstruction, quant_loss = self.model(volume)

                # log images
                wandb.log({
                    "reconstruction1": wandb.Image(reconstruction[0][0][80, :, :].cpu().detach().numpy() * 255),
                    "reconstruction2": wandb.Image(reconstruction[0][0][:, 80, :].cpu().detach().numpy() * 255),
                    "reconstruction3": wandb.Image(reconstruction[0][0][:, :, 80].cpu().detach().numpy() * 255),
                    "original1": wandb.Image(volume[0][0][80, :, :].cpu().detach().numpy() * 255),
                    "original2": wandb.Image(volume[0][0][:, 80, :].cpu().detach().numpy() * 255),
                    "original3": wandb.Image(volume[0][0][:, :, 80].cpu().detach().numpy() * 255),
                })

                self.update_intensities(volume.detach().cpu(), reconstruction.detach().cpu())

                #l1 loss 
                self.l1_loss += torch.nn.functional.l1_loss(reconstruction, volume, reduction='mean').item()
                #mse loss
                self.mse_loss += torch.nn.functional.mse_loss(reconstruction, volume, reduction='mean').item()
                #ssim
                self.calculate_ssim(volume, reconstruction)
                #lpips
                self.calculate_lpips(volume, reconstruction)
            
            self.l1_loss /= len(curr_dataloader)
            self.mse_loss /= len(curr_dataloader)
            self.ssim = sum(self.ssim) / len(self.ssim)
            self.sagittal_lpips = torch.cat(self.sagittal_lpips)
            self.coronal_lpips = torch.cat(self.coronal_lpips)
            self.axial_lpips = torch.cat(self.axial_lpips)

            self.recon_intensities['mean'] /= len(curr_dataloader)
            self.volume_intensities['mean'] /= len(curr_dataloader)

            wandb.log({
                f'{dataloader_name}_l1_loss': self.l1_loss,
                f'{dataloader_name}_mse_loss': self.mse_loss,
                f'{dataloader_name}_ssim': self.ssim,
                f'{dataloader_name}_sagittal_lpips_dim1': self.sagittal_lpips.mean(),
                f'{dataloader_name}_coronal_lpips_dim2': self.coronal_lpips.mean(),
                f'{dataloader_name}_axial_lpips_dim3': self.axial_lpips.mean(),
                f'{dataloader_name}_recon_intensities': self.recon_intensities,
                f'{dataloader_name}_volume_intensities': self.volume_intensities
            })
        

    def calculate_lpips(self, volume, reconstruction):
        self.lpips_model.to(self.device)
        # define lpips loss
        batch_size, channels, num_sagittal_slices, num_coronal_slices, num_axial_slices = volume.shape
        # lipis expects RGB images
        volume = volume.repeat(1, 3, 1, 1, 1)
        reconstruction = reconstruction.repeat(1, 3, 1, 1, 1)

        for i in range(batch_size): # batch
            volume_i = volume[i]
            recon_i = reconstruction[i]
            # reorder to have each dimension be the first once
            # shape: (channels, num_sagittal_slices, num_coronal_slices, num_axial_slices)
            # sagittal first
            sag_vol = volume_i.permute(1, 0, 2, 3)
            sag_recon = recon_i.permute(1, 0, 2, 3)
            lpips_loss = self.lpips_model.forward(sag_vol, sag_recon)
            self.sagittal_lpips.append(lpips_loss.detach().cpu())
            del sag_vol, sag_recon

            #coronal first
            cor_vol = volume_i.permute(2, 0, 1, 3)
            cor_recon = recon_i.permute(2, 0, 1, 3)
            lpips_loss = self.lpips_model.forward(cor_vol, cor_recon)
            self.coronal_lpips.append(lpips_loss.detach().cpu())
            del cor_vol, cor_recon

            #axial first
            ax_vol = volume_i.permute(3, 0, 1, 2)
            ax_recon = recon_i.permute(3, 0, 1, 2)
            lpips_loss = self.lpips_model.forward(ax_vol, ax_recon)
            self.axial_lpips.append(lpips_loss.detach().cpu())
            del ax_vol, ax_recon
            torch.cuda.empty_cache()
        self.lpips_model.to('cpu')

    def calculate_ssim(self, volume, reconstruction):
        batch_size, _, _, _, _ = volume.shape
        for i in range(batch_size):
            volume_i = volume[i].squeeze().cpu().detach().numpy()
            reconstruction_i = reconstruction[i].squeeze().cpu().detach().numpy()
            ssim_score = ssim(volume_i, reconstruction_i, data_range=reconstruction_i.max() - reconstruction_i.min())
            self.ssim.append(ssim_score)

    def update_intensities(self, volume, recon):
        if recon.max() > self.recon_intensities['max']:
            self.recon_intensities['max'] = recon.max()
        
        if recon.min() < self.recon_intensities['min']:
            self.recon_intensities['min'] = recon.min()

        self.recon_intensities['mean'] += recon.mean()
        
        if volume.max() > self.volume_intensities['max']:
            self.volume_intensities['max'] = volume.max()

        if volume.min() < self.volume_intensities['min']:
            self.volume_intensities['min'] = volume.min()

        self.volume_intensities['mean'] += volume.mean()
