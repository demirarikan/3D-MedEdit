from core.Trainer import Trainer
import tqdm
import wandb
import torch
import os
import logging
import numpy as np
from model_zoo.inferers.inferer import LatentDiffusionInferer
from model_zoo.schedulers.ddpm import DDPMScheduler
from model_zoo.vqvae import VQVAE


class DDPMTrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super().__init__(training_params, model, data, device, log_wandb)

        self.autoencoder = self.setup_autoencoder(training_params["autoencoder"], device)
        self.ddpm = model
        self.scheduler = DDPMScheduler(num_train_timesteps=training_params["num_train_timesteps"],
                                       schedule=training_params["schedule"],
                                    #    **{"beta_start": training_params["beta_start"],
                                    #       "beta_end": training_params["beta_end"]}
                                        )
        self.inferer = LatentDiffusionInferer(scheduler=self.scheduler,
                                              scale_factor=self.calculate_scale_factor())
        
        self.val_interval = training_params["val_interval"]
        self.save_interval = training_params["save_interval"]

        self.test_res_dir = training_params["test_res_dir"]
        try:
            self.test_ds = data.test_dataloader()
        except:
            logging.info("[DDPMTrainer::init]: Error loading test dataset")
            self.test_ds = None

    def calculate_scale_factor(self):
        first_batch = next(iter(self.train_ds))[0].to(self.device)
        z = self.autoencoder.encode_stage_2_inputs(first_batch)
        self.example_latent = z
        scale_factor = 1/torch.std(z)
        return scale_factor
    
    def setup_autoencoder(self, autoencoder_params, device):
        checkpoint_path = autoencoder_params.pop("checkpoint_path")
        autoencoder = VQVAE(**autoencoder_params).to(device)
        autoencoder.load_state_dict(torch.load(checkpoint_path)['model_weights'])
        return autoencoder

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        if model_state is not None:
            self.ddpm.load_state_dict(model_state)
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

        self.early_stop = False

        # VQVAE should be pretrained
        self.autoencoder.eval()
        
        epoch_losses = []

        for epoch in range(self.training_params["nr_epochs"]):
            self.ddpm.train()
            epoch_loss = 0

            if start_epoch > epoch:
                continue
            if self.early_stop:
                logging.info(
                    "[Trainer::train]: ################ Finished training (early stopping) ################"
                )
                break
                
            progress_bar = tqdm.tqdm(enumerate(self.train_ds), total=len(self.train_ds))
            progress_bar.set_description(f"Epoch {epoch}")

            for batch_idx, (volume, _) in progress_bar:
                volume = volume.to(self.device)
                self.optimizer.zero_grad()

                noise = torch.randn_like(self.example_latent).to(self.device)

                #timesteps
                timesteps = torch.randint(
                    low=0, high=self.inferer.scheduler.num_train_timesteps, size=(volume.shape[0],), device=self.device
                )

                noise_pred = self.inferer(
                    inputs=volume, autoencoder_model=self.autoencoder, diffusion_model=self.ddpm, noise=noise, timesteps=timesteps
                )

                loss = self.criterion_rec(noise_pred, noise)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=epoch_loss / (batch_idx + 1))
            epoch_losses.append(epoch_loss / (batch_idx + 1))

            wandb.log({
                "epoch": epoch,
                "loss": epoch_losses[-1]
            })

            torch.save(
                {
                    "model_weights": self.model.state_dict(),
                    "optimizer_weights": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(self.client_path, "latest.pt"),
            )

            if epoch % self.training_params["save_interval"] == 0 and epoch != 0:
                logging.info(
                    f"[Trainer::train]: Saving model at epoch {epoch}"
                )
                torch.save(
                    {
                        "model_weights": self.model.state_dict(),
                        "optimizer_weights": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.client_path, f"epoch_{epoch}.pt"),
                )

            if epoch % self.val_interval == 0 and epoch != 0:
                self.test(
                    model_weights=self.model.state_dict(),
                    test_data=self.val_ds,
                    task="Val",
                    optimizer_weights=self.optimizer.state_dict(),
                    epoch=epoch,
                )
        
        if self.test_ds:
            self.test(
                model_weights=self.best_weights,
                test_data=self.test_ds,
                task="Test",
                optimizer_weights=self.best_opt_weights,
                epoch=-1,
            )

        return self.best_weights, self.best_opt_weights
    

    def test(self, model_weights, test_data, task="Val", optimizer_weights=None, epoch=0):
        self.ddpm.load_state_dict(model_weights)
        self.optimizer.load_state_dict(optimizer_weights)
        self.ddpm.eval()

        if task == "Val":
            logging.info("[Trainer::test]: Started validation")

            val_loss = 0
            with torch.no_grad():
                for batch_idx, (volume, _) in enumerate(test_data):
                    volume = volume.to(self.device)

                    noise = torch.randn_like(self.example_latent).to(self.device)

                    #timesteps
                    timesteps = torch.randint(
                        low=0, high=self.inferer.scheduler.num_train_timesteps, size=(volume.shape[0],), device=self.device
                    )

                    noise_pred = self.inferer(
                        inputs=volume, autoencoder_model=self.autoencoder, diffusion_model=self.ddpm, noise=noise, timesteps=timesteps
                    )

                    loss = self.criterion_rec(noise_pred, noise)
                    val_loss += loss.item()

                    if batch_idx == 0:
                        recons = self.autoencoder.decode_stage_2_inputs(noise_pred)
                        recons = recons.detach().cpu().numpy()
                        volume = volume.detach().cpu().numpy()
                        wandb.log({
                            "recon_loss": loss.item(),
                            "reconstruction1": wandb.Image(recons[0][0][80, :, :]*255),
                            "reconstruction2": wandb.Image(recons[0][0][:, 80, :]*255),
                            "reconstruction3": wandb.Image(recons[0][0][:, :, 80]*255),
                            "original1": wandb.Image(volume[0][0][80, :, :]*255),
                            "original2": wandb.Image(volume[0][0][:, 80, :]*255),
                            "original3": wandb.Image(volume[0][0][:, :, 80]*255),
                        })
                    
            val_loss /= len(test_data)

            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                torch.save(
                    {
                        "model_weights": model_weights,
                        "optimizer_weights": optimizer_weights,
                        "epoch": epoch,
                    },
                    os.path.join(self.client_path, f"best_epoch_{epoch}.pt"),
                )
                self.best_weights = model_weights
                self.best_opt_weights = optimizer_weights
            
            logging.info(f"[Trainer::test]: Validation loss: {val_loss}")
            return val_loss

