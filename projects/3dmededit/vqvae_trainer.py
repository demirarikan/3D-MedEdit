from core.Trainer import Trainer
import tqdm
import wandb
import torch
import os
import logging


class VQVAETrainer(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super().__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params["val_interval"]
        self.save_interval = training_params["save_interval"]

    def train(self, model_state=None, opt_state=None, start_epoch=0):
        if model_state is not None:
            self.model.load_state_dict(model_state)  # load weights
        if opt_state is not None:
            self.optimizer.load_state_dict(opt_state)  # load optimizer
        
        epoch_recon_losses = []
        epoch_quant_losses = []
        
        self.early_stop = False

        for epoch in range(self.training_params["nr_epochs"]):
            self.model.train()
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
                reconstruction, quant_loss = self.model(volume)
                # print("recon shape!!:", reconstruction.shape)
                # print("volume shape!!:", volume.shape)
                recon_loss = self.criterion_rec(volume, reconstruction) # l1 loss
                loss = recon_loss + quant_loss
                loss.backward()
                self.optimizer.step()

                epoch_loss += recon_loss.item()

                progress_bar.set_postfix(
                    recon_loss=epoch_loss / (batch_idx + 1), quant_loss=quant_loss.item() / (batch_idx + 1)
                )

            epoch_recon_losses.append(epoch_loss / (batch_idx + 1))
            epoch_quant_losses.append(quant_loss.item() / (batch_idx + 1))

            wandb.log({
                "epoch": epoch,
                "recon_loss": epoch_recon_losses[-1],
                "quant_loss": epoch_quant_losses[-1]
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
            
            if epoch % self.val_interval == 0 :
                self.test(
                    model_weights=self.model.state_dict(),
                    test_data=self.val_ds,
                    task="val",
                    optimizer_weights=self.optimizer.state_dict(),
                    epoch=epoch,
                )
        return self.best_weights, self.best_opt_weights
    
    def test(self, model_weights, test_data, task="Val", optimizer_weights=None, epoch=0):
        self.model.load_state_dict(model_weights)
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (volume, _) in enumerate(test_data):
                volume = volume.to(self.device)
                reconstruction, quant_loss = self.model(volume)

                if batch_idx == 1:
                    wandb.log({
                        "reconstruction": wandb.Image(reconstruction[0, 0, 100].cpu().detach().numpy() * 255),
                        "original": wandb.Image(volume[0, 0, 100].cpu().detach().numpy() * 255),
                    })


                recon_loss = self.criterion_rec(volume, reconstruction)
                val_loss += recon_loss.item()
        val_loss /= len(test_data)
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss

            torch.save(
                {
                    "model_weights": model_weights,
                    "optimizer_weights": optimizer_weights,
                    "epoch": epoch,
                },
                os.path.join(self.client_path, "best.pt"),
            )

            self.best_weights = model_weights
            self.best_opt_weights = optimizer_weights
        wandb.log({f"{task}_loss": val_loss, "epoch": epoch})
        logging.info(
            f"[Trainer::test]: {task} loss: {val_loss}"
        )
        return val_loss