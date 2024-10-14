"""
DownstreamEvaluator.py

Run Downstream Tasks after training has finished
"""

import logging
import torch
import wandb


from optim.losses.ln_losses import *

from dl_utils.radnet_utils import compute_fid

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor, join
from optim.metrics import dice_coefficient_batch
import numpy as np
from core.DownstreamEvaluator import DownstreamEvaluator
import os
NNUNET_STUFF = {
    "sitk_stuff": {
        "spacing": (1.0, 1.0),
        "origin": (0.0, 0.0),
        "direction": (1.0, 0.0, 0.0, 1.0),
    },
    "spacing": [999.0, 1.0, 1.0],
}


class Editor(DownstreamEvaluator):
    def __init__(self, model, device, checkpoint_path, encoding_ratio, resample_steps):
        super(Editor, self).__init__(model, device, checkpoint_path)

        self.encoding_ratio = encoding_ratio
        self.resample_steps = resample_steps

        # fixed bootstrapping indices to compute FID
        self.bootstrap_indices = [
            [
                39,
                23,
                32,
                36,
                39,
                52,
                46,
                23,
                50,
                7,
                40,
                27,
                26,
                36,
                24,
                20,
                31,
                0,
                34,
                28,
                25,
                33,
                26,
                11,
                43,
                42,
                44,
                45,
                24,
                21,
                3,
                14,
                33,
                52,
                36,
                15,
                52,
                40,
                22,
                32,
                51,
                29,
                37,
                16,
                27,
                50,
                19,
                40,
                52,
                25,
                29,
                49,
                6,
            ],
            [
                38,
                5,
                5,
                5,
                24,
                7,
                32,
                14,
                44,
                41,
                27,
                41,
                52,
                26,
                25,
                50,
                52,
                1,
                10,
                18,
                26,
                50,
                52,
                14,
                30,
                41,
                20,
                36,
                28,
                6,
                8,
                44,
                27,
                24,
                3,
                3,
                22,
                39,
                11,
                14,
                18,
                52,
                16,
                36,
                45,
                45,
                50,
                38,
                10,
                47,
                41,
                6,
                13,
            ],
            [
                15,
                15,
                22,
                7,
                52,
                15,
                10,
                13,
                32,
                15,
                3,
                46,
                15,
                51,
                28,
                32,
                38,
                11,
                0,
                7,
                32,
                15,
                20,
                33,
                16,
                34,
                7,
                17,
                10,
                40,
                19,
                11,
                41,
                41,
                16,
                20,
                17,
                11,
                47,
                18,
                38,
                23,
                22,
                1,
                2,
                38,
                18,
                48,
                40,
                15,
                29,
                35,
                18,
            ],
            [
                48,
                24,
                52,
                11,
                36,
                27,
                48,
                34,
                52,
                19,
                21,
                28,
                40,
                12,
                47,
                27,
                23,
                30,
                49,
                40,
                32,
                32,
                18,
                15,
                0,
                49,
                36,
                14,
                49,
                19,
                2,
                26,
                41,
                6,
                1,
                26,
                4,
                37,
                39,
                26,
                15,
                36,
                6,
                49,
                19,
                0,
                35,
                36,
                12,
                33,
                29,
                2,
                44,
            ],
            [
                31,
                24,
                23,
                52,
                32,
                52,
                17,
                47,
                52,
                21,
                25,
                43,
                39,
                45,
                13,
                15,
                45,
                38,
                10,
                13,
                50,
                40,
                52,
                50,
                32,
                26,
                17,
                5,
                9,
                51,
                22,
                26,
                50,
                33,
                38,
                15,
                48,
                25,
                8,
                43,
                48,
                0,
                26,
                3,
                34,
                38,
                34,
                51,
                48,
                21,
                36,
                52,
                10,
            ],
            [
                14,
                43,
                43,
                45,
                25,
                35,
                17,
                26,
                13,
                42,
                48,
                29,
                37,
                12,
                24,
                52,
                47,
                50,
                47,
                44,
                14,
                29,
                2,
                36,
                33,
                4,
                46,
                3,
                48,
                26,
                40,
                3,
                22,
                26,
                40,
                6,
                7,
                17,
                0,
                35,
                52,
                22,
                20,
                10,
                47,
                1,
                44,
                11,
                30,
                11,
                19,
                12,
                29,
            ],
            [
                44,
                38,
                30,
                4,
                25,
                28,
                15,
                42,
                22,
                11,
                33,
                27,
                25,
                25,
                23,
                1,
                19,
                18,
                0,
                6,
                42,
                33,
                2,
                52,
                41,
                16,
                29,
                46,
                9,
                0,
                31,
                32,
                43,
                13,
                17,
                0,
                25,
                13,
                10,
                22,
                36,
                14,
                25,
                20,
                29,
                28,
                36,
                43,
                28,
                2,
                12,
                52,
                50,
            ],
            [
                0,
                21,
                35,
                14,
                7,
                10,
                52,
                13,
                19,
                40,
                35,
                44,
                24,
                32,
                21,
                20,
                30,
                23,
                46,
                21,
                30,
                19,
                16,
                12,
                10,
                7,
                4,
                8,
                30,
                37,
                5,
                51,
                20,
                51,
                8,
                11,
                23,
                40,
                14,
                20,
                0,
                24,
                27,
                7,
                24,
                28,
                52,
                30,
                18,
                16,
                2,
                24,
                10,
            ],
            [
                38,
                49,
                38,
                12,
                8,
                16,
                10,
                38,
                14,
                51,
                43,
                25,
                29,
                37,
                4,
                49,
                10,
                12,
                25,
                18,
                10,
                3,
                9,
                11,
                2,
                21,
                32,
                48,
                22,
                37,
                50,
                25,
                27,
                45,
                2,
                43,
                44,
                31,
                52,
                23,
                39,
                24,
                49,
                27,
                7,
                0,
                6,
                18,
                15,
                15,
                23,
                8,
                21,
            ],
            [
                27,
                46,
                28,
                13,
                20,
                41,
                17,
                6,
                48,
                8,
                32,
                13,
                0,
                46,
                0,
                35,
                50,
                23,
                42,
                0,
                0,
                28,
                36,
                12,
                51,
                34,
                52,
                43,
                36,
                25,
                40,
                11,
                48,
                14,
                47,
                37,
                29,
                12,
                22,
                45,
                29,
                20,
                47,
                46,
                30,
                34,
                8,
                45,
                4,
                49,
                7,
                36,
                33,
            ],
        ]
        self.bootstrap_indices = torch.tensor(self.bootstrap_indices)

        self.radnet = torch.hub.load(
            "Warvito/radimagenet-models:main",
            model="radimagenet_resnet50",
            verbose=True,
        )
        self.radnet.to(self.device)
        self.radnet.eval()

        self.nnunet = nnUNetPredictor(
            device=torch.device("cuda", 0),
        )
        
        nnUNet_model = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "nnunet_model")

        self.nnunet.initialize_from_trained_model_folder(
            join(nnUNet_model, "folds"),
            use_folds=(0, 1, 2, 3, 4),
            checkpoint_name="checkpoint_final.pth",
        )

    def start_task(self, global_model, test_data, reference_atlas_evaluation_split, task):
        """
        Function to perform analysis after training is complete, e.g., call downstream tasks routines, e.g.
        anomaly detection, classification, etc..

        :param global_model: dict
            dictionary with the model weights of the federated collaborators
        """

        model_weights = None
        if "model_weights" in global_model.keys():
            model_weights = global_model["model_weights"]
            logging.info("[Configurator::train::INFO]: Model weights loaded!")

        self.model.load_state_dict(model_weights)
        self.model.to(self.device)
        self.model.eval()

        test_total = 0

        first_batch = True

        with torch.no_grad():
            for data in test_data:
                priors = data[0].to(self.device)
                patho_masks = data[1].to(self.device)
                brain_masks = data[2].to(self.device)

                b, _, _, _ = priors.shape
                test_total += b

                if task == "sdedit":
                    counterfactuals = self.model.sdedit(
                        original_images=priors,
                        patho_masks=patho_masks,
                        brain_masks=brain_masks,
                        encoding_ratio=self.encoding_ratio,
                    )
                elif task == "mededit":
                    dilated_patho_masks = data[3].to(self.device)
                    inpaint_masks = dilated_patho_masks

                    counterfactuals = self.model.mededit(
                        original_images=priors,
                        inpaint_masks=inpaint_masks,
                        patho_masks=patho_masks,
                        brain_masks=brain_masks,
                        resample_steps=self.resample_steps,
                    )
                elif task == "naive_repaint":
                    inpaint_masks = patho_masks

                    counterfactuals = self.model.mededit(
                        original_images=priors,
                        inpaint_masks=inpaint_masks,
                        patho_masks=patho_masks,
                        brain_masks=brain_masks,
                        resample_steps=self.resample_steps,
                    )
                elif task == "palette":
                    palette_masks = data[4].to(self.device)

                    counterfactuals = self.model.palette(
                        original_images=priors, palette_masks=palette_masks
                    )

                counterfactuals_np = counterfactuals.detach().cpu().numpy()
                predicted_masks = self._predict_segmentation_mask(counterfactuals_np, b)

                if first_batch:
                    all_patho_masks = patho_masks.detach()
                    all_counterfactuals = counterfactuals.detach()
                    all_predicted_masks = predicted_masks
                    first_batch = False
                else:
                    all_patho_masks = torch.cat(
                        (all_patho_masks, patho_masks.detach()), dim=0
                    )
                    all_counterfactuals = torch.cat(
                        (all_counterfactuals, counterfactuals.detach()), dim=0
                    )
                    all_predicted_masks.extend(predicted_masks)

                for batch_idx in range(b):
                    grid_image = self._plot_triplet(
                        priors, patho_masks, counterfactuals, batch_idx
                    )
                    wandb.log({task + "/Example_": [wandb.Image(grid_image)]})

            first_batch = True
            with torch.no_grad():
                for data in reference_atlas_evaluation_split:
                    x = data[0].to(self.device)
                    if first_batch:
                        all_same_atlas = x.detach()
                        first_batch = False
                    else:
                        all_same_atlas = torch.cat((all_same_atlas, x.detach()), dim=0)

            self._compute_metrics(
                all_counterfactuals,
                all_patho_masks,
                all_same_atlas,
                all_predicted_masks,
            )

    def _predict_segmentation_mask(self, counterfactuals_np, batch_size):
        counterfactuals_np = [
            np.expand_dims(counterfactuals_np[i], axis=0) for i in range(batch_size)
        ]

        iterator = self.nnunet.get_data_iterator_from_raw_npy_data(
            counterfactuals_np, None, [NNUNET_STUFF for _ in range(batch_size)], None, 1
        )
        predicted_masks = self.nnunet.predict_from_data_iterator(iterator, False, 1)
        return predicted_masks

    def _plot_triplet(self, priors, patho_masks, counterfactuals, batch_idx):
        prior = priors[batch_idx].detach().cpu().numpy()
        patho_mask = patho_masks[batch_idx].detach().cpu().numpy()
        counterfactual = counterfactuals[batch_idx].detach().cpu().numpy()

        grid_image = np.hstack([prior, patho_mask, counterfactual])

        return grid_image

    def _compute_metrics(
        self,
        all_counterfactuals,
        all_patho_masks,
        all_same_atlas,
        all_predicted_masks,
    ):
        fid_mean, fid_std = compute_fid(
            self.radnet,
            all_counterfactuals,
            all_same_atlas,
            bootstrap=True,
            indices_=self.bootstrap_indices,
        )
        wandb.log({"FID mean": fid_mean})
        wandb.log({"FID std": fid_std})

        all_patho_masks = all_patho_masks.detach().cpu().numpy()
        all_predicted_masks = np.stack(all_predicted_masks)

        # Compute the Dice coefficient for each batch
        dice = dice_coefficient_batch(all_patho_masks, all_predicted_masks)
        wandb.log({"DICE Score": dice})
