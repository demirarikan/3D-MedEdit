# Based on Inferer module from MONAI:
# -----------------------------------------------------------------------------------------------
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math


import torch
import torch.nn as nn
from tqdm import tqdm

from net_utils.diffusion_unet import DiffusionModelUNet
from net_utils.schedulers.ddim import DDIMScheduler
from net_utils.schedulers.ddpm import DDPMScheduler
from net_utils.noise import generate_noise

has_tqdm = True


class DDPM(nn.Module):

    def __init__(
        self,
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        train_scheduler="ddpm",
        inference_scheduler="ddpm",
        inference_steps=1000,
        noise_level_recon=300,
        noise_type="gaussian",
        prediction_type="epsilon",
        resample_steps=4,
        encoding_ratio=0.5,
        image_path="",
    ):
        super().__init__()
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
        )
        self.noise_level_recon = noise_level_recon
        self.prediction_type = prediction_type
        self.resample_steps = resample_steps
        self.image_path = image_path

        self.encoding_ratio = encoding_ratio

        # set up scheduler and timesteps
        if train_scheduler == "ddpm":
            self.train_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )
        else:
            raise NotImplementedError(
                f"{train_scheduler} does is not implemented for {self.__class__}"
            )

        if inference_scheduler == "ddim":
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )
        else:
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                noise_type=noise_type,
                prediction_type=prediction_type,
            )

        self.inference_scheduler.set_timesteps(inference_steps)
        self.device = torch.device(
            "cuda:" + str(0) if torch.cuda.is_available() else "cpu"
        )

    def forward(
        self,
        inputs,
        patho_masks=None,
        brain_masks=None,
        noise=None,
        timesteps=None,
        condition=None,
    ):
        # only for torch_summary to work
        if noise is None:
            noise = torch.randn_like(inputs)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.train_scheduler.num_train_timesteps,
                (inputs.shape[0],),
                device=inputs.device,
            ).long()

        noisy_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=timesteps
        )

        unet_input = noisy_image

        if patho_masks is not None:
            unet_input = torch.cat((noisy_image, patho_masks), dim=1)

        if brain_masks is not None:
            unet_input = torch.cat((unet_input, brain_masks), dim=1)

        return self.unet(x=unet_input, timesteps=timesteps, context=condition)

    def palette(
        self,
        original_images: torch.tensor,
        palette_masks: torch.tensor,
        encoding_ratio=1.0,
        verbose=True,
    ):
        batch_size = original_images.shape[0]
        timesteps = self.inference_scheduler.get_timesteps(
            noise_level=int(encoding_ratio * self.noise_level_recon)
        )

        # image = torch.randn_like(original_image)

        # (generates) then adds noise to the original samples up to noise level 999.
        # the generated signal (image) is not exactly a random gaussian. it is almost
        # a random gaussian. because it still has some information from the original image if
        # alpha ( of the scheduler) is not 0 for the last step T (999 here).
        # see Abbeel lecture L6 on diff models time: 1h:07 mins for more info.
        # https://www.youtube.com/watch?v=DsEDMjdxOv4&t=5145s&ab_channel=PieterAbbeel

        noise = generate_noise(self.train_scheduler.noise_type, original_images)

        timesteps_noising = torch.full(
            [batch_size],
            int(encoding_ratio * self.noise_level_recon),
            device=original_images.device,
        ).long()

        image = self.inference_scheduler.add_noise(
            original_samples=original_images,
            noise=noise,
            timesteps=timesteps_noising,
        )

        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)

        for t in progress_bar:

            timesteps = torch.full([batch_size], t, device=self.device).long()
            # generate uknown part of diffusion reverse process
            predicted_noise = self.unet(
                torch.cat((image, palette_masks), dim=1),
                timesteps=torch.tensor((t,)).to(image.device),
                context=None,
            )

            # inference_scheduler.step() internallhy checks if t>0. if t>0, noise z=0 in the
            # sampling equation. take a look at # 6. Add noise, line 203 in the .step() function
            image, _ = self.inference_scheduler.step(predicted_noise, t, image)

        return image

    def sdedit(
        self,
        original_images: torch.tensor,
        patho_masks: torch.tensor,
        brain_masks: torch.tensor,
        encoding_ratio=0.2,
        verbose=True,
    ):
        batch_size = original_images.shape[0]
        timesteps = self.inference_scheduler.get_timesteps(
            noise_level=int(encoding_ratio * self.noise_level_recon)
        )

        # image = torch.randn_like(original_image)

        # (generates) then adds noise to the original samples up to noise level 999.
        # the generated signal (image) is not exactly a random gaussian. it is almost
        # a random gaussian. because it still has some information from the original image if
        # alpha ( of the scheduler) is not 0 for the last step T (999 here).
        # see Abbeel lecture L6 on diff models time: 1h:07 mins for more info.
        # https://www.youtube.com/watch?v=DsEDMjdxOv4&t=5145s&ab_channel=PieterAbbeel

        noise = generate_noise(self.train_scheduler.noise_type, original_images)

        timesteps_noising = torch.full(
            [batch_size],
            int(encoding_ratio * self.noise_level_recon),
            device=original_images.device,
        ).long()

        image = self.inference_scheduler.add_noise(
            original_samples=original_images,
            noise=noise,
            timesteps=timesteps_noising,
        )

        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)

        for t in progress_bar:

            timesteps = torch.full([batch_size], t, device=self.device).long()
            # generate uknown part of diffusion reverse process
            predicted_noise = self.unet(
                torch.cat((image, patho_masks, brain_masks), dim=1),
                timesteps=torch.tensor((t,)).to(image.device),
                context=None,
            )

            # inference_scheduler.step() internallhy checks if t>0. if t>0, noise z=0 in the
            # sampling equation. take a look at # 6. Add noise, line 203 in the .step() function
            image, _ = self.inference_scheduler.step(predicted_noise, t, image)

        return image

    def mededit(
        self,
        original_images: torch.tensor,
        inpaint_masks: torch.tensor,
        patho_masks: torch.tensor,
        brain_masks: torch.tensor,
        resample_steps=1,
        verbose=True,
    ):
        batch_size = original_images.shape[0]
        timesteps = self.inference_scheduler.get_timesteps(
            noise_level=self.noise_level_recon
        )

        # image = torch.randn_like(original_image)

        # (generates) then adds noise to the original samples up to noise level 999.
        # the generated signal (image) is not exactly a random gaussian. it is almost
        # a random gaussian. because it still has some information from the original image if
        # alpha ( of the scheduler) is not 0 for the last step T (999 here).
        # see Abbeel lecture L6 on diff models time: 1h:07 mins for more info.
        # https://www.youtube.com/watch?v=DsEDMjdxOv4&t=5145s&ab_channel=PieterAbbeel

        noise = generate_noise(self.train_scheduler.noise_type, original_images)

        timesteps_full_noise = torch.full(
            [batch_size], self.noise_level_recon, device=self.device
        ).long()

        image = self.inference_scheduler.add_noise(
            original_samples=original_images,
            noise=noise,
            timesteps=timesteps_full_noise,
        )

        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)

        for t in progress_bar:

            for u in range(resample_steps):
                # generate known part with forward process q()
                if t > 0:
                    noise = generate_noise(
                        self.inference_scheduler.noise_type, original_images
                    )
                else:
                    noise = torch.zeros_like(original_images)

                timesteps = torch.full([batch_size], t, device=self.device).long()
                x_known = self.inference_scheduler.add_noise(
                    original_samples=original_images, noise=noise, timesteps=timesteps
                )

                # generate uknown part of diffusion reverse process
                predicted_noise = self.unet(
                    torch.cat((image, patho_masks, brain_masks), dim=1),
                    timesteps=torch.tensor((t,)).to(image.device),
                    context=None,
                )

                # inference_scheduler.step() internallhy checks if t>0. if t>0, noise z=0 in the
                # sampling equation. take a look at # 6. Add noise, line 203 in the .step() function
                x_unknown, _ = self.inference_scheduler.step(predicted_noise, t, image)

                # join known and uknown parts
                image = inpaint_masks * x_unknown + (1 - inpaint_masks) * x_known

                if t > 0 and u < (resample_steps - 1):
                    # sample from q(xt/x_t-1): diffuse back to xt
                    image = torch.sqrt(
                        1 - self.inference_scheduler.betas[t - 1]
                    ) * image + torch.sqrt(
                        self.inference_scheduler.betas[t - 1]
                    ) * torch.randn_like(
                        original_images, device=self.device
                    )

        return image

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        noise_level: int | None,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            noise_level: noising step until which noise is added before sampling
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        image = input_noise
        timesteps = self.inference_scheduler.get_timesteps(noise_level)
        if verbose and has_tqdm:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            model_output = self.unet(
                image,
                timesteps=torch.Tensor((t,)).to(input_noise.device),
                context=conditioning,
            )

            # print("model_output", model_output.size())

            # 2. compute previous image: x_t -> x_t-1
            denoised_image, _ = self.inference_scheduler.step(
                model_output, t, image[:, 0, :, :].unsqueeze(1)
            )
            image[:, 0, :, :] = denoised_image[:, 0, :, :]

            # print("image size", image.size())
            # print("denoised image size", denoised_image.size())
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return denoised_image, intermediates
        else:
            return denoised_image

    @torch.no_grad()
    # function to noise and then sample from the noise given an image to get healthy reconstructions of anomalous input images
    def sample_from_image(
        self,
        inputs: torch.Tensor,
        patho_masks: torch.Tensor | None = None,
        brain_masks: torch.Tensor | None = None,
        noise_level: int | None = 500,
        save_intermediates: bool | None = False,
        intermediate_steps: int | None = 100,
        conditioning: torch.Tensor | None = None,
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Sample to specified noise level and use this as noisy input to sample back.
        Args:
            inputs: input images, NxCxHxW[xD]
            noise_level: noising step until which noise is added before
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """
        noise = generate_noise(self.train_scheduler.noise_type, inputs, noise_level)

        t = torch.full((inputs.shape[0],), noise_level, device=inputs.device).long()
        noised_image = self.train_scheduler.add_noise(
            original_samples=inputs, noise=noise, timesteps=t
        )

        unet_input = noised_image
        if patho_masks is not None:
            unet_input = torch.cat((noised_image, patho_masks), dim=1)
        if brain_masks is not None:
            unet_input = torch.cat((unet_input, brain_masks), dim=1)

        image = self.sample(
            input_noise=unet_input,
            noise_level=noise_level,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            verbose=verbose,
        )
        print("sample from image output size", image.size())
        return image, {"z": None}

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        save_intermediates: bool | None = False,
        conditioning: torch.Tensor | None = None,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
        verbose: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Computes the log-likelihoods for an input.
        Args:
            inputs: input images, NxCxHxW[xD]
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
        """

        if self.train_scheduler._get_name() != "DDPMScheduler":
            raise NotImplementedError(
                f"Likelihood computation is only compatible with DDPMScheduler,"
                f" you are using {self.train_scheduler._get_name()}"
            )
        if verbose and has_tqdm:
            progress_bar = tqdm(self.train_scheduler.timesteps)
        else:
            progress_bar = iter(self.train_scheduler.timesteps)
        intermediates = []
        total_kl = torch.zeros(inputs.shape[0]).to(inputs.device)
        for t in progress_bar:
            # Does this change things if we use different noise for every step?? before it was just one gaussian noise for all steps
            noise = generate_noise(self.train_scheduler.noise_type, inputs, t)

            timesteps = torch.full(inputs.shape[:1], t, device=inputs.device).long()
            noisy_image = self.train_scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps
            )
            model_output = self.unet(
                x=noisy_image, timesteps=timesteps, context=conditioning
            )
            # get the model's predicted mean, and variance if it is predicted
            if model_output.shape[1] == inputs.shape[
                1
            ] * 2 and self.train_scheduler.variance_type in [
                "learned",
                "learned_range",
            ]:
                model_output, predicted_variance = torch.split(
                    model_output, inputs.shape[1], dim=1
                )
            else:
                predicted_variance = None

            # 1. compute alphas, betas
            alpha_prod_t = self.train_scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.train_scheduler.alphas_cumprod[t - 1]
                if t > 0
                else self.train_scheduler.one
            )
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            # 2. compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            if self.train_scheduler.prediction_type == "epsilon":
                pred_original_sample = (
                    noisy_image - beta_prod_t ** (0.5) * model_output
                ) / alpha_prod_t ** (0.5)
            elif self.train_scheduler.prediction_type == "sample":
                pred_original_sample = model_output
            elif self.train_scheduler.prediction_type == "v_prediction":
                pred_original_sample = (alpha_prod_t**0.5) * noisy_image - (
                    beta_prod_t**0.5
                ) * model_output
            # 3. Clip "predicted x_0"
            if self.train_scheduler.clip_sample:
                pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

            # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_original_sample_coeff = (
                alpha_prod_t_prev ** (0.5) * self.train_scheduler.betas[t]
            ) / beta_prod_t
            current_sample_coeff = (
                self.train_scheduler.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
            )

            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            predicted_mean = (
                pred_original_sample_coeff * pred_original_sample
                + current_sample_coeff * noisy_image
            )
            # get the posterior mean and variance
            posterior_mean = self.train_scheduler._get_mean(
                timestep=t, x_0=inputs, x_t=noisy_image
            )
            posterior_variance = self.train_scheduler._get_variance(
                timestep=t, predicted_variance=predicted_variance
            )

            log_posterior_variance = torch.log(posterior_variance)
            log_predicted_variance = (
                torch.log(predicted_variance)
                if predicted_variance
                else log_posterior_variance
            )

            if t == 0:
                # compute -log p(x_0|x_1)
                kl = -self._get_decoder_log_likelihood(
                    inputs=inputs,
                    means=predicted_mean,
                    log_scales=0.5 * log_predicted_variance,
                    original_input_range=original_input_range,
                    scaled_input_range=scaled_input_range,
                )
            else:
                # compute kl between two normals
                kl = 0.5 * (
                    -1.0
                    + log_predicted_variance
                    - log_posterior_variance
                    + torch.exp(log_posterior_variance - log_predicted_variance)
                    + ((posterior_mean - predicted_mean) ** 2)
                    * torch.exp(-log_predicted_variance)
                )
            total_kl += kl.view(kl.shape[0], -1).mean(axis=1)
            if save_intermediates:
                intermediates.append(kl.cpu())

        if save_intermediates:
            return total_kl, intermediates
        else:
            return total_kl

    def _approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal. Code adapted from https://github.com/openai/improved-diffusion.
        """

        return 0.5 * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.Tensor([2.0 / math.pi]).to(x.device))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )

    def _get_decoder_log_likelihood(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        original_input_range: tuple | None = (0, 255),
        scaled_input_range: tuple | None = (0, 1),
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image. Code adapted from https://github.com/openai/improved-diffusion.
        Args:
            input: the target images. It is assumed that this was uint8 values,
                        rescaled to the range [-1, 1].
            means: the Gaussian mean Tensor.
            log_scales: the Gaussian log stddev Tensor.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
        """
        assert inputs.shape == means.shape
        bin_width = (scaled_input_range[1] - scaled_input_range[0]) / (
            original_input_range[1] - original_input_range[0]
        )
        centered_x = inputs - means
        inv_stdv = torch.exp(-log_scales)
        plus_in = inv_stdv * (centered_x + bin_width / 2)
        cdf_plus = self._approx_standard_normal_cdf(plus_in)
        min_in = inv_stdv * (centered_x - bin_width / 2)
        cdf_min = self._approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            inputs < -0.999,
            log_cdf_plus,
            torch.where(
                inputs > 0.999,
                log_one_minus_cdf_min,
                torch.log(cdf_delta.clamp(min=1e-12)),
            ),
        )
        assert log_probs.shape == inputs.shape
        return log_probs
