import os
import random
from collections import OrderedDict
from typing import Union, Literal, List, Optional
from PIL import Image
import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GuidanceConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model_inpaint import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainInpaintProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.train_tools import precondition_model_outputs_flow_match

from utils import decode_and_save,save_mask_heatmaps_batch
ID2TAG = {
    0: "<ADD/REMOVE>",
    1: "<EDIT>",
    2: "<STYLE>",
}


prompt_to_task_id = {
    # ---------- ADD (0) ----------
    "Let the person wear sunglasses": 0,
    "Let the person wear a cowboy hat": 0,
    "Let the person wear a cap": 0,
    "Let the person wear earrings": 0,
    "Make the person wear a scarf": 0,
    "Make the person wear a suit and tie": 0,
    "Add some water stains": 0,
    "Add some coffee stain": 0,
    "Add some blue ink stains": 0,
    "Add some scratches": 0,

    "remove the person": 0,
    "remove text": 0,
    "remove the laptop": 0,
    "remove the plate": 0,

    # ---------- LOCAL EDIT (2) ----------
    "Change the person's hair color to red": 1,
    "Change the person's hair color to blonde": 1,
    "Change the person's hair color to blue": 1,
    "Make the person have curly hair": 1,
    "Make the person look older": 1,
    "Make the person look young": 1,
    "Change the background to a night scene": 1,
    "Change the background to a sunset view": 1,
    "Change the background to a forest": 1,
    "Change the background to a bright, snowy landscape": 1,
    "Change the background to an underwater scene": 1,
    
    "Make the credit card look like a beautiful birthday card": 1,
    "Make the credit card look like a leather credit card": 1,
    "Make the credit card look like a golden engraved credit card": 1,
    "Make the credit card look like a handwritten love letter": 1,
    "Transform the credit card into a luxurious VIP membership card": 1,
    "Make the credit card look like a marble credit card": 1,
    "Turn the credit card into an elegant wedding invitation": 1,
    "Make the credit card look like a wooden carved credit card": 1,
    "Make the person look like a cyborg": 1,
    "Turn the person into a cartoon character": 1,
    "Turn the person into an anime character": 1,
    "Make the person look like a superheroe": 1,
    "Turn the person into a vampire": 1,
    "Make the person look like a ghost": 1,

    # ---------- GLOBAL STYLE / SEMANTIC TRANSFORMATION (1) ----------

    "Turn to be vintage": 2,
    "Turn to be modern": 2,
    "Turn to be medieval": 2,
    "Turn to be high-tech": 2,
    "Make it look like the 1920s": 2,
    "Make it look like the 1980s": 2,
    "Make it look futuristic": 2,
    "Make it look like the Victorian era": 2,
    "Turn to an ancient Chinese style": 2,
    "Turn to a traditional Indian style": 2,
    "Turn to an ancient Egyptian style": 2,
    "Turn to a medieval European style": 2,

    "Turn to an oil painting style": 2,
    "Turn to a pencil sketch style": 2,
    "Turn to a watercolor painting style": 2,
    "Turn to a Monet-style watercolor painting": 2,
    "Turn to a Van Gogh-style painting": 2,

    "Make it look like a luxury brand advertisement": 2,
    "Make it look like a perfume advertisement": 2,
    "Make it look like a magazine cover": 2,
    "Make it look like an old, weathered paper": 2,
    "Make it look like an ancient manuscript": 2,


    "Make it look like a wooden carved ticket": 2,
    "Make it look like a golden engraved ticket": 2,
    "Make it look like a marble ticket": 2,
    "Make it look like a leather ticket": 2,
}



TASK2ID = {
    "ADD": 0,
    "REMOVE": 1,
    "REPLACE": 2,
    "STYLE": 3,
}

def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainInpaintProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.scaler = torch.cuda.amp.GradScaler()

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        # if self.train_config.dtype in ["fp16", "float16"]:
        #     # patch the scaler to allow fp16 training
        #     org_unscale_grads = self.scaler._unscale_grads_
        #     def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        #         return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        #     self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None

        self.dfe: Optional[DiffusionFeatureExtractor] = None

    def before_model_load(self):
        pass

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder:
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings

                print("\n***** UNLOADING TEXT ENCODER *****")
                print("This will train only with a blank prompt or trigger word, if set")
                print("If this is not what you want, remove the unload_text_encoder flag")
                print("***********************************")
                print("")
                self.sd.text_encoder_to(self.device_torch)
                self.cached_blank_embeds = self.sd.encode_prompt("")
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word)

                # move back to cpu
                self.sd.text_encoder_to('cpu')
                flush()


    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    def prompt_to_task_id(self,prompt: str):
        p = prompt.lower()
    
        if "remove" in p:
            return TASK2ID["REMOVE"]
    
        if any(k in p for k in ["wear", "add", "make the person wear"]):
            return TASK2ID["ADD"]
    
        if any(k in p for k in ["hair", "older", "younger", "cyborg"]):
            return TASK2ID["REPLACE"]
    
        if any(k in p for k in ["anime", "cartoon", "oil painting", "watercolor", "van gogh"]):
            return TASK2ID["STYLE"]
    
        if any(k in p for k in ["1920", "medieval", "ancient", "victorian", "futuristic"]):
            return TASK2ID["STYLE"]
    
        if any(k in p for k in ["background", "forest", "night", "underwater"]):
            return TASK2ID["REPLACE"]
    
        return TASK2ID["REPLACE"]


    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            weight_multiplier: Union[torch.Tensor, float] = 0.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        
        # target = (noise - batch.latents)[:, :, batch.latents.size(2) // 2:]
        target = (noise - batch.latents)[:,:,batch.latents.size(2)//2:,batch.latents.size(3)//2:].detach()
        target_keep = (noise - batch.latents)[:,:,batch.latents.size(2)//2:,:batch.latents.size(3)//2].detach()



        if target is None:
            target = noise

        #pred = noise_pred
        pred = noise_pred[:,:,noise_pred.size(2)//2:,noise_pred.size(3)//2:]
        # pred = noise_pred[:, :, noise_pred.size(2) // 2:]

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if self.train_config.loss_type == "mae":
            loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            loss_keep = torch.nn.functional.l1_loss(pred.float(), target_keep.float(), reduction="none")
        else:
            loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
            loss_keep = torch.nn.functional.mse_loss(pred.float(), target_keep.float(), reduction="none")

        # handle linear timesteps and only adjust the weight of the timesteps

        # multiply by our mask
        # print(has_mask,mask_multiplier.size())
        loss = loss * mask_multiplier
        loss_keep = loss_keep * weight_multiplier
        #loss = loss * mask
        #loss = loss / 4.0

        prior_loss = None
        loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        loss = loss * loss_multiplier
        if prior_loss is not None:
            loss = loss + prior_loss

        loss = loss.mean()
        loss_keep = loss_keep.mean()

        return loss + additional_loss + loss_keep


    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            **kwargs
        )

        return loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            scaler=self.scaler,
            **kwargs
        )

        return loss

    def get_guided_loss_targeted_polarity(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            # Perform targeted guidance (working title)
            dtype = get_torch_dtype(self.train_config.dtype)

            conditional_latents = batch.latents.to(self.device_torch, dtype=dtype).detach()
            unconditional_latents = batch.unconditional_latents.to(self.device_torch, dtype=dtype).detach()

            mean_latents = (conditional_latents + unconditional_latents) / 2.0

            unconditional_diff = (unconditional_latents - mean_latents)
            conditional_diff = (conditional_latents - mean_latents)

            # we need to determine the amount of signal and noise that would be present at the current timestep
            # conditional_signal = self.sd.add_noise(conditional_diff, torch.zeros_like(noise), timesteps)
            # unconditional_signal = self.sd.add_noise(torch.zeros_like(noise), unconditional_diff, timesteps)
            # unconditional_signal = self.sd.add_noise(unconditional_diff, torch.zeros_like(noise), timesteps)
            # conditional_blend = self.sd.add_noise(conditional_latents, unconditional_latents, timesteps)
            # unconditional_blend = self.sd.add_noise(unconditional_latents, conditional_latents, timesteps)

            # target_noise = noise + unconditional_signal

            conditional_noisy_latents = self.sd.add_noise(
                mean_latents,
                noise,
                timesteps
            ).detach()

            unconditional_noisy_latents = self.sd.add_noise(
                mean_latents,
                noise,
                timesteps
            ).detach()

            # Disable the LoRA network so we can predict parent network knowledge without it
            self.network.is_active = False
            self.sd.unet.eval()

            # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
            # This acts as our control to preserve the unaltered parts of the image.
            baseline_prediction = self.sd.predict_noise(
                latents=unconditional_noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
                timestep=timesteps,
                guidance_scale=1.0,
                **pred_kwargs  # adapter residuals in here
            ).detach()

            # double up everything to run it through all at once
            cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
            cat_latents = torch.cat([conditional_noisy_latents, conditional_noisy_latents], dim=0)
            cat_timesteps = torch.cat([timesteps, timesteps], dim=0)

            # since we are dividing the polarity from the middle out, we need to double our network
            # weights on training since the convergent point will be at half network strength

            negative_network_weights = [weight * -2.0 for weight in network_weight_list]
            positive_network_weights = [weight * 2.0 for weight in network_weight_list]
            cat_network_weight_list = positive_network_weights + negative_network_weights

            # turn the LoRA network back on.
            self.sd.unet.train()
            self.network.is_active = True

            self.network.multiplier = cat_network_weight_list

        # do our prediction with LoRA active on the scaled guidance latents
        prediction = self.sd.predict_noise(
            latents=cat_latents.to(self.device_torch, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(self.device_torch, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        )

        pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)

        pred_pos = pred_pos - baseline_prediction
        pred_neg = pred_neg - baseline_prediction

        pred_loss = torch.nn.functional.mse_loss(
            pred_pos.float(),
            unconditional_diff.float(),
            reduction="none"
        )
        pred_loss = pred_loss.mean([1, 2, 3])

        pred_neg_loss = torch.nn.functional.mse_loss(
            pred_neg.float(),
            conditional_diff.float(),
            reduction="none"
        )
        pred_neg_loss = pred_neg_loss.mean([1, 2, 3])

        loss = (pred_loss + pred_neg_loss) / 2.0

        # loss = self.apply_snr(loss, timesteps)
        loss = loss.mean()
        loss.backward()

        # detach it so parent class can run backward on no grads without throwing error
        loss = loss.detach()
        loss.requires_grad_(True)

        return loss

    def get_guided_loss_masked_polarity(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            **kwargs
    ):
        with torch.no_grad():
            # Perform targeted guidance (working title)
            dtype = get_torch_dtype(self.train_config.dtype)

            conditional_latents = batch.latents.to(self.device_torch, dtype=dtype).detach()
            unconditional_latents = batch.unconditional_latents.to(self.device_torch, dtype=dtype).detach()
            inverse_latents = unconditional_latents - (conditional_latents - unconditional_latents)

            mean_latents = (conditional_latents + unconditional_latents) / 2.0

            # unconditional_diff = (unconditional_latents - mean_latents)
            # conditional_diff = (conditional_latents - mean_latents)

            # we need to determine the amount of signal and noise that would be present at the current timestep
            # conditional_signal = self.sd.add_noise(conditional_diff, torch.zeros_like(noise), timesteps)
            # unconditional_signal = self.sd.add_noise(torch.zeros_like(noise), unconditional_diff, timesteps)
            # unconditional_signal = self.sd.add_noise(unconditional_diff, torch.zeros_like(noise), timesteps)
            # conditional_blend = self.sd.add_noise(conditional_latents, unconditional_latents, timesteps)
            # unconditional_blend = self.sd.add_noise(unconditional_latents, conditional_latents, timesteps)

            # make a differential mask
            differential_mask = torch.abs(conditional_latents - unconditional_latents)
            max_differential = \
                differential_mask.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            differential_scaler = 1.0 / max_differential
            differential_mask = differential_mask * differential_scaler
            spread_point = 0.1
            # adjust mask to amplify the differential at 0.1
            differential_mask = ((differential_mask - spread_point) * 10.0) + spread_point
            # clip it
            differential_mask = torch.clamp(differential_mask, 0.0, 1.0)

            # target_noise = noise + unconditional_signal

            conditional_noisy_latents = self.sd.add_noise(
                conditional_latents,
                noise,
                timesteps
            ).detach()

            unconditional_noisy_latents = self.sd.add_noise(
                unconditional_latents,
                noise,
                timesteps
            ).detach()

            inverse_noisy_latents = self.sd.add_noise(
                inverse_latents,
                noise,
                timesteps
            ).detach()

            # Disable the LoRA network so we can predict parent network knowledge without it
            self.network.is_active = False
            self.sd.unet.eval()

            # Predict noise to get a baseline of what the parent network wants to do with the latents + noise.
            # This acts as our control to preserve the unaltered parts of the image.
            # baseline_prediction = self.sd.predict_noise(
            #     latents=unconditional_noisy_latents.to(self.device_torch, dtype=dtype).detach(),
            #     conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype).detach(),
            #     timestep=timesteps,
            #     guidance_scale=1.0,
            #     **pred_kwargs  # adapter residuals in here
            # ).detach()

            # double up everything to run it through all at once
            cat_embeds = concat_prompt_embeds([conditional_embeds, conditional_embeds])
            cat_latents = torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0)
            cat_timesteps = torch.cat([timesteps, timesteps], dim=0)

            # since we are dividing the polarity from the middle out, we need to double our network
            # weights on training since the convergent point will be at half network strength

            negative_network_weights = [weight * -1.0 for weight in network_weight_list]
            positive_network_weights = [weight * 1.0 for weight in network_weight_list]
            cat_network_weight_list = positive_network_weights + negative_network_weights

            # turn the LoRA network back on.
            self.sd.unet.train()
            self.network.is_active = True

            self.network.multiplier = cat_network_weight_list

        # do our prediction with LoRA active on the scaled guidance latents
        prediction = self.sd.predict_noise(
            latents=cat_latents.to(self.device_torch, dtype=dtype).detach(),
            conditional_embeddings=cat_embeds.to(self.device_torch, dtype=dtype).detach(),
            timestep=cat_timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        )

        pred_pos, pred_neg = torch.chunk(prediction, 2, dim=0)

        # create a loss to balance the mean to 0 between the two predictions
        differential_mean_pred_loss = torch.abs(pred_pos - pred_neg).mean([1, 2, 3]) ** 2.0

        # pred_pos = pred_pos - baseline_prediction
        # pred_neg = pred_neg - baseline_prediction

        pred_loss = torch.nn.functional.mse_loss(
            pred_pos.float(),
            noise.float(),
            reduction="none"
        )
        # apply mask
        pred_loss = pred_loss * (1.0 + differential_mask)
        pred_loss = pred_loss.mean([1, 2, 3])

        pred_neg_loss = torch.nn.functional.mse_loss(
            pred_neg.float(),
            noise.float(),
            reduction="none"
        )
        # apply inverse mask
        pred_neg_loss = pred_neg_loss * (1.0 - differential_mask)
        pred_neg_loss = pred_neg_loss.mean([1, 2, 3])

        # make a loss to balance to losses of the pos and neg so they are equal
        # differential_mean_loss_loss = torch.abs(pred_loss - pred_neg_loss)
        #
        # differential_mean_loss = differential_mean_pred_loss + differential_mean_loss_loss
        #
        # # add a multiplier to balancing losses to make them the top priority
        # differential_mean_loss = differential_mean_loss

        # remove the grads from the negative as it is only a balancing loss
        # pred_neg_loss = pred_neg_loss.detach()

        # loss = pred_loss + pred_neg_loss + differential_mean_loss
        loss = pred_loss + pred_neg_loss

        # loss = self.apply_snr(loss, timesteps)
        loss = loss.mean()
        loss.backward()

        # detach it so parent class can run backward on no grads without throwing error
        loss = loss.detach()
        loss.requires_grad_(True)

        return loss

    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        if self.train_config.unload_text_encoder:
            raise ValueError("Prior predictions currently do not support unloading text encoder")
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                embeds_to_use = self.sd.encode_prompt(
                    prompt_list,
                    long_prompts=self.do_long_prompts).to(
                    self.device_torch,
                    dtype=dtype).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux:
                # we need to remove the image embeds from the prompt except for flux
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def predict_noise(
            self,
            noisy_latents: torch.Tensor,
            timesteps: Union[int, torch.Tensor] = 1,
            conditional_embeds: Union[PromptEmbeds, None] = None,
            unconditional_embeds: Union[PromptEmbeds, None] = None,
            **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=self.train_config.cfg_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            **kwargs
        )

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        self.timer.start('preprocess_batch')
        batch = self.preprocess_batch(batch)
        dtype = get_torch_dtype(self.train_config.dtype)
        # sanity check
        if self.sd.vae.dtype != self.sd.vae_torch_dtype:
            self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
        if isinstance(self.sd.text_encoder, list):
            for encoder in self.sd.text_encoder:
                if encoder.dtype != self.sd.te_torch_dtype:
                    encoder.to(self.sd.te_torch_dtype)
        else:
            if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                self.sd.text_encoder.to(self.sd.te_torch_dtype)

        noisy_latents, noise, timesteps, conditioned_prompts, imgs, noisy_latents_swap = self.process_general_training_batch(batch)
        


        network_weight_list = batch.get_network_weight_list()
        

        has_adapter_img = batch.control_tensor is not None
        has_clip_image = batch.clip_image_tensor is not None
        has_clip_image_embeds = batch.clip_image_embeds is not None
        # force it to be true if doing regs as we handle those differently
       
        match_adapter_assist = False



        self.timer.stop('preprocess_batch')

        is_reg = False
        with torch.no_grad():
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
          
            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            #batch.mask_tensor = self.sd.mask
            #print("batch.mask_tensor",batch.mask_tensor.size())

            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(noisy_latents.shape[2], noisy_latents.shape[3])
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()

                
        def get_weight_multiplier(noisy_latents, tau=0.5, eps=1e-6):
            """
            noisy_latents: [B, C, H, W]
            return: weight_multiplier [B, 1, h, w]
            """
            B, C, H, W = noisy_latents.shape
            h, w = H // 2, W // 2
        
            ul = noisy_latents[:, :, :h, :w]
            ur = noisy_latents[:, :, :h, w:]

            
            diffmap = torch.norm(ul - ur, p=2, dim=1, keepdim=True)

        
            # diff = torch.abs(ul - ur)  # [B, C, h, w]
            # diffmap = diff.mean(dim=1, keepdim=True)  # [B, 1, h, w]
        
            weight_multiplier = torch.exp(-diffmap / (tau + eps))
        
            return weight_multiplier

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True

            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            if self.tag_gen is not None:
                tag_gen = self.tag_gen
            else:
                tag_gen = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        clip_images=None
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            
            conditioned_prompts_empty_list = ["" for prompt in prompts_1]
            conditioned_prompts_empty_list = [conditioned_prompts_empty_list]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, conditioned_prompts_empty, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                conditioned_prompts_empty_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                tag_gen.train()
                B, C, H, W = noisy_latents.shape  # 例如 B=1, C=16, H=128, W=128
                H2, W2 = H // 2, W // 2
    
                # 拆出四块
                ref_before_latent  = noisy_latents[:, :, 0:H2, 0:W2]  # 左上
                ref_after_latent   = noisy_latents[:, :, 0:H2, W2:W]  # 右上

                logits = self.tag_gen(ref_before_latent, ref_after_latent) # logits: [B, num_tasks]
                # print(logits)
                task_gt = torch.tensor(
                    [prompt_to_task_id[p] for p in conditioned_prompts],
                    device=logits.device,
                    dtype=torch.long
                )
                # for p in conditioned_prompts:
                #     print(p)
                # print(task_gt)

                task_loss = torch.nn.functional.cross_entropy(logits, task_gt)
                print("task_loss",task_loss)

                task_pred = logits.argmax(dim=-1)  # [B]
                # print("task_pred",task_pred)

                def inject_task_tag(prompt, task_id):
                    tag = ID2TAG[task_id.item()]
                    return f"{tag} {prompt}"

                
                conditioned_prompts = [
                    inject_task_tag(p, t)
                    for p, t in zip(conditioned_prompts, task_pred)
                ]
                # print("inject",conditioned_prompts)
                # exit()

                
                
                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    
                    with torch.set_grad_enabled(False):
                        # make sure it is in eval mode
                        if isinstance(self.sd.text_encoder, list):
                            for te in self.sd.text_encoder:
                                te.eval()
                        else:
                            self.sd.text_encoder.eval()
                        if isinstance(self.adapter, CustomAdapter):
                            self.adapter.is_unconditional_run = False
                        conditional_embeds = self.sd.encode_prompt(
                            conditioned_prompts, prompt_2,
                            dropout_prob=self.train_config.prompt_dropout_prob,
                            long_prompts=self.do_long_prompts).to(
                            self.device_torch,
                            dtype=dtype)
                        conditional_embeds_empty  = self.sd.encode_prompt(
                            conditioned_prompts_empty, prompt_2,
                            dropout_prob=self.train_config.prompt_dropout_prob,
                            long_prompts=self.do_long_prompts).to(
                            self.device_torch,
                            dtype=dtype)
                        if self.train_config.do_cfg:
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = True
                            unconditional_embeds = self.sd.encode_prompt(
                                self.batch_negative_prompt,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts).to(
                                self.device_torch,
                                dtype=dtype)
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        conditional_embeds_empty = conditional_embeds_empty.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(
                                unconditional_embeds.text_embeds, 
                                is_unconditional=True
                            )

                # flush()
                pred_kwargs = {}

                

                
                prior_pred = None

                do_reg_prior = False
                # if is_reg and (self.network is not None or self.adapter is not None):
                #     # we are doing a reg image and we have a network or adapter
                #     do_reg_prior = True

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_correct_pred_norm_prior = self.train_config.correct_pred_norm

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True


                self.before_unet_predict()
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )

                else:
                    with self.timer('predict_unet'):
                        if unconditional_embeds is not None:
                            unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            **pred_kwargs
                        )
                    self.after_unet_predict()
                    weight_multiplier = get_weight_multiplier(noisy_latents)

                    

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        #if self.train_config.loss_calculate_type==1:
                        #print("loss_calculate_type: 1")
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=1.0,
                            weight_multiplier=weight_multiplier,
                            prior_pred=prior_pred,
                        )
                        # elif self.train_config.loss_calculate_type==2:
                        #     #print("loss_calculate_type: 2")
                        #     loss = self.calculate_loss2(
                        #         noise_pred=noise_pred,
                        #         noise=noise,
                        #         noisy_latents=noisy_latents,
                        #         timesteps=timesteps,
                        #         batch=batch,
                        #         mask_multiplier=mask_multiplier,
                        #         prior_pred=prior_pred,
                        #    )

                    # scheduler = self.sd.noise_scheduler

                    # 从 noisy_latents + noise_pred 反推出 x0
                    pred_x0 = noise-noise_pred
                
                    B, C, H, W = noisy_latents_swap.shape
                    h, w = H // 2, W // 2
                    
                    # 用 pred_x0 的左下角，替换 noisy_latents_swap 的左下角
                    noisy_latents_swap = noisy_latents_swap.clone()
                    noisy_latents_swap[:, :, h:H, 0:w] = pred_x0[:, :, h:H, w:W]


                    with self.timer('predict_unet_swap'):
                        noise_pred_swap = self.predict_noise(
                            noisy_latents=noisy_latents_swap.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds_empty.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            **pred_kwargs
                        )
                    
                    with self.timer('calculate_loss_swap'):
                        loss_swap = self.calculate_loss(
                            noise_pred=noise_pred_swap,
                            noise=noise,                         # 注意：noise 仍然是原 noise
                            noisy_latents=noisy_latents_swap,   # 👈 关键
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=1.0,
                            weight_multiplier=weight_multiplier,
                            prior_pred=prior_pred,
                        )
                        
                    


                # check if nan
                if torch.isnan(loss):
                    print("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean() + loss_swap * loss_multiplier.mean() + task_loss
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:
                    if not self.do_grad_scale:
                        loss.backward()
                    else:
                        self.scaler.scale(loss).backward()
                

        return loss.detach()
        # flush()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            loss = self.train_single_accumulation(batch)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if self.do_grad_scale:
                    self.scaler.unscale_(self.optimizer)
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        torch.nn.utils.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                # self.optimizer.step()
                if not self.do_grad_scale:
                    self.optimizer.step()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
                if self.adapter and isinstance(self.adapter, CustomAdapter):
                    self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        self.end_of_training_loop()

        return loss_dict
