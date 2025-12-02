import time, random
import inspect
import logging
from typing import Optional

import tqdm, pickle
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from mld.data.base import BaseDataModule
from mld.config import instantiate_from_config
from mld.utils.temos_utils import lengths_to_mask, remove_padding
from mld.utils.utils import count_parameters, get_guidance_scale_embedding, extract_into_tensor

from .base import BaseModel
logger = logging.getLogger(__name__)


from ReAlignModule.models.spm import process_T5_outputs
class MLD(BaseModel):
    def __init__(self, cfg: DictConfig, datamodule: BaseDataModule) -> None:
        super().__init__()

        self.cfg = cfg
        self.nfeats = cfg.DATASET.NFEATS
        self.njoints = cfg.DATASET.NJOINTS
        self.latent_dim = cfg.model.latent_dim
        self.guidance_scale = cfg.model.guidance_scale
        self.datamodule = datamodule

        if cfg.model.guidance_scale == 'dynamic':
            s_cfg = cfg.model.scheduler
            self.guidance_scale = s_cfg.cfg_step_map[s_cfg.num_inference_steps]
            logger.info(f'Guidance Scale set as {self.guidance_scale}')

        self.text_encoder = instantiate_from_config(cfg.model.text_encoder)
        self.vae = instantiate_from_config(cfg.model.motion_vae)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.noise_scheduler = instantiate_from_config(cfg.model.noise_scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self._get_t2m_evaluator(cfg)

        self.metric_list = cfg.METRIC.TYPE
        self.configure_metrics()

        self.feats2joints = datamodule.feats2joints

        self.vae_scale_factor = cfg.model.get("vae_scale_factor", 1.0)
        self.guidance_uncondp = cfg.model.get('guidance_uncondp', 0.0)

        logger.info(f"vae_scale_factor: {self.vae_scale_factor}")
        logger.info(f"prediction_type: {self.scheduler.config.prediction_type}")
        logger.info(f"guidance_scale: {self.guidance_scale}")
        logger.info(f"guidance_uncondp: {self.guidance_uncondp}")

    
        

        self.summarize_parameters()
        self.do_classifier_free_guidance = self.guidance_scale > 1.0
        
    def summarize_parameters(self) -> None:
        logger.info(f'VAE Encoder: {count_parameters(self.vae.encoder)}M')
        logger.info(f'VAE Decoder: {count_parameters(self.vae.decoder)}M')
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')


    def forward(self, batch: dict) -> tuple:
        texts = batch["text"]
        feats_ref = batch.get("motion")
        lengths = batch["length"]
        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)
        
        # text_emb = self.text_encoder(texts)
        t_len, sent_emb, text_emb = process_T5_outputs(texts, self.text_encoder.text_model)
        latents = torch.randn((len(lengths), *self.latent_dim), device=text_emb.device)
        #mask = batch.get('mask', lengths_to_mask(lengths, text_emb.device))s
        latents = self._diffusion_reverse(latents, text_emb)
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, lengths)

        joints = self.feats2joints(feats_rst.detach().cpu())
        joints = remove_padding(joints, lengths)

        joints_ref = None
        if feats_ref is not None:
            joints_ref = self.feats2joints(feats_ref.detach().cpu())
            joints_ref = remove_padding(joints_ref, lengths)

        return joints, joints_ref

    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        assert 1 == 2
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon

    def _diffusion_reverse(self, cond, rm_args) -> torch.Tensor:
        m_len = rm_args['m_len']
        bsz = cond.shape[0]
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
        latents = torch.randn((bsz, *self.latent_dim), device=cond.device, dtype=torch.float)
        latents = latents * self.scheduler.init_noise_sigma
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(cond.device)
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            lengths_reverse = (m_len * 2 if self.do_classifier_free_guidance else m_len)
            # predict the noise residual
            noise_pred = self.denoiser(sample=latent_model_input, timestep=t,
                                    encoder_hidden_states=cond, lengths=lengths_reverse)[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
            if self.cfg.lambda_t2m != 0 or self.cfg.lambda_m2m != 0:
                with torch.enable_grad():
                    xt = latents.permute(1, 0, 2).clone()
                    xt = xt.requires_grad_()
                    motion_recons = self.vae.decode(xt, rm_args['m_len'])
                    rm_args['timestep'] = t.clone()
                    latents = self.RewardModel.GradGuidance_MLD(xt, motion_recons, rm_args)
                    latents = latents.permute(1, 0, 2)
                    
            # 正确的位置
        latents = latents.permute(1, 0, 2)
        return latents
    def _diffusion_process(self, latents: torch.Tensor, encoder_hidden_states: torch.Tensor, lengths) -> dict:

        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

    
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps, # 1000
            (bsz,),
            device=latents.device
        )
        timesteps = timesteps.long()
        noisy_latents = self.scheduler.add_noise(latents.clone(), noise, timesteps)
        
        
        noise_pred = self.denoiser(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            lengths=lengths)[0]

        n_set = {
            "noise": noise,
            "noise_pred": noise_pred,
            #"sample_pred": latents_pred,
            "sample_pred": 0,
            "sample_gt": latents
        }
        return n_set

    def train_diffusion_forward(self, batch: dict) -> dict:
        feats_ref = batch["motion"]
        mask = batch['mask']
        lengths=batch["length"]
        with torch.no_grad():
            z, dist = self.vae.encode(feats_ref, lengths)
            z = z * self.vae_scale_factor

        text = batch["text"]
        text = [
            "" if np.random.rand(1) < self.guidance_uncondp else i
            for i in text
        ]
        t_len, sent_emb, text_emb = self.text_encoder(text)
        n_set = self._diffusion_process(z, text_emb, lengths=lengths)

        loss_dict = dict()

        
        # DM
        if self.scheduler.config.prediction_type == "epsilon":
            model_pred, target = n_set['noise_pred'], n_set['noise']
        elif self.scheduler.config.prediction_type == "sample":
            model_pred, target = n_set['sample_pred'], n_set['sample_gt']
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")
        diff_loss = F.mse_loss(model_pred, target, reduction="mean")

        loss_dict['diff_loss'] = diff_loss

        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        return loss_dict

    def t2m_eval(self, batch: dict) -> dict:

        texts = batch["text"]
        bsz = len(texts)
            
        feats_ref = batch["motion"]
        #mask = batch['mask']
        lengths = batch["length"]
        word_embs = batch["word_embs"]
        pos_ohot = batch["pos_ohot"]
        text_lengths = batch["text_len"]
        
        start = time.time()

        if self.datamodule.is_mm:
            texts = texts * self.cfg.TEST.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            #mask = mask.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.cfg.TEST.MM_NUM_REPEATS

            word_embs = word_embs.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.TEST.MM_NUM_REPEATS, dim=0)

        

        if self.do_classifier_free_guidance:
            texts = texts + [""] * len(texts)

        text_st = time.time()
        # text_emb = self.text_encoder(texts)
        t_len, sent_emb, text_emb = process_T5_outputs(texts, self.text_encoder.text_model)
        text_et = time.time()
        self.text_encoder_times.append(text_et - text_st)
      

        diff_st = time.time()
        
        rm_args = {'text':texts[:bsz], 'm_len':lengths[:bsz], 'sent_emb':sent_emb[:bsz], 't_len':t_len[:bsz], 'closest_motion':None, 'cls_token': text_emb[:bsz]}

        latents = self._diffusion_reverse(cond=text_emb, rm_args=rm_args)
        diff_et = time.time()
        self.diffusion_times.append(diff_et - diff_st)

        vae_st = time.time()
        feats_rst = self.vae.decode(latents / self.vae_scale_factor, lengths)        
        vae_et = time.time()
        self.vae_decode_times.append(vae_et - vae_st)
        self.frames.extend(lengths)

        end = time.time()
        self.times.append(end - start)

        # joints recover
        joints_rst = self.feats2joints(feats_rst)
        joints_ref = self.feats2joints(feats_ref)

        # renorm for t2m evaluators
        feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_ref = self.datamodule.renorm4t2m(feats_ref)

        # t2m motion encoder
        m_lens = lengths.copy()
        m_lens = torch.tensor(m_lens, device=feats_ref.device)
        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        feats_ref = feats_ref[align_idx]
        feats_rst = feats_rst[align_idx]
        m_lens = m_lens[align_idx]
        m_lens = torch.div(m_lens, eval(f"self.cfg.DATASET.{self.cfg.DATASET.NAME.upper()}.UNIT_LEN"),
                           rounding_mode="floor")

        recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
        recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
        motion_mov = self.t2m_moveencoder(feats_ref[..., :-4]).detach()
        motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

        # t2m text encoder
        
        text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

        rs_set = {"m_ref": feats_ref, "m_rst": feats_rst,
                  "lat_t": text_emb, "lat_m": motion_emb, "lat_rm": recons_emb,
                  "joints_ref": joints_ref, "joints_rst": joints_rst}


        return rs_set

    def allsplit_step(self, split: str, batch: dict) -> Optional[dict]:
        
        if split in ["test", "val"]:
            rs_set = self.t2m_eval(batch)

            if self.datamodule.is_mm:
                metric_list = ['MMMetrics']
            else:
                metric_list = self.metric_list

            for metric in metric_list:
                if metric == "TM2TMetrics":
                    getattr(self, metric).update(
                        rs_set["lat_t"],
                        rs_set["lat_rm"],
                        rs_set["lat_m"],
                        batch["length"])
                elif metric == "MMMetrics" and self.datamodule.is_mm:
                    getattr(self, metric).update(rs_set["lat_rm"].unsqueeze(0), batch["length"])
                else:
                    raise TypeError(f"Not support this metric: {metric}.")

        if split in ["train", "val"]:
            loss_dict = self.train_diffusion_forward(batch)
            return loss_dict
