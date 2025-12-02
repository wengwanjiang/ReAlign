from typing import Optional, Union

import torch
import torch.nn as nn

from mld.models.operator.embeddings import TimestepEmbedding, Timesteps
from mld.models.operator.attention import (SkipTransformerEncoder,
                                           SkipTransformerDecoder,
                                           TransformerDecoderLayer,
                                           TransformerEncoderLayer)

from mld.models.operator.utils import lengths_to_mask
from mld.models.operator.position_encoding import build_position_encoding


def load_balancing_loss_func(router_logits: tuple, num_experts: int = 4, topk: int = 2):
    router_logits = torch.cat(router_logits, dim=0)
    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, topk, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)
    overall_loss = num_experts * torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss


class MldDenoiser(nn.Module):

    def __init__(self,
                 latent_dim: list = [1, 256],
                 text_dim: int = 512,
                 time_dim: int = 512,
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 return_intermediate_dec: bool = False,
                 freq_shift: float = 0,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 #guidance_scale: float = 7.5, For A2M 
                 #guidance_uncondp: float = 0.1, For A2M
                 condition: str = "text",
                 ) -> None:
        super(MldDenoiser, self).__init__()

        self.latent_dim = latent_dim[-1]
        self.arch = arch
        self.condition = condition
        self.text_encoded_dim = text_dim
        if self.condition in ["text", "text_uncond"] or True:
            self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
            self.time_embedding = TimestepEmbedding(time_dim, self.latent_dim)
        elif self.condition == "action":
            assert 1 == 2, "Not implemented yet"
        if text_dim != self.latent_dim:
            self.emb_proj = nn.Sequential(nn.ReLU(), nn.Linear(text_dim, self.latent_dim))
        

        self.query_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)
        # self.mem_pos = build_position_encoding(self.latent_dim, position_embedding=position_embedding)


        if self.arch == "trans_enc":        
            encoder_layer = TransformerEncoderLayer(self.latent_dim, num_heads, 
                                                    ff_size, dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(self.latent_dim)
            self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)

        elif self.arch == 'trans_dec':
            decoder_layer = TransformerDecoderLayer(self.latent_dim, num_heads, ff_size, dropout,
                                                    activation, normalize_before)

            decoder_norm = nn.LayerNorm(self.latent_dim)
            self.decoder = SkipTransformerDecoder(decoder_layer, num_layers, decoder_norm, 
                                                  return_intermediate=return_intermediate_dec,)
        else:
            raise ValueError(f"Not supported architecture: {self.arch}!")


    def forward(self,
                sample: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                lengths = None) -> tuple:
        # 0.  dimension matching
        # sample [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        sample = sample.permute(1, 0, 2)
        if lengths not in [None, []]:
            mask = lengths_to_mask(lengths, sample.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML

        timesteps = timestep.expand(sample.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=sample.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        # text_emb [seq_len, batch_size, text_dim] <= [batch_size, seq_len, text_dim]
        if self.condition in ["text", "text_uncond"]:
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
            if self.text_encoded_dim != self.latent_dim:
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
        elif self.condition == "action":
            assert 1 == 2, "Not implemented yet"
        # text embedding projection
        #print(time_emb.shape, text_emb_latent.shape)
        #exit(0)
        emb_latent = torch.cat((time_emb, text_emb_latent), 0)

        # 3. transformer
        if self.arch == "trans_enc":
            xseq = torch.cat((sample, emb_latent), axis=0)
            xseq = self.query_pos(xseq)
            tokens = self.encoder(xseq)
            sample = tokens[:sample.shape[0]]
        elif self.arch == 'trans_dec':
            sample = self.query_pos(sample)
            emb_latent = self.mem_pos(emb_latent)
            sample = self.decoder(tgt=sample, memory=emb_latent).squeeze(0)
        else:
            raise TypeError(f"{self.arch} is not supported")
        
        sample = sample.permute(1, 0, 2)
        return (sample, )
