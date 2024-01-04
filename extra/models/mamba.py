from __future__ import annotations
import math
import json
import time
from typing import Any, Union
import tinygrad.nn as nn
from tinygrad.tensor import Tensor
from dataclasses import dataclass
from tinygrad.nn.state import torch_load, load_state_dict
from extra.models.mask_rcnn import topk
import numpy as np


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
            
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple

class Mamba:
    def __init__(self, args: ModelArgs):
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = [ResidualBlock(args) for _ in range(args.n_layer)]
        self.norm_f = RMSNorm(args.d_model)
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights. See "Weight Tying" paper

    def __call__(self, input_ids:Tensor):
      x = self.embedding(input_ids)
      for layer in self.layers:
        x = layer(x)
      x = self.norm_f(x)
      logits = self.lm_head(x)
      return logits
    
    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.
    
        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'
                            
        Returns:
            model: Mamba model with weights loaded
    
        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file
        
        def load_config_hf(model_name):
            resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return json.load(open(resolved_archive_file))
        
        
        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
                                                _raise_exceptions_for_missing_entries=False)
            return torch_load(resolved_archive_file)
        
        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data['d_model'],
            n_layer=config_data['n_layer'],
            vocab_size=config_data['vocab_size']
        )
        model = Mamba(args)
        
        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = state_dict[key]
        load_state_dict(model, new_state_dict)
        return model


    
        

class ResidualBlock:
    def __init__(self, args:ModelArgs):
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
    def __call__(self, x:Tensor):
       return self.mixer(self.norm(x)) + x
    

class MambaBlock:
    def __init__(self, args:ModelArgs):
      self.args = args
      self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
      self.conv1d = nn.Conv1d(in_channels=args.d_inner, out_channels=args.d_inner, kernel_size=args.d_conv, groups=args.d_inner, bias=args.conv_bias, padding=args.d_conv-1)
      self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
      self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
      
      A = Tensor.arange(1, args.d_state + 1)
      A = A.repeat((args.d_inner, 1))
      self.A_log = Tensor.log(A)
      self.D = Tensor.ones(args.d_inner)
      self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
    
    def __call__(self, x:Tensor):
        (b,l,d) = x.shape
        x_and_res = self.in_proj(x)
        (x,res) = x_and_res.split(sizes=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)[:, :, :l]
        x = x.permute(0, 2, 1)

        x = x.silu()
        
        y = self.ssm(x)
        y = y * Tensor.silu(res)
        output = self.out_proj(y)
        return output


    def ssm(self,x:Tensor):
       (d_in, n) = self.A_log.shape
       A = -self.A_log.float().exp()
       D = self.D.float()
       x_dbl = self.x_proj(x)
       (delta, B , C) = x_dbl.split(sizes=[self.args.dt_rank, n, n], dim=-1)
       delta = self.dt_proj(delta).softplus()
       y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
       return y
    
    def selective_scan(self, u:Tensor, delta:Tensor, A:Tensor, B:Tensor, C:Tensor, D:Tensor):
       (b, l, d_in) = u.shape
       n = A.shape[1]
       deltaA = Tensor.einsum('b l d, d n -> b l d n', delta,A).exp()
       deltaB_u = Tensor.einsum('b l d, b l n, b l d -> b l d n', delta,B,u)
       x = Tensor.zeros((b, d_in, n))
       ys = []
       for i in range(l):
          x = (deltaA[:, i] * x) + deltaB_u[:, i]
          y = Tensor.einsum('b d n, b n -> b d', x, C[:, i, :])
          ys.append(y)
       y = Tensor.stack(ys, dim=1)
  
       y = y + u*D
       return y
       
class RMSNorm:
  def __init__(self, dim, eps=1e-6):
    self.eps = eps
    self.weight = Tensor.ones(dim)

  def __call__(self, x:Tensor):
    return (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt()) * self.weight
  

def generate(model, tokenizer,prompt: str,n_tokens_to_gen: int = 25, sample: bool = True, top_k: int = 40):
 
    input_ids = tokenizer(prompt, return_tensors='np').input_ids
    Tensor.no_grad = True
   
    for _ in range(n_tokens_to_gen):
        indices_to_input: Tensor = Tensor(input_ids)
        next_token_logits:Tensor = model(indices_to_input)[:, -1]
        probs = next_token_logits.softmax( axis=-1)
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            start = time.time()
            (values, indices) = topk(probs, k=top_k)
            print(f"topk took {time.time() - start}")
            probs = Tensor.where(probs < values[:, -1, None], Tensor.zeros_like(probs), probs)
            probs = probs / probs.sum(axis=1, keepdim=True)
            
        if sample:
            next_indices = probs.multinomial(num_samples=1).item()
        else:
            next_indices = probs.argmax(dim=-1)[:, None]
        next_indices = np.array([[next_indices]])
        input_ids = np.concatenate((input_ids, next_indices), axis=1)
        print(tokenizer.decode(input_ids.tolist()[0]))

        
    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions
