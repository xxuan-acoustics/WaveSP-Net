import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from pytorch_wavelets import DWTForward
import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from pytorch_wavelets import DWTForward


class WaveletBlock(nn.Module):
    def __init__(self, wave='haar', J=2, input_dim=1024, output_dim=1024):
        super(WaveletBlock, self).__init__()
        self.dwt = DWTForward(J=J, wave=wave)  
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        """
        input: (batch, token, dim)
        output: (batch, token, dim)
        """
        B, T, D = x.shape
        assert D == self.input_dim, f"Input dimension (dim={D}) must match WaveletBlock's input_dim ({self.input_dim})"

        x = x.unsqueeze(dim=1) # channel 1
        # wavelet transform
        LL, band = self.dwt(x)  #  LL([b, 1, 3, 512]) band （LH/HL/HH）([b, 1, 3, 3, 512])
        bands= band[0]  
        LL = LL.unsqueeze(dim=2) #  LL([b, 1, 3, 1, 512]) 
        # print(bands.shape, 'bands')
        # print(LL.shape,'LL')
        features = torch.cat((LL, bands), dim=2).view(B, -1, D)# (batch, token, output_dim)
        return features


class WPT_XLSR(torch.nn.Module):
    def __init__(self, model_dir, prompt_dim, device='cuda', sampling_rate=16000, num_prompt_tokens=6, num_wavelet_tokens = 4
                 , dropout=0.1,visual=False):
        super(WPT_XLSR, self).__init__()

        # Set device (GPU or CPU)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = sampling_rate

        # Load the pre-trained model configuration and weights
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)

        # Enable output of hidden states
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.visual = visual    
        # Create a learnable prompt embedding for 24 layers
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens 
        self.num_wavelet_tokens = num_wavelet_tokens
        self.prompt_embedding = nn.Parameter(torch.zeros(24, self.num_prompt_tokens, prompt_dim)) 
        self.fprompt_embedding = nn.Parameter(torch.zeros(24, self.num_wavelet_tokens, prompt_dim)) 
        self.wavelet_block = WaveletBlock(wave='haar', J=1, input_dim=1024, output_dim=1024)
        # Xavier initialization for prompt_embedding
        val = math.sqrt(6. / float(2 * prompt_dim))  # Xavier initialization factor
        nn.init.uniform_(self.prompt_embedding.data, -val, val)
        nn.init.uniform_(self.fprompt_embedding.data, -val, val)
        # Dropout layer for the prompt
        self.prompt_dropout = nn.Dropout(p=dropout)
        
    def forward(self, audio_data):
        # Process the input audio using Wav2Vec2 Feature Extractor
        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        feat = feat.squeeze(dim=0)  
        
        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            feat = feat.transpose(1, 2)
            # Feature projection
            hidden_state, extract_features = self.model.feature_projection(feat)
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            hidden_state = hidden_state + position_embeddings
            hidden_state = self.model.encoder.dropout(hidden_state) # equal to hidden_state = hidden_states[0]

        B = feat.size(0)  
        if self.visual:
            all_self_attentions = []
            
        for i in range(self.model.config.num_hidden_layers):
            if i == 0:
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt) 
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt,prompt, hidden_state), dim=1)
                if self.visual:
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]
            else:    
                prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                prompt = self.prompt_dropout(prompt)  # Apply dropout to prompt
                fprompt = self.fprompt_embedding[i].expand(B, -1, -1).to(self.device)
                fprompt = self.prompt_dropout(self.wavelet_block(fprompt))
                hidden_state = torch.cat((fprompt, prompt, hidden_state[:, self.num_prompt_tokens + fprompt.shape[1]:, :]), dim=1)
                if self.visual: 
                    hidden_state, attention_weight = self.model.encoder.layers[i](hidden_state, output_attentions=self.visual)
                    all_self_attentions.append(attention_weight)  
                else:
                    hidden_state = self.model.encoder.layers[i](hidden_state)[0]    
                    
        if self.visual:
            print(len(all_self_attentions), "all_self_attentions")
            return hidden_state,all_self_attentions
        else:
            # print(hidden_state.shape,'hidden_state')
            return hidden_state


    def extract_features(self, audio_data):
        # Process the input audio and extract the features using the forward pass
        return self.forward(audio_data)  # Return the final layer's output


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from abc import ABC, abstractmethod
from typing import Tuple 

# =============================================================================
# LEARNABLE WAVELET-DOMAIN SPARSE PROMPT TUNING
# =============================================================================

class WaveletFilter(ABC):
    
    @property
    @abstractmethod
    def filter_bank(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class SoftOrthogonalWavelet(WaveletFilter, nn.Module):
    def __init__(self, dec_lo: torch.Tensor, dec_hi: torch.Tensor, 
                 rec_lo: torch.Tensor, rec_hi: torch.Tensor):
        super().__init__()
        self.dec_lo = nn.Parameter(dec_lo)  
        self.dec_hi = nn.Parameter(dec_hi)  
        self.rec_lo = nn.Parameter(rec_lo)  
        self.rec_hi = nn.Parameter(rec_hi)  

    @property
    def filter_bank(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dec_lo, self.dec_hi, self.rec_lo, self.rec_hi

    def __len__(self) -> int:
        return self.dec_lo.shape[-1]



class WaveCore(nn.Module):
    
    def __init__(self, dim=1024, sparsity_ratio=0.01, filter_length=8):
        super().__init__()
        self.dim = dim
        self.sparsity_ratio = sparsity_ratio
        self.filter_length = filter_length
        
        self._initialize_learnable_wavelet()
        
        self._create_sparse_system()
        
    def _initialize_learnable_wavelet(self):
        if self.filter_length == 2:
            # Haar wavelet as initialization
            sqrt2 = math.sqrt(2)
            dec_lo = torch.tensor([1.0, 1.0]) / sqrt2
            dec_hi = torch.tensor([1.0, -1.0]) / sqrt2
            rec_lo = torch.tensor([1.0, 1.0]) / sqrt2  
            rec_hi = torch.tensor([1.0, -1.0]) / sqrt2
        else:
            base_lo = torch.tensor([1.0, 1.0]) / math.sqrt(2)
            base_hi = torch.tensor([1.0, -1.0]) / math.sqrt(2)
            
            dec_lo = F.pad(base_lo, (0, self.filter_length - 2)) + torch.randn(self.filter_length) * 0.01
            dec_hi = F.pad(base_hi, (0, self.filter_length - 2)) + torch.randn(self.filter_length) * 0.01
            rec_lo = dec_lo.clone()
            rec_hi = dec_hi.clone()
            
        self.wavelet = SoftOrthogonalWavelet(dec_lo, dec_hi, rec_lo, rec_hi)
        
    def _create_sparse_system(self):
        total_coeffs = self.dim
        self.num_sparse_params = max(1, int(total_coeffs * self.sparsity_ratio))
        
        # Trainable sparse wavelet coefficients
        self.sparse_coeffs = nn.Parameter(torch.zeros(self.num_sparse_params))
        
        sparse_indices = torch.randperm(total_coeffs)[:self.num_sparse_params]
        self.register_buffer('sparse_indices', sparse_indices)
        
        # Xavier initialization
        std = math.sqrt(2.0 / self.dim)
        nn.init.normal_(self.sparse_coeffs.data, 0, std)
    
    def _dwt_1d(self, x):
        dec_lo, dec_hi, _, _ = self.wavelet.filter_bank
        batch_size, seq_len = x.shape
        
        pad_size = len(dec_lo) - 1
        x_padded = F.pad(x, (pad_size, pad_size), mode='reflect')
        x_expanded = x_padded.unsqueeze(1) 
        
        coeffs_lo = F.conv1d(x_expanded, dec_lo.flip(0).unsqueeze(0).unsqueeze(0))
        coeffs_hi = F.conv1d(x_expanded, dec_hi.flip(0).unsqueeze(0).unsqueeze(0))
        
        coeffs_lo = coeffs_lo[:, :, ::2].squeeze(1)
        coeffs_hi = coeffs_hi[:, :, ::2].squeeze(1)
        
        return coeffs_lo, coeffs_hi
    
    def _idwt_1d(self, coeffs_lo, coeffs_hi, target_length):

        _, _, rec_lo, rec_hi = self.wavelet.filter_bank
        batch_size = coeffs_lo.size(0)
        coeffs_len = coeffs_lo.size(1)
        

        upsampled_lo = torch.zeros(batch_size, coeffs_len * 2, device=coeffs_lo.device)
        upsampled_hi = torch.zeros(batch_size, coeffs_len * 2, device=coeffs_hi.device)
        upsampled_lo[:, ::2] = coeffs_lo
        upsampled_hi[:, ::2] = coeffs_hi
        

        pad_size = len(rec_lo) - 1
        upsampled_lo = F.pad(upsampled_lo.unsqueeze(1), (pad_size, pad_size), mode='reflect')
        upsampled_hi = F.pad(upsampled_hi.unsqueeze(1), (pad_size, pad_size), mode='reflect')
        
        recon_lo = F.conv1d(upsampled_lo, rec_lo.unsqueeze(0).unsqueeze(0))
        recon_hi = F.conv1d(upsampled_hi, rec_hi.unsqueeze(0).unsqueeze(0))
        
        reconstruction = (recon_lo + recon_hi).squeeze(1)
        
        if reconstruction.size(1) != target_length:
            if reconstruction.size(1) > target_length:
                reconstruction = reconstruction[:, :target_length]
            else:
                pad_needed = target_length - reconstruction.size(1)
                reconstruction = F.pad(reconstruction, (0, pad_needed))
                
        return reconstruction
    
    def _apply_sparse_update(self, coeffs_lo, coeffs_hi):

        batch_size = coeffs_lo.size(0)
        
        flat_coeffs = torch.cat([coeffs_lo.flatten(1), coeffs_hi.flatten(1)], dim=1)
        total_coeffs = flat_coeffs.size(1)
        
        sparse_update = torch.zeros(batch_size, total_coeffs, device=flat_coeffs.device)
        
        valid_indices = self.sparse_indices[self.sparse_indices < total_coeffs]
        if len(valid_indices) > 0:
            sparse_params = self.sparse_coeffs[:len(valid_indices)]
            sparse_update[:, valid_indices] = sparse_params.unsqueeze(0).expand(batch_size, -1)
        
        updated_coeffs = flat_coeffs + sparse_update
        
        lo_size = coeffs_lo.numel() // batch_size
        updated_lo = updated_coeffs[:, :lo_size].view_as(coeffs_lo)
        updated_hi = updated_coeffs[:, lo_size:lo_size*2].view_as(coeffs_hi)
        
        return updated_lo, updated_hi
    
    def forward(self, x):

        batch_size = x.size(0)
        
        # 1. Learnable Wavelet Decomposition (LWD)
        coeffs_lo, coeffs_hi = self._dwt_1d(x)
        
        # 2. Wavelet Domain Sparsification (WDS)
        updated_lo, updated_hi = self._apply_sparse_update(coeffs_lo, coeffs_hi)
        
        # 3.  Learnable Wavelet Reconstruction (LWR)
        enhanced_x = self._idwt_1d(updated_lo, updated_hi, self.dim)
        
        return x + enhanced_x
    
    # ================Ablation Experiments================

    # # =========Ablation Experiments 1 - w/o LWD=========
    # def forward(self, x):
    #     coeffs_lo, coeffs_hi = x, torch.zeros_like(x)  
    #     # coeffs_lo, coeffs_hi = self._dwt_1d(x)        
        
    #     updated_lo, updated_hi = self._apply_sparse_update(coeffs_lo, coeffs_hi)
    #     enhanced_x = self._idwt_1d(updated_lo, updated_hi, self.dim)
    #     output = x + enhanced_x
    #     return output


    # # =========Ablation Experiments 2 - w/o WDS=========
    # def forward(self, x):
    #     coeffs_lo, coeffs_hi = self._dwt_1d(x) 
    #     updated_lo, updated_hi = coeffs_lo, coeffs_hi         
    #     enhanced_x = self._idwt_1d(updated_lo, updated_hi, self.dim)
    #     output = x + enhanced_x
    #     return output


    # # =========Ablation Experiments 3 - w/o LWR=========
    # def forward(self, x):
    #     coeffs_lo, coeffs_hi = self._dwt_1d(x)
    #     updated_lo, updated_hi = self._apply_sparse_update(coeffs_lo, coeffs_hi)

    #     if updated_lo.shape[-1] != self.dim:
    #         enhanced_x = torch.nn.functional.interpolate(
    #             updated_lo.unsqueeze(1), size=self.dim, mode='linear', align_corners=False
    #         ).squeeze(1)
    #     else:
    #         enhanced_x = updated_lo
        
    #     output = x + enhanced_x
    #     return output



class WSPTLayer(nn.Module):

    def __init__(self, prompt_dim=1024, num_tokens=4, sparsity_ratio=0.01, filter_length=8):
        super().__init__()
        self.prompt_dim = prompt_dim
        self.num_tokens = num_tokens
    
        self.base_embedding = nn.Parameter(torch.zeros(num_tokens, prompt_dim))
        
        self.core = WaveCore(
            dim=prompt_dim, 
            sparsity_ratio=sparsity_ratio,
            filter_length=filter_length
        )
        
        val = math.sqrt(6. / float(2 * prompt_dim))
        nn.init.uniform_(self.base_embedding.data, -val, val)
        # val = √(6 / (2 * 1024)) 
        #     = √(6 / 2048) 
        #     = √0.0029296875 
        #     ≈ 0.054126
    
    def forward(self, batch_size=1):

        base_prompt = self.base_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        
        enhanced_tokens = []
        for i in range(self.num_tokens):
            token = base_prompt[:, i, :]  
        
            enhanced_token = self.core(token)  

            enhanced_token = enhanced_token.unsqueeze(1) 
            
            enhanced_tokens.append(enhanced_token)
        
        enhanced_prompt = torch.cat(enhanced_tokens, dim=1)
        
        return enhanced_prompt
    

class Partial_WSPT_XLSR(nn.Module):

    def __init__(self, model_dir, prompt_dim, device, sampling_rate=16000, 
                 num_prompt_tokens=6, num_waveft_tokens=4, sparsity_ratio=0.01,
                 filter_length=8, enable_traditional_prompt=True,
                 dropout=0.1, visual=False):
        super().__init__()
        
        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = torch.device('cpu')
            else:
                self.device = torch.device(device)
        else:
            self.device = device
        self.sampling_rate = sampling_rate
        self.visual = visual
        
        self.config = Wav2Vec2Config.from_json_file(f"{model_dir}/config.json")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2Model.from_pretrained(model_dir).to(self.device)
        
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True  
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False  
            
        # config
        self.prompt_dim = prompt_dim
        self.num_prompt_tokens = num_prompt_tokens
        self.num_waveft_tokens = num_waveft_tokens
        self.sparsity_ratio = sparsity_ratio
        self.enable_traditional_prompt = enable_traditional_prompt
        

        if enable_traditional_prompt:
            self.prompt_embedding = nn.Parameter(torch.zeros(24, num_prompt_tokens, prompt_dim))
            val = math.sqrt(6. / float(2 * prompt_dim))
            nn.init.uniform_(self.prompt_embedding.data, -val, val)
        
        
        # WSPT
        self.wspt_layers = nn.ModuleList([
                WSPTLayer(
                prompt_dim=prompt_dim,
                num_tokens=num_waveft_tokens, 
                sparsity_ratio=sparsity_ratio,
                filter_length=filter_length
            ) for _ in range(24)  
        ])

        # Dropout
        self.prompt_dropout = nn.Dropout(p=dropout)
    
    def forward(self, audio_data):
        # print("audio_data",audio_data.shape)#torch.Size([16, 64600])     

        feat = self.processor(audio_data, sampling_rate=self.sampling_rate, 
                            return_tensors="pt").input_values.to(self.device)
        # print("feat1",feat.shape)#torch.Size([1, 16, 64600])
        feat = feat.squeeze(dim=0)
        # print("feat2",feat.shape)#torch.Size([16, 64600])

        with torch.no_grad():
            feat = self.model.feature_extractor(feat)
            # print("feat3",feat.shape)# torch.Size([16, 512, 201])
            feat = feat.transpose(1, 2)
            # print("feat4",feat.shape)#torch.Size([16, 201, 512])
            hidden_state, extract_features = self.model.feature_projection(feat)
            # print("hidden_state1",hidden_state.shape)#torch.Size([16, 201, 1024])
            position_embeddings = self.model.encoder.pos_conv_embed(hidden_state)
            # print("position_embeddings",position_embeddings.shape)# torch.Size([16, 201, 1024])
            hidden_state = hidden_state + position_embeddings
            # print("hidden_state2",hidden_state.shape)#torch.Size([16, 201, 1024])
            hidden_state = self.model.encoder.dropout(hidden_state)
            # print("hidden_state3",hidden_state.shape)#torch.Size([16, 201, 1024])
        
        B = feat.size(0)
        # print("B",B) #16
        if self.visual:
            all_self_attentions = []
        
        for i in range(self.model.config.num_hidden_layers): 
            prompt_tokens = []
        
            if self.enable_traditional_prompt:
                traditional_prompt = self.prompt_embedding[i].expand(B, -1, -1).to(self.device)
                traditional_prompt = self.prompt_dropout(traditional_prompt)
                prompt_tokens.append(traditional_prompt)  

            wspt_prompt = self.waveft_layers[i](batch_size=B)
            wspt_prompt = self.prompt_dropout(wspt_prompt)
            prompt_tokens.append(wspt_prompt)  

            all_prompts = torch.cat(prompt_tokens, dim=1)
            # print(f"Layer {i} - All prompts concatenated shape: {all_prompts.shape}")  # torch.Size([16, 10, 1024])
            
            if i == 0:

                hidden_state = torch.cat((all_prompts, hidden_state), dim=1)
                # print("hidden_state_1",all_prompts.shape)#torch.Size([16, 10, 1024])
            else:
                total_prompt_len = all_prompts.size(1)
                # print("total_prompt_len",total_prompt_len)#total_prompt_len 10
                hidden_state = torch.cat((all_prompts, hidden_state[:, total_prompt_len:, :]), dim=1)
                # print("hidden_state_2",hidden_state.shape)#hidden_state_2 torch.Size([16, 211, 1024])
            
            if self.visual:
                hidden_state, attention_weight = self.model.encoder.layers[i](
                    hidden_state, output_attentions=self.visual)
                all_self_attentions.append(attention_weight)
            else:
                hidden_state = self.model.encoder.layers[i](hidden_state)[0]
        

        if self.visual:
            return hidden_state, all_self_attentions
        else:
            return hidden_state
    
    def extract_features(self, audio_data):
        return self.forward(audio_data)
    
    
    
    
if __name__ == "__main__":

    wav = torch.ones(2, 64600).cuda()

    # model = WPT_XLSR(model_dir='/xxuan/pre-model/huggingface/wav2vec2-xls-r-300m/', prompt_dim=1024, num_prompt_tokens=6, num_wavelet_tokens=4).cuda()

    model = Partial_WSPT_XLSR(
        model_dir="path/to/wav2vec2-xlsr",
        prompt_dim=1024,
        num_prompt_tokens=6,
        num_waveft_tokens=4,
        sparsity_ratio=0.01,  
        filter_length=8,      
        wavelet_type='soft_orthogonal',  
        dropout=0.1
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params_in_million = trainable_params / 1e6

    print(f"Trainable parameters (in million): {trainable_params_in_million:.4f}M")

    features = model(wav)
    
    model.print_model_info()
