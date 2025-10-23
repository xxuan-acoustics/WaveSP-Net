import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
import numpy as np
from typing import Union
import torchvision.models as models
import torchaudio
from dataclasses import dataclass
from exp.learnable_wavelet_domain_sparse_PT import *


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # att_map = torch.matmul(att_map, self.att_weight12)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)

        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))  # self.mp = nn.MaxPool2d((1,4))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        # print('out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        #print('conv2 out',out.shape)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out



class Rawaasist(nn.Module):
    def __init__(self):
        super().__init__()

        filts = [70, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims =  [64, 32]
        pool_ratios = [0.5, 0.7, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        self.conv_time = CONV(out_channels=filts[0],
                              kernel_size=128,
                              in_channels=1)
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])

        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=False):

        x = x.unsqueeze(1).cuda()
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # get embeddings using encoder
        # (#bs, #filt, #spec, #seq)
        e = self.encoder(x)

        # spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        '''
        x1  :(#bs, #node, #dim)
        x2  :(#bs, #node, #dim)
        '''
        # print('x1',x1.shape)
        # print('x2',x2.shape)
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        # print('num_type1',num_type1)
        # print('num_type2',num_type2)
        x1 = self.proj_type1(x1)
        # print('proj_type1',x1.shape)
        x2 = self.proj_type2(x2)
        # print('proj_type2',x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('Concat x1 and x2',x.shape)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
            # print('master',master.shape)
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)
        # print('master',master.shape)
        # directional edge for master node
        master = self._update_master(x, master)
        # print('master',master.shape)
        # projection
        x = self._project(x, att_map)
        # print('proj x',x.shape)
        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        # print('x1',x1.shape)
        x2 = x.narrow(1, num_type1, num_type2)
        # print('x2',x2.shape)
        return x1, x2, master

    def _update_master(self, x, master):

        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)

        return master

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))

        att_map = torch.matmul(att_map, self.att_weightM)

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _derive_att_map(self, x, num_type1, num_type2):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board

        # apply temperature
        att_map = att_map / self.temp

        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _project_master(self, x, master, att_map):

        x1 = self.proj_with_attM(torch.matmul(
            att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        # print('out',out.shape)
        out = self.conv1(x)

        # print('aft conv1 out',out.shape)
        out = self.bn2(out)
        out = self.selu(out)
        # print('out',out.shape)
        out = self.conv2(out)
        # print('conv2 out',out.shape)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        # out = self.mp(out)
        return out


class SSLAASIST(nn.Module):
    def __init__(self):
        super().__init__()

        # AASIST parameters
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        ####
        # create network wav2vec 2.0
        ####

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # RawNet2 encoder
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))
        self.LL = nn.Linear(1024, 128)
        
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),

        )
        # position encoding
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        # Graph module
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1],
                                               gat_dims[0],
                                               temperature=temperatures[1])
        # HS-GAL layer
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2])

        # Graph pooling layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)


    def forward(self, x):

        x = x.squeeze(dim=1)

        x = self.LL(x)
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.unsqueeze(dim=1)  # add channel
        x = F.max_pool2d(x, (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # RawNet2-based encoder
        x = self.encoder(x)
        x = self.first_bn1(x)
        x = self.selu(x)

        w = self.attention(x)

        # ------------SA for spectral feature-------------#
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S

        # graph module layer
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (#bs, #node, #dim)

        # ------------SA for temporal feature-------------#
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)

        e_T = m1.transpose(1, 2)

        # graph module layer
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # learnable master node
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # inference 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1)

        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # Readout operation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)

        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden,output




class XLSRAASIST(nn.Module):
    def __init__(self, model_dir, device='cuda', freeze = True, visual=False):
        super(XLSRAASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2 = XLSR(
            model_dir=model_dir,
            device=device,
            freeze=freeze,
            visual=visual
        )

        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()
        self.visual = visual
    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
            return last_hidden, output, attention_weights
        # Extract features using XLSRWithPrompt
        features = self.wav2vec2.extract_features(audio_data)

        # Pass the features through W2VAASIST
        last_hidden, output = self.w2vaasist(features)
        return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.w2vaasist.train(mode)
        else:
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2.eval()   



    
        
class ResNet18ForAudio(nn.Module):
    def __init__(self, enc_dim=256, nclasses=2):
        super(ResNet18ForAudio, self).__init__()

        self.resnet18 = models.resnet18(pretrained=False)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(9, 3), stride=(3, 1), padding=(1, 1), bias=False)
        self.resnet18.fc = nn.Identity()

        self.fc = nn.Linear(512, enc_dim)
        self.fc_mu = nn.Linear(enc_dim, nclasses) if nclasses >= 2 else nn.Linear(enc_dim, 1)

        self.spec = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, win_length=512, power=2, normalized=True)

        self.initialize_params()

    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def forward(self, x):
        x = self.spec(x.cuda().float()).unsqueeze(dim=1)
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)  
        feat = self.fc(x)
        mu = self.fc_mu(feat)
        return feat, mu



from mamba_ssm.modules.mamba_simple import Mamba


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, n_state):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model
        
        self.mamba = Mamba(d_model, n_state)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # self.concat_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )


    def forward(self, x):

        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence

        # x_flip = self.norm1(x_flip)#2.14加
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # print("mamba_out_forward",mamba_out_forward.shape)#torch.Size([20, 208, 144])
        # print("mamba_out_backward",mamba_out_backward.shape)#torch.Size([20, 208, 144])

        # ADD forward and backward
        mamba_out = mamba_out_forward + mamba_out_backward
        # print("add_mamba_out",mamba_out.shape) #([20, 208, 144])
        ff_out = self.feed_forward(mamba_out)
        ff_out = self.norm2(ff_out)

        output = ff_out + residual
        return output

    
class BiMambas_FFN(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.,
        conv_causal = False
    ):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(BiMambaEncoder(
                d_model = dim, 
                n_state = 16
            ))

    def forward(self, x):

        for block in self.layers:
            x = block(x)

        return x
    
class PTW2V2AASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=10, dropout=0.1, visual=False):
        super(PTW2V2AASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()

            
class WPTW2V2AASIST(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=5, num_wavelet_tokens=6, dropout=0.1, visual=False):
        super(WPTW2V2AASIST, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = WPT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens= num_wavelet_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = visual
        # Initialize W2VAASIST (main model)
        self.w2vaasist = SSLAASIST()

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
        if self.visual:
            features, attention_weights = self.wav2vec2_with_prompt.extract_features(audio_data)
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output, attention_weights
        else:
            features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Pass the features through W2VAASIST
            last_hidden, output = self.w2vaasist(features)
        
            return last_hidden, output

    def train(self, mode=True):
        # Set train status for both components
        if mode:
            self.wav2vec2_with_prompt.train(mode)
            self.w2vaasist.train(mode)
        else:
            self.wav2vec2_with_prompt.eval()
            self.w2vaasist.eval()

    def eval(self):
        # Set eval status for both components
        self.w2vaasist.eval()
        self.wav2vec2_with_prompt.eval()


class PT_XLSR_BiMamba(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=6, num_wavelet_tokens=4, dropout=0.1, visual=False):
        super(PT_XLSR_BiMamba, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = PT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = False
        self.LL = nn.Linear(1024, 144)
        print('PT_Fake_Mamba')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension


        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)


    def forward(self, audio_data, return_embedding=False):
        # Extract features using XLSRWithPrompt
    
        features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(features)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # embedding = x[:, 0, :] 
        

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        # if return_embedding:
        #     return self.fc5(x_pooled), x_pooled
        #     # return self.fc5(x_pooled), embedding
        
        return x_pooled, self.fc5(x_pooled)



class WPT_XLSR_BiMamba(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=6, num_wavelet_tokens=4, dropout=0.1, visual=False):
        super(WPT_XLSR_BiMamba, self).__init__()

        # Initialize XLSRWithPrompt (features extractor)
        self.wav2vec2_with_prompt = WPT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_wavelet_tokens= num_wavelet_tokens,
            dropout=dropout,
            visual=visual
        )
        self.visual = False
        self.LL = nn.Linear(1024, 144)
        print('WPT_Fake_Mamba')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension


        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
    
        features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(features)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)


        # embedding = x[:, 0, :] 
        

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        # if return_embedding:
        #     return self.fc5(x_pooled), x_pooled
        #     # return self.fc5(x_pooled), embedding
        
        return x_pooled, self.fc5(x_pooled)



class FourierPT_XLSR_BiMamba(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=6, num_fourier_tokens=4, dropout=0.1, visual=False):
        super(FourierPT_XLSR_BiMamba, self).__init__()

        # Initialize FourierPT_XLSR (features extractor)
        self.wav2vec2_with_prompt = FourierPT_XLSR(
            model_dir=model_dir,
            prompt_dim=prompt_dim,
            device=device,
            sampling_rate=sampling_rate,
            num_prompt_tokens=num_prompt_tokens,
            num_fourier_tokens= num_fourier_tokens,
            dropout=dropout
        )
        self.visual = False
        self.LL = nn.Linear(1024, 144)
        print('FourierPT_Fake_Mamba')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension


        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
    
        features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(features)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # embedding = x[:, 0, :]  

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        # if return_embedding:
        #     return self.fc5(x_pooled), x_pooled
        #     # return self.fc5(x_pooled), embedding
        
        return x_pooled, self.fc5(x_pooled)



class WaveSP_Net(nn.Module):
    def __init__(self, model_dir, prompt_dim=1024, device='cuda', sampling_rate=16000, num_prompt_tokens=6, num_wavelet_tokens=4, dropout=0.1, visual=False):
        super(WaveSP_Net, self).__init__()

        # Initialize Partial-WSPT-XLSR (features extractor)
        self.wav2vec2_with_prompt = Partial_WSPT_XLSR(
            model_dir= model_dir,
            prompt_dim=1024,
            device=device,
            num_prompt_tokens=6,       
            num_wavelet_tokens=4,       
            # num_wavelet_tokens=10,      
            sparsity_ratio=0.01,       
            filter_length=8,           
            dropout=0.1,
            visual=False  
        )
        self.visual = False
        self.LL = nn.Linear(1024, 144)
        print('WaveSP_Net')

        # Additional layers before encoder
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Encoder (BiMamba)
        self.encoder = BiMambas_FFN(dim=144, depth=12)

        # Pooling layer (Attention Pooling)
        self.attention_pool = nn.Linear(144, 1)  # Output attention weights along the time dimension


        # Fully connected layer for classification
        self.fc5 = nn.Linear(144, 2)  # Binary classification (real vs fake)

    def forward(self, audio_data):
        # Extract features using XLSRWithPrompt
    
        features = self.wav2vec2_with_prompt.extract_features(audio_data)
        # Step 2: Apply linear layer to reduce feature dimension
        x = self.LL(features)  # (B, T, emb_size) -> (B, T, 144)

        # Step 3: Preprocess features for encoder
        x = x.unsqueeze(dim=1)  # Add channel dimension -> (B, 1, T, 144)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)  # Remove channel dimension -> (B, T, 144)

        # Step 4: Pass through BiMamba encoder
        x = self.encoder(x)  # (B, T, 144)

        # embedding = x[:, 0, :]  
        

        # Step 5: Attention pooling along the time dimension
        attention_weights = F.softmax(self.attention_pool(x), dim=1)  # Compute weights along time dimension -> (B, T, 1)
        x_pooled = torch.matmul(
            attention_weights.transpose(-1, -2), x
        ).squeeze(-2)  # Weighted sum along time dimension -> (B, 144)

        # Step 6: Classification head
        # x_pooled=x[:,0,:] #[bs, emb_size]

        # if return_embedding:
        #     return self.fc5(x_pooled), x_pooled
        #     # return self.fc5(x_pooled), embedding
        
        return x_pooled, self.fc5(x_pooled)
    