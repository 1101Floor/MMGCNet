import copy
from typing import Optional  # need to pay attention

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from model.Transformer.transformerlib import MultiheadAttention


class Attention(Module):

    def __init__(self, d_model, nhead, dropout=0., motion_only=True, cross_range=0, num_conv_layer=3):
        super().__init__()

        self.model_dim = d_model

        self.temporal_attention = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_info = nn.Linear(d_model, d_model)
        self.temporal_gate = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=False, attn_mask=None, seq_mask=False):

        # [T N sample_num model_dim]
        assert len(query.shape) == len(key.shape) == len(value.shape) == 4
        assert query.shape[1] == key.shape[1] == value.shape[1]  # check equality of num_agent
        assert query.shape[2] == key.shape[2] == value.shape[2]  # check equality of sample_num
        assert key.shape[0] == value.shape[0]
        num_agent = query.shape[1]
        sample_num = query.shape[2]
        query_len = query.shape[0]  # T - query
        key_len = key.shape[0]  # T - key and value

        # zeros of [N N]
        spatial_mask = attn_mask

        # generate temporal mask, since temporal attention may not be sparse
        temporal_mask = torch.zeros([query_len, key_len], device=attn_mask.device)
        if seq_mask and query_len == key_len:
            for i in range(query_len - 1):
                temporal_mask[i, i + 1:] = float('-inf')

        # [T N*sample_num model_dim]
        t_query = query.reshape([query_len, num_agent * sample_num, self.model_dim])
        t_key = key.reshape([key_len, num_agent * sample_num, self.model_dim])
        t_value = value.reshape([key_len, num_agent * sample_num, self.model_dim])
        t_attn, t_weights = self.temporal_attention(t_query, t_key, t_value, key_padding_mask=key_padding_mask,
                                                    need_weights=need_weights, attn_mask=temporal_mask)  # input

        t_info = torch.tanh(self.temporal_info(t_attn))  
        t_gate = torch.sigmoid(self.temporal_gate(t_attn)) 
        t_out = t_info * t_gate
        t_out = t_out.reshape([query_len, num_agent, sample_num, self.model_dim])

        attn_weights = None
        
        return t_out, attn_weights


class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.self_attn = Attention(d_model, nhead, dropout=dropout, motion_only=False, cross_range=2, num_conv_layer=7)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", cross_motion_only=False):
        super().__init__()

        self.self_attn = Attention(d_model, nhead, dropout=dropout, motion_only=False, cross_range=2, num_conv_layer=7)
        self.PGA = Attention(d_model, nhead, dropout=dropout, motion_only=cross_motion_only, cross_range=2, num_conv_layer=7)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.cross_motion_only = cross_motion_only

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, seq_mask = False,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, need_weights = False) -> Tensor:
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, seq_mask=seq_mask,
                                                 key_padding_mask=tgt_key_padding_mask,
                                                 need_weights=need_weights)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, PGA_weights = self.PGA(tgt, memory, memory, attn_mask=memory_mask,
                                                   key_padding_mask=memory_key_padding_mask,
                                                   need_weights=need_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, PGA_weights 


class TransformerEncoder(Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, num_agent=1) -> Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, seq_mask = False, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, need_weights = False, num_agent=1) -> Tensor:
        output = tgt

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, self_attn_weights[i], cross_attn_weights[i] = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask, seq_mask=seq_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         need_weights=need_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
            cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()

        return output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))