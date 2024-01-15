import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
from torch.autograd import Variable
import copy

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """positional encoding

        Args:
            x (torch tensor): shape (B, 26, dim)

        Returns:
            torch tensor: posittional encoded x
        """
        # print("x in pos emb", x.shape)
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False).to(x.device)
        return self.dropout(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # print("before pos emb src", src.shape)
        q = k = self.with_pos_embed(src, pos)
        # print("attn_mask:", src_key_padding_mask.shape)
        src2 = self.self_attn(q, k, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):

        if type(src) == list:
            output = src[-1]
        else:
            output = src

        # print("q k", src.shape, src_key_padding_mask.shape)
        i = 0
        for layer in self.layers:
            if type(src) == list:
                src_item = src[i]
                i += 1
            else:
                src_item = src
            output = layer(output + src_item, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer_TP(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.d_model_self = 1024
        self.d_model = d_model

        self.height = 16
        self.width = 64

        print("nhead", nhead)
        print("dropout", dropout)
        print("d_model", d_model)



        self.self_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(self.d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("pos:", tensor.shape, pos.shape)
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        # L, N, C = tgt.shape

        #tgt2 = self.self_attn(q, k, tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)[0]
        #tgt = tgt + self.dropout1(tgt2)

        # print("tgt", tgt.shape)
        # print("memory", memory.shape)

        tgt2, attn_weights = self.multihead_attn(self.with_pos_embed(tgt, query_pos),
                                   self.with_pos_embed(memory, pos),
                                   memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask) # q, k, v
        # print("tgt2", tgt2.shape)
        

        # print("attn_weights:", np.unique(attn_weights[0].data.cpu().numpy()))

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                text_prior: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        attn_weights = None

        for layer in self.layers:

            # if not text_prior is None:
            #     output = output + self.norm(text_prior)

            output, attn_weights = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_weights

        return output, attn_weights


class InfoTransformer(nn.Module):

    def __init__(self, d_model=1024, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, feat_height=16, feat_width=64):
        super().__init__()

        

        # ConvTransformerDecoderLayer
        decoder_layer = TransformerDecoderLayer_TP(d_model, nhead, dim_feedforward,
                                                dropout, activation,
                                                normalize_before)#, feat_height=feat_height, feat_width=feat_width

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        # 1024
        self.gru_encoding = nn.GRU(d_model * feat_height, d_model * feat_height // 2, bidirectional=True, batch_first=True)
       

        # self.gru_encoding_horizontal = nn.GRU(d_model, d_model// 2, bidirectional=True,
        #                                                        batch_first=True)

        # self.gru_encoding_vertical = nn.GRU(d_model, d_model // 2, bidirectional=True,
        #                                       batch_first=True)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.feat_size = (feat_height, feat_width)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, memory, mask, query_embed, pos_embed, tgt=None, text_prior=None, spatial_size=(16, 64)):
        # flatten NxCxHxW to HWxNxC
        _, bs, hc = tgt.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # print("query_embed:", query_embed.shape)
        '''
        if not self.training:
            H, W = spatial_size
            up = int((W - H) / 2)
            bottom = H + int((W - H) / 2)
            query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)
            query_embed = query_embed[up:bottom, ...]
            query_embed = query_embed.reshape(spatial_size[0] * spatial_size[1], bs, hc)
        '''

        # print("shape:", tgt.shape, query_embed.shape)

        query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)\
            .permute(1, 2, 0, 3)\
            .reshape(self.feat_size[1], bs, self.feat_size[0] * hc)
        query_embed, _ = self.gru_encoding(query_embed)
        query_embed = query_embed.reshape(self.feat_size[1], bs, self.feat_size[0], hc)\
            .permute(2, 0, 1, 3)\
            .reshape(self.feat_size[0] * self.feat_size[1], bs, hc)

        '''
        query_embed = query_embed.reshape(self.feat_size[0], self.feat_size[1], bs, hc)
        #[H, B, C]
        query_embed_vertical = query_embed.mean(1)
        #[W, B, C]
        query_embed_horizontal = query_embed.mean(0)
        query_embed_vertical, _ = self.gru_encoding_vertical(query_embed_vertical)
        query_embed_horizontal, _ = self.gru_encoding_horizontal(query_embed_horizontal)
        # [H, 1, B, C] + [1, W, B, C]
        query_embed = query_embed_vertical.unsqueeze(1) + query_embed_horizontal.unsqueeze(0)
        query_embed = query_embed.reshape(self.feat_size[0] * self.feat_size[1], bs, hc)
        '''
        if tgt is None:
            tgt = torch.zeros_like(query_embed)

        # print("tgt (image):", tgt.shape)
        # print('src (text):', src.shape)
        # print('memory (text after encoding):', memory.shape)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs