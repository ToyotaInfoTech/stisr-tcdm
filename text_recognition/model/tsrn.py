import math
from turtle import forward
import torch
from torch import nn
from .transformer import *
from .tps_spatial_transformer import *
from .stn_head import *

SHUT_BN = False

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x

class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x

class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        # self.conv_proj = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.gru2 = GruBlock(channels, channels) # + text_channels

        # self.concat_conv = nn.Conv2d(channels + text_channels, channels, kernel_size=3, padding=1)

    def forward(self, x, text_emb):
        residual = self.conv1(x)
        if not SHUT_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if not SHUT_BN:
            residual = self.bn2(residual)

        ############ Fusing with TL ############
        cat_feature = torch.cat([residual, text_emb], 1)
        # residual = self.concat_conv(cat_feature)
        ########################################
        # fused_feature = self.conv_proj(cat_feature)

        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)

class TPInterpreter(nn.Module):
    def __init__(
                self,
                t_emb,
                out_text_channels,
                output_size=(16, 64),
                feature_in=64,
                # d_model=512,
                t_encoder_num=1,
                t_decoder_num=2,
                 ):
        super(TPInterpreter, self).__init__()

        d_model = out_text_channels # * output_size[0]

        
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        self.transformer = InfoTransformer(d_model=d_model,
                                          dropout=0.1,
                                          nhead=4,
                                          dim_feedforward=d_model,
                                          num_encoder_layers=t_encoder_num,
                                          num_decoder_layers=t_decoder_num,
                                          normalize_before=False,
                                          return_intermediate_dec=True, feat_height=output_size[0], feat_width=output_size[1])

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)
        # self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=26)


        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0] #output_size[1] ** 2 #
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, txt_emb):

        # H, W = self.output_size
        # x = tp_input

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i
        L, _, _ = txt_emb.shape


        x_tar = image_feature

        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)  # (1024, B, 64)
        # print(x_im.shape)

        device = image_feature.device
        # print('x:', x.shape)
        # x = x.permute(0, 3, 1, 2).squeeze(-1)  # (B, 26, 37)
        # x = self.activation(self.fc_in(x))  # (B, 26, 64)
        # N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N_i, L, C_i)).to(device)).permute(1, 0, 2)
        mask = torch.zeros((N_i, L)).to(device).bool()

        # print(x.shape)
        text_prior, pr_weights = self.transformer(txt_emb, mask, self.init_factor.weight, x_pos, tgt=x_im, spatial_size=(H, W)) # self.init_factor.weight
        # print(text_prior.shape)
        text_prior = text_prior.mean(0)

        # [N, L, C] -> [N, C, H, W]
        text_prior = text_prior.permute(1, 2, 0).view(N_i, C_i, H, W)
        # print(text_prior.shape)

        return text_prior, pr_weights

class Txt_Encoder_lr(nn.Module):
    def __init__(self, num_encoder_layers) -> None:
        super().__init__()

        t_emb = 37
        d_model = 64
        self.fc_in = nn.Linear(t_emb, d_model)
        self.activation = nn.PReLU()

        nhead = 4
        dim_feedforward = d_model
        dropout = 0.1
        activation = 'relu'
        normalize_before = False
        # num_encoder_layers = 1

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

    def forward(self, text_emb):
        x = text_emb
        x = x.permute(0, 3, 1, 2).squeeze(-1)  # (B, 26, 37)
        x = self.activation(self.fc_in(x))  # (B, 26, 64)
        N, L, C = x.shape
        x = x.permute(1, 0, 2)

        x_pos = self.pe(torch.zeros((N, L, C)).to(x.device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(x.device).bool()

        memory = self.encoder(x, src_key_padding_mask=mask, pos=x_pos)
        return memory


class Txt_Encoder_gt(nn.Module):
    def __init__(self, num_encoder_layers) -> None:
        super().__init__()

        t_emb = 63
        d_model = 64
        self.fc_in = nn.Linear(t_emb, d_model)
        self.activation = nn.PReLU()

        nhead = 4
        dim_feedforward = d_model
        dropout = 0.1
        activation = 'relu'
        normalize_before = False
        # num_encoder_layers = 1

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

    def forward(self, text_emb):
        x = text_emb
        x = x.permute(0, 3, 1, 2).squeeze(-1)  # (B, 26, 37)
        x = self.activation(self.fc_in(x))  # (B, 26, 64)
        N, L, C = x.shape
        x = x.permute(1, 0, 2)

        x_pos = self.pe(torch.zeros((N, L, C)).to(x.device)).permute(1, 0, 2)
        mask = torch.zeros((N, L)).to(x.device).bool()

        memory = self.encoder(x, src_key_padding_mask=mask, pos=x_pos)
        return memory



class TSRN_TL_TRANS(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #37, #26+26+1 3965
                 out_text_channels=64, # 32 256
                 feature_rotate=False,
                 rotate_train=3.):
        super(TSRN_TL_TRANS, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels)) #RecurrentResidualBlockTL

        self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(height//scale_factor, width//scale_factor)) # InfoGen(text_emb, out_text_channels)

        self.feature_rotate = feature_rotate
        self.rotate_train = rotate_train

        if not SHUT_BN:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        nn.BatchNorm2d(2 * hidden_units)
                    ))
        else:
            setattr(self, 'block%d' % (srb_nums + 2),
                    nn.Sequential(
                        nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                        # nn.BatchNorm2d(2 * hidden_units)
                    ))

        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        # block_ = []
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize,
                mode='tsrn')

        self.block_range = [k for k in range(2, self.srb_nums + 2)]

    # print("self.block_range:", self.block_range)

    def forward(self, x, text_emb=None, text_emb_gt=None, feature_arcs=None, rand_offs=None):
        # print(x.shape)
        # print(text_emb.shape)

        if self.stn and self.training:
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        if text_emb is None:
            text_emb = torch.zeros(1, 37, 1, 26).to(x.device) #37
        padding_feature = block['1']
        # print(padding_feature.shape)

        tp_map_gt, pr_weights_gt = None, None
        tp_map, pr_weights = self.infoGen(padding_feature, text_emb)
        # N, C, H, W

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in self.block_range:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], tp_map)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)])) #

        output = torch.tanh(block[str(self.srb_nums + 3)])
        # print('output shape', output.shape)

        self.block = block

        if self.training:
            ret_mid = {
                "pr_weights": pr_weights,
                "pr_weights_gt": pr_weights_gt,
                "spatial_t_emb": tp_map,
                "spatial_t_emb_gt": tp_map_gt,
                "in_feat": block["1"],
                "trans_feat": tp_map
            }

            return output, ret_mid

        else:
            return output, pr_weights
