import torch
from model.tsrn import *
# from loss.image_loss import *


class TATT(object):
    def __init__(self, cfg):
        self.tatt_init(cfg)

    def tatt_init(self, cfg):

        model = TSRN_TL_TRANS(scale_factor=cfg.down_sample_scale, width=cfg.width, height=cfg.height,
                                    STN=cfg.STN, mask=cfg.mask, srb_nums=cfg.srb_nums,
                                    hidden_units=cfg.hd_u)


        # if not cfg.resume == '':
        #     print('loading pre-trained model from %s ' % cfg.resume)
        #     # model.load_state_dict(torch.load(resume))
        #     model.load_state_dict(torch.load(cfg.resume)['state_dict_G'])

        self.model = model
        # self.crit = image_crit