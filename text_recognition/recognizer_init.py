import torch
from .model import crnn
from .model.aster import *
from .model.moran import *

import text_recognition.model.aster as aster

import string
from collections import OrderedDict


def CRNN_init(model_path):
    model = crnn.CRNN(32, 1, 37, 256)
    print('loading pretrained crnn model from %s' % model_path)
    stat_dict = torch.load(model_path)
    model.load_state_dict(stat_dict)
    return model

def Aster_init(model_path):
    voc_type = 'all'
    max_len = 100
    EOS = 'EOS'
    PADDING = 'PADDING'
    UNKNOWN = 'UNKNOWN'

    voc = get_vocabulary(voc_type, EOS=EOS, PADDING=PADDING, UNKNOWN=UNKNOWN)
    rec_num_classes = len(voc)
    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))
    ASTER = aster.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=max_len,
                                             eos=char2id[EOS], STN_ON=True)

    ASTER.load_state_dict(torch.load(model_path)['state_dict'])
    aster_info = {}
    aster_info['EOS'] = EOS
    aster_info['UNKNOWN'] = UNKNOWN
    aster_info['char2id'] = char2id
    aster_info['id2char'] = id2char

    return ASTER, aster_info

def MORAN_init(model_path):
        # cfg = self.config.TRAIN
        alphabet = ':'.join(string.digits+string.ascii_lowercase+'$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        # model_path = self.config.TRAIN.VAL.moran_pretrained
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        # MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN)
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN


def get_vocabulary(voc_type, EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN'):
    '''
    voc_type: str: one of 'LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'
    '''
    voc = None
    types = ['digit', 'lower', 'upper', 'all']
    if voc_type == 'digit':
        voc = list(string.digits)
    elif voc_type == 'lower':
      voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'upper':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'all':
        voc = list(string.digits + string.ascii_letters + string.punctuation)
    elif voc_type == 'chinese':
        voc = list(open("al_chinese.txt", "r").readlines()[0].replace("\n", ""))
    else:
        raise KeyError('voc_type Error')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    return voc