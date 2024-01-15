from torch.utils.data import Dataset
import torch
import numpy as np
import lmdb
import random
from PIL import Image
import six
import os

from text_recognition.recognizer_init import *
from text_recognition.utils import *

def get_eval_string(word, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    new_word = [ch for ch in word if ch in alphabet]
    new_word = ''.join(new_word)
    return new_word

class str_dataset(Dataset):
    def __init__(self, root, d_name):
        super(str_dataset, self).__init__()

        txn, nSamples = self.get_txn(os.path.join(f"{root}/training/label/real", d_name))

        self.txn = txn
        self.nSamples = nSamples
        print('n_samples:', self.nSamples)

    def __len__(self):
        return self.nSamples

    def get_txn(self, root):
        env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        txn = env.begin(write=False)
        nSamples = int(txn.get(b'num-samples'))
        return txn, nSamples

    def buf2PIL(self, txn, key, type='RGB'):
        imgbuf = txn.get(key)
        if imgbuf is None:
            print("===============================")
            return None

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im

    def __getitem__(self, idx):

        idx += 1
        label_key = b"label-%09d" % idx
        img_key = b"image-%09d" % idx

        img = self.buf2PIL(self.txn, img_key, 'RGB')
        word = self.txn.get(label_key)
        word = str(word.decode())


        return img, word


if __name__ == "__main__":

    root = "../dataset/data_CVPR2021"
    out_path = "./dataset/STR"
    d_name_list = [ '1.SVT', "2.IIIT", "3.IC13", '4.IC15', '6.RCTW17', '7.Uber', '8.ArT', '9.LSVT', '10.MLT19', '11.ReCTS']
    # d_name_list = [ '1.SVT']

    text_rec = CRNN_init("./text_recognition/ckpt/crnn.pth")
    text_rec.cuda()
    text_rec.eval()

    for d_name in d_name_list:

        img_list = []
        word_list = []
        data = str_dataset(root, d_name)
        for i in range(len(data)):
            img, word = data[i]
            img = img.resize((128, 32), Image.BICUBIC)
            img_arr = np.array(img).astype(np.uint8)
            img_arr = img_arr.transpose(2,0,1)
            img_arr = img_arr[np.newaxis,...]
            img_list.append(img_arr)
            word_list.append(word)

        imgs = np.concatenate(img_list)

        # ====  Split and samples ====
        n_samples = imgs.shape[0]
        batch_size = 1000
        n_batches = math.ceil(n_samples/batch_size)
        print(f'batch_size: {batch_size}, n_batches: {n_batches}')
        sample_list = [imgs[batch_size*b:batch_size*(b+1)] for b in range(n_batches)]
        word_list_list = [word_list[batch_size*b:batch_size*(b+1)] for b in range(n_batches)]
        for bi, (split_samples, split_word_list) in enumerate(zip(sample_list, word_list_list)):
            print(split_samples.shape, len(split_word_list))
            split_batch_size = len(split_word_list)

            # ====  check text recognition acc of hr samples ====
            samples_ = torch.from_numpy(split_samples).float()
            samples_ = samples_ / 255
            samples_ = samples_.cuda()

            pred, _ = text_rec(samples_)
            pred_words = get_string_crnn(pred)
            is_correct_list = []
            for pred_word, gt_word in zip(pred_words, split_word_list):
                gt_word = gt_word.lower()
                gt_word = get_eval_string(gt_word)
                # print(pred_word, gt_word)
                if pred_word == gt_word:
                    is_correct_list.append(True)
                else:
                    is_correct_list.append(False)

            # ====  check word length ====
            is_valid_wl_list = []
            for j in range(split_batch_size):
                word = split_word_list[j]
                if len(word) < 25:
                    is_valid_wl_list.append(True)
                else:
                    is_valid_wl_list.append(False)

            # ====  make valid hr-lr pairs ====
            is_correct_tens = torch.tensor(is_correct_list)
            is_valid_wl_tens = torch.tensor(is_valid_wl_list)
            is_valid_tens = torch.logical_and(is_valid_wl_tens, is_correct_tens)
            # is_valid_tens = is_high_var_tens
            print('acc: {}, valid rate: {}'.format(np.mean(is_correct_list), np.mean(is_valid_wl_list)))

            valid_ori_samples = split_samples[is_valid_tens]
            valid_word_list = [b for a, b in zip(is_valid_tens.tolist(), split_word_list) if a]
            # valid_word_list = split_word_list[is_valid_tens.tolist()]
            print(valid_ori_samples.shape, len(valid_word_list))

            outpath_img = os.path.join(out_path, 'img')
            outpath_word = os.path.join(out_path, 'word')
            out_filename_img = d_name + '_' + str(bi) + '.npz'
            out_filename_word = d_name + '_' + str(bi) + '.txt'
            np.savez(os.path.join(outpath_img, out_filename_img), valid_ori_samples)
            with open(os.path.join(outpath_word, out_filename_word), 'w') as f:
                for word in valid_word_list:
                    f.write(word)
                    f.write('\n')     

            torch.cuda.empty_cache()
