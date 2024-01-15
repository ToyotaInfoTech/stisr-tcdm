import random

from PIL import Image
import blobfile as bf

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import lmdb
import six
import os
import glob


def load_data_textzoom_test():
    dataset = TextZoomDataset_SR_Test()
    # print("Number of dataset:", len(dataset))
    return dataset


def load_data_textzoom(batch_size):
    dataset = TextZoomDataset()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader

def load_data_textzoom_deg(batch_size):
    dataset = TextZoomDataset_Deg()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader


def load_data_str_textzoom(batch_size):
    dataset = TextZoomSTRDataset()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    )
    while True:
        yield from loader



def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def random_text():
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_lengths = 7
    length = random.randint(1, max_lengths-1)
    text = ""
    for l in range(length):
        char_idx = random.randint(0, len(alphabet)-1)
        char = alphabet[char_idx]
        text += char
    return text


class RandomEnglishWords(Dataset):
    def __init__(self, max_word_length):
        super().__init__()

        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.lexicon = np.load('./lexicon/words_1_20.npy', allow_pickle=True)
        self.max_word_length = max_word_length

    def gen_text(self):
        mix_lengths = 2
        max_lengths = self.max_word_length
        length = random.randint(mix_lengths, max_lengths)
        p_digit = 0.1
        if random.random() < p_digit:
            word = self.text_from_digit(length)
        else:
            word = self.text_from_lexicon(length)

        print(length, word)
        return word

    def text_from_digit(self, length):
        digit = '0123456789'
        text = ""
        for l in range(length):
            char_idx = random.randint(0, len(digit)-1)
            char = digit[char_idx]
            text += char
        return text

    def text_from_lexicon(self, length):
        
        words_list = self.lexicon[length-1]

        is_valid = False
        while True:
            word = random.choice(words_list)
            if sum([1 for ch in word if ch not in self.alphabet]) == 0:
                is_valid = True
            if is_valid:
                break
        
        return word



    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, i):

        gt_text_label = self.gen_text()

        out_dict = {}
        out_dict["gt_text_label"] = gt_text_label
        return None, out_dict

    def get_imgarr(self, nSamples):
        gt_text_label_list = []
        for i in range(nSamples):
            _, out_dict = self.__getitem__(i)
            gt_text_label = out_dict["gt_text_label"]
            gt_text_label_list.append(gt_text_label)

        return gt_text_label_list

class TextZoomSTRDataset(Dataset):
    def __init__(self):
        super().__init__()

        # ==== TextZoom Dataset ====
        TZdataset = TextZoomDataset()
        tz_hr_samples_list = []
        tz_all_word_list = []
        print('Loading TextZoom...')
        for i in range(len(TZdataset)):
            img_HR, out_dict = TZdataset[i]
            label_str = out_dict['gt_text_label']

            img_HR = torch.from_numpy(img_HR)
            img_HR = img_HR.unsqueeze(0)
            tz_hr_samples_list.append(img_HR)
            tz_all_word_list.append(label_str)

        tz_all_hr_samples = torch.cat(tz_hr_samples_list, dim=0)

        # ==== STR Dataset ====
        print('Loading STR dataset...')
        img_filenames = sorted(glob.glob(os.path.join('./dataset/STR/img', '*.npz')))
        words_filenames = sorted(glob.glob(os.path.join('./dataset/STR/word', '*.txt')))
        str_samples_list = []
        str_word_list = []
        for img_filename, words_filename in zip(img_filenames, words_filenames):

            samples = np.load(img_filename)
            samples = samples["arr_0"]
            samples = torch.from_numpy(samples).float()
            samples = samples / 127.5 - 1

            word_list = []
            with open(words_filename, 'r') as f:
                line = f.readline()
                while line:
                    line = line[:-1]
                    word_list.append(line)
                    line = f.readline()

            assert samples.shape[0] == len(word_list)

            str_samples_list.append(samples)
            str_word_list.extend(word_list)
        
        str_samples = torch.cat(str_samples_list, dim=0)

        all_samples = torch.cat([tz_all_hr_samples, str_samples], dim=0)
        all_word_list = tz_all_word_list + str_word_list
        print(all_samples.shape, len(all_word_list))

        self.nSamples = all_samples.shape[0]
        self.samples = all_samples
        self.word_list = all_word_list

        print('Finish loading dataset.')

    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):

        sample = self.samples[idx]
        word = self.word_list[idx]

        out_dict = {}
        out_dict["low_res"] = sample
        out_dict["gt_text_label"] = word
        return sample, out_dict




class TextZoomDataset(Dataset):
    def __init__(self):
        super().__init__()

        root_train1 = os.path.join("./dataset/TextZoom", "train1")
        env1 = lmdb.open(root_train1, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn1 = env1.begin(write=False)
        self.nSamples1 = int(self.txn1.get(b'num-samples'))

        root_train2 = os.path.join("./dataset/TextZoom", "train2")
        env2 = lmdb.open(root_train2, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn2 = env2.begin(write=False)
        self.nSamples2 = int(self.txn2.get(b'num-samples'))

    def buf2PIL(self, txn, key, type='RGB'):
        imgbuf = txn.get(key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im


    def __len__(self):
        return self.nSamples1 + self.nSamples2
    
    def __getitem__(self, idx):
        # while idx == 0 or idx == self.nSamples1:
            # idx = random.randint(0, self.nSamples1 + self.nSamples2 -1)
        idx += 1
        if idx <= self.nSamples1:
            local_idx = idx
            txn = self.txn1
        elif idx <= self.nSamples1 + self.nSamples2:
            local_idx = idx - self.nSamples1
            txn = self.txn2
        else:
            raise RuntimeError("invalid idx was input", idx)

        label_key = b'label-%09d' % local_idx
        img_HR_key = b'image_hr-%09d' % local_idx
        img_lr_key = b'image_lr-%09d' % local_idx

        img_HR = self.buf2PIL(txn, img_HR_key, 'RGB')
        img_lr = self.buf2PIL(txn, img_lr_key, 'RGB')
        # print(img_HR.size, img_lr.size)
        img_HR = img_HR.resize((128, 32), Image.BICUBIC)
        img_lr = img_lr.resize((128, 32), Image.BICUBIC)
        word = txn.get(label_key)
        word = str(word.decode())

        img_HR_arr = np.array(img_HR).astype(np.float32)
        img_lr_arr = np.array(img_lr).astype(np.float32)

        img_HR_arr = img_HR_arr / 127.5 - 1
        img_lr_arr = img_lr_arr / 127.5 - 1

        img_HR_arr = np.transpose(img_HR_arr, [2, 0, 1])
        img_lr_arr = np.transpose(img_lr_arr, [2, 0, 1])

        out_dict = {}
        out_dict["low_res"] = img_lr_arr
        out_dict["gt_text_label"] = word
        return img_HR_arr, out_dict


class TextZoomDataset_Deg(Dataset):
    def __init__(self):
        super().__init__()

        root_train1 = os.path.join("./dataset/TextZoom", "train1")
        env1 = lmdb.open(root_train1, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn1 = env1.begin(write=False)
        self.nSamples1 = int(self.txn1.get(b'num-samples'))

        root_train2 = os.path.join("./dataset/TextZoom", "train2")
        env2 = lmdb.open(root_train2, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn2 = env2.begin(write=False)
        self.nSamples2 = int(self.txn2.get(b'num-samples'))

    def buf2PIL(self, txn, key, type='RGB'):
        imgbuf = txn.get(key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im


    def __len__(self):
        return self.nSamples1 + self.nSamples2
    
    def __getitem__(self, idx):

        idx += 1
        if idx <= self.nSamples1:
            local_idx = idx
            txn = self.txn1
        elif idx <= self.nSamples1 + self.nSamples2:
            local_idx = idx - self.nSamples1
            txn = self.txn2
        else:
            raise RuntimeError("invalid idx was input", idx)

        label_key = b'label-%09d' % local_idx
        img_HR_key = b'image_hr-%09d' % local_idx
        img_lr_key = b'image_lr-%09d' % local_idx

        img_HR = self.buf2PIL(txn, img_HR_key, 'RGB')
        img_lr = self.buf2PIL(txn, img_lr_key, 'RGB')
        # print(img_HR.size, img_lr.size)
        img_HR = img_HR.resize((128, 32), Image.BICUBIC)
        img_lr = img_lr.resize((128, 32), Image.BICUBIC)
        word = txn.get(label_key)
        word = str(word.decode())

        img_HR_arr = np.array(img_HR).astype(np.float32)
        img_lr_arr = np.array(img_lr).astype(np.float32)

        img_HR_arr = img_HR_arr / 127.5 - 1
        img_lr_arr = img_lr_arr / 127.5 - 1

        img_HR_arr = np.transpose(img_HR_arr, [2, 0, 1])
        img_lr_arr = np.transpose(img_lr_arr, [2, 0, 1])

        out_dict = {}
        out_dict["low_res"] = img_HR_arr
        out_dict["gt_text_label"] = word
        return img_lr_arr, out_dict


class TextZoomDataset_SR_Test(Dataset):
    def __init__(self, level):
        super().__init__()

        root = os.path.join("./dataset/TextZoom", "test", level)
        env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = env.begin(write=False)
        self.nSamples = int(self.txn.get(b'num-samples'))

    def buf2PIL(self, txn, key, type='RGB'):
        imgbuf = txn.get(key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im


    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, idx):
        idx += 1
        label_key = b'label-%09d' % idx
        img_HR_key = b'image_hr-%09d' % idx
        img_lr_key = b'image_lr-%09d' % idx

        img_HR = self.buf2PIL(self.txn, img_HR_key, 'RGB')
        img_lr = self.buf2PIL(self.txn, img_lr_key, 'RGB')

        img_HR = img_HR.resize((128, 32), Image.BICUBIC)
        img_lr = img_lr.resize((128, 32), Image.BICUBIC)
        word = self.txn.get(label_key)
        word = str(word.decode())

        img_HR_arr = np.array(img_HR).astype(np.float32)
        img_lr_arr = np.array(img_lr).astype(np.float32)

        img_HR_arr = img_HR_arr / 127.5 - 1
        img_lr_arr = img_lr_arr / 127.5 - 1

        img_HR_arr = np.transpose(img_HR_arr, [2, 0, 1])
        img_lr_arr = np.transpose(img_lr_arr, [2, 0, 1])

        out_dict = {}
        out_dict["low_res"] = img_lr_arr
        out_dict["gt_text_label"] = word
        return img_HR_arr, out_dict

    def get_imgarr(self):
        imgarr = []
        gt_text_label_list = []
        for i in range(self.nSamples):
            _, out_dict = self.__getitem__(i)
            lr_img = out_dict["low_res"]
            gt_text_label = out_dict["gt_text_label"]
            imgarr.append(lr_img)
            gt_text_label_list.append(gt_text_label)
        imgarr = np.array(imgarr)

        return imgarr, gt_text_label_list

class TextZoomDataset_Rec_Test(Dataset):
    def __init__(self, textzoom_root, textzoom_sr_root):
        super().__init__()

        env = lmdb.open(textzoom_root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = env.begin(write=False)
        self.nSamples = int(self.txn.get(b'num-samples'))

        self.sr_imgs = np.load(textzoom_sr_root)["arr_0"]

    def buf2PIL(self, txn, key, type='RGB'):
        imgbuf = txn.get(key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        im = Image.open(buf).convert(type)
        return im


    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, idx):
        idx += 1
        label_key = b'label-%09d' % idx
        img_HR_key = b'image_hr-%09d' % idx
        img_lr_key = b'image_lr-%09d' % idx

        img_HR = self.buf2PIL(self.txn, img_HR_key, 'RGB')
        img_lr = self.buf2PIL(self.txn, img_lr_key, 'RGB')

        img_HR = img_HR.resize((128, 32), Image.BICUBIC)
        img_lr = img_lr.resize((128, 32), Image.BICUBIC)
        word = self.txn.get(label_key)
        word = str(word.decode())

        img_HR_arr = np.array(img_HR).astype(np.float32)
        img_lr_arr = np.array(img_lr).astype(np.float32)

        img_HR_arr = img_HR_arr / 255
        img_lr_arr = img_lr_arr / 255

        img_HR_arr = np.transpose(img_HR_arr, [2, 0, 1])
        img_lr_arr = np.transpose(img_lr_arr, [2, 0, 1])

        img_SR_arr = self.sr_imgs[idx-1]
        img_SR_arr = img_SR_arr.astype(np.float32)
        img_SR_arr = img_SR_arr / 255
        img_SR_arr = np.transpose(img_SR_arr, [2, 0, 1])

        return img_HR_arr, img_lr_arr, img_SR_arr, word


if __name__ == '__main__':

    # tz_root = '../../dataset/TextZoom/test/easy'
    # tz_sr_root = '../diff_samples/textzoom/easy_sr_samples_gt_dimss.npz'

    dataset = TextZoomSTRDataset()
    img, word = dataset[10]
    print(img.shape)
    print(word)