import torch
import numpy as np
import os
import glob

from text_recognition.recognizer_init import *
from text_recognition.utils import *

def get_eval_string(word, alphabet='0123456789abcdefghijklmnopqrstuvwxyz'):
    new_word = [ch for ch in word if ch in alphabet]
    new_word = ''.join(new_word)
    return new_word

if __name__ == "__main__":

    root = './diff_samples/mr_samples'
    n_files = len(glob.glob(os.path.join(root, '*npz')))

    img_template = os.path.join(root, '1000_samples_{}.npz')
    word_template = os.path.join(root, 'words_list_{}.txt')

    print('n_files:', n_files)

    text_rec, aster_info = Aster_init("./text_recognition/ckpt/aster_demo.pth.tar")
    text_rec.cuda()
    text_rec.eval()

    for i in range(n_files):
        img_filename = img_template.format(i+1)
        words_list_filename = word_template.format(i+1)

        word_list = []
        with open(words_list_filename, 'r') as f:
            line = f.readline()
            while line:
                line = line[:-1]
                word_list.append(line)
                line = f.readline()

        # ====  check text recognition acc of text images ====
        samples = np.load(img_filename)
        samples = samples["arr_0"]
        n_samples = samples.shape[0]
        samples_ = torch.from_numpy(samples).float()

        samples_ = samples_ / 255
        samples_ = samples_.permute(0, 3, 1, 2)
        samples_ = samples_.cuda()

        pred = text_rec(samples_*2 - 1)
        pred_words = get_string_aster(pred, aster_info)
        is_correct_list = []
        for pred_word, gt_word in zip(pred_words, word_list):
            if pred_word == gt_word.lower():
                is_correct_list.append(True)
            else:
                is_correct_list.append(False)
    
        print('valid rate: {}'.format(np.mean(is_correct_list)))

        valid_samples_list = []
        valid_word_list = []
        for j in range(n_samples):
            if is_correct_list[j]:

                valid_samples_list.append(samples[j][np.newaxis])
                valid_word_list.append(word_list[j])
        
        valid_samples = np.concatenate(valid_samples_list, axis=0)
        print(valid_samples.shape, len(valid_word_list))

        out_img_filename = img_filename.replace('mr_samples', 'mr_samples/postprocessed')
        out_words_list_filename = words_list_filename.replace('mr_samples', 'mr_samples/postprocessed')

        print(out_img_filename, out_words_list_filename)

        np.savez(out_img_filename, valid_samples)
        with open(out_words_list_filename, 'w') as f:
            for word in valid_word_list:
                f.write(word)
                f.write('\n')

