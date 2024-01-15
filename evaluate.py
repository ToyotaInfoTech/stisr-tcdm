from tqdm import tqdm
import os

from text_recognition.recognizer_init import *
from text_recognition.utils import *
from guided_diffusion.image_datasets import TextZoomDataset_Rec_Test





def get_test_data_w_sr(dir_, sr_dir, batch_size):

    test_dataset = TextZoomDataset_Rec_Test(dir_, sr_dir)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=8,
        drop_last=False)
    return test_dataset, test_loader


def main(tz_root, tz_sr_dir, batch_size=64):

    _, loader = get_test_data_w_sr(tz_root, tz_sr_dir, batch_size)

    crnn = CRNN_init('./text_recognition/ckpt/crnn.pth')
    aster, aster_info = Aster_init("./text_recognition/ckpt/aster_demo.pth.tar")
    moran = MORAN_init('./text_recognition/ckpt/demo.pth')

    crnn = crnn.cuda()
    aster = aster.cuda()
    moran = moran.cuda()

    return eval_w_sr(loader, crnn, aster, aster_info, moran)


def eval_w_sr(val_loader, crnn, aster, aster_info, moran):
    # get_string = get_string_crnn if cfg.VAL.test_recognizer == 'RCNN' else get_string_aster

    crnn.eval()
    aster.eval()
    moran.eval()
    n_correct = 0
    n_correct_lr = 0
    n_correct_hr = 0
    n_correct_as = 0
    n_correct_lr_as = 0
    n_correct_hr_as = 0
    n_correct_mo = 0
    n_correct_lr_mo = 0
    n_correct_hr_mo = 0
    sum_images = 0

    cal_ssim = SSIM()
    cal_psnr = calculate_psnr

    ssim_list = list()
    psnr_list = list()

    for _, data in tqdm((enumerate(val_loader))):
        images_hr, images_lr, images_sr, label_strs = data
        batch_size = images_lr.shape[0]

        images_lr = images_lr.cuda()
        images_hr = images_hr.cuda()
        images_sr = images_sr.cuda()

        test_output_lr, _ = crnn(images_lr[:, :3, :, :])
        test_output_hr, _ = crnn(images_hr[:, :3, :, :])
        test_output_sr, _ = crnn(images_sr[:, :3, :, :])

        test_output_lr_aster = aster(images_lr[:, :3, :, :]*2-1)
        test_output_hr_aster = aster(images_hr[:, :3, :, :]*2-1)
        test_output_sr_aster = aster(images_sr[:, :3, :, :]*2-1)


        x, length, text, text_rev, converter_moran = parse_moran_data(images_sr)
        test_output_sr_moran, _ = moran(x, length, text, text_rev, test=True)
        _, preds = test_output_sr_moran.max(1)
        preds_ = preds.clone()
        sim_preds = converter_moran.decode(preds.data, length.data)
        predict_result_sr_moran = [pred.split('$')[0] for pred in sim_preds]

        x, length, text, text_rev, converter_moran = parse_moran_data(images_lr)
        test_output_lr_moran, _ = moran(x, length, text, text_rev, test=True)
        _, preds = test_output_lr_moran.max(1)
        sim_preds = converter_moran.decode(preds.data, length.data)
        predict_result_lr_moran = [pred.split('$')[0] for pred in sim_preds]

        x, length, text, text_rev, converter_moran = parse_moran_data(images_hr)
        test_output_hr_moran, _ = moran(x, length, text, text_rev, test=True)
        _, preds = test_output_hr_moran.max(1)
        sim_preds = converter_moran.decode(preds.data, length.data)
        predict_result_hr_moran = [pred.split('$')[0] for pred in sim_preds]



        for b in range(batch_size):
            ssim_list.append(cal_ssim(images_sr[b][None], images_hr[b][None]).item())
            psnr_list.append(cal_psnr(images_sr[b][None], images_hr[b][None]).item())
        
        predict_result_lr = get_string_crnn(test_output_lr)
        predict_result_hr = get_string_crnn(test_output_hr)
        predict_result_sr = get_string_crnn(test_output_sr)

        predict_result_lr_aster = get_string_aster(test_output_lr_aster, aster_info)
        predict_result_hr_aster = get_string_aster(test_output_hr_aster, aster_info)
        predict_result_sr_aster = get_string_aster(test_output_sr_aster, aster_info)


        filter_mode = 'lower'
        for b in range(batch_size):
            label = label_strs[b]
            # print(predict_result_sr_moran[b], predict_result_lr_moran[b], predict_result_hr_moran[b], label)
            if str_filt(predict_result_sr[b], filter_mode) == str_filt(label, filter_mode):
                n_correct += 1

            if str_filt(predict_result_lr[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_lr += 1

            if str_filt(predict_result_hr[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_hr += 1

            if str_filt(predict_result_sr_aster[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_as += 1

            if str_filt(predict_result_lr_aster[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_lr_as += 1

            if str_filt(predict_result_hr_aster[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_hr_as += 1

            if str_filt(predict_result_sr_moran[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_mo += 1

            if str_filt(predict_result_lr_moran[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_lr_mo += 1

            if str_filt(predict_result_hr_moran[b], filter_mode) == str_filt(label, filter_mode):
                n_correct_hr_mo += 1

        sum_images += batch_size
        torch.cuda.empty_cache()

    accuracy = round(n_correct / sum_images, 4)
    accuracy_lr = round(n_correct_lr / sum_images, 4)
    accuracy_hr = round(n_correct_hr / sum_images, 4)

    accuracy_as = round(n_correct_as / sum_images, 4)
    accuracy_lr_as = round(n_correct_lr_as / sum_images, 4)
    accuracy_hr_as = round(n_correct_hr_as / sum_images, 4)

    accuracy_mo = round(n_correct_mo / sum_images, 4)
    accuracy_lr_mo = round(n_correct_lr_mo / sum_images, 4)
    accuracy_hr_mo = round(n_correct_hr_mo / sum_images, 4)

    ssim = round(sum(ssim_list) / sum_images, 6)
    psnr = round(sum(psnr_list) / sum_images, 6)

    print("===== Evaluation results =====")
    print('sr_accuray: %.2f%%' % (accuracy * 100))
    print('lr_accuray: %.2f%%' % (accuracy_lr * 100))
    print('hr_accuray: %.2f%%' % (accuracy_hr * 100))

    print('sr_accuray aster: %.2f%%' % (accuracy_as * 100))
    print('lr_accuray aster: %.2f%%' % (accuracy_lr_as * 100))
    print('hr_accuray aster: %.2f%%' % (accuracy_hr_as * 100))

    print('sr_accuray moran: %.2f%%' % (accuracy_mo * 100))
    print('lr_accuray moran: %.2f%%' % (accuracy_lr_mo * 100))
    print('hr_accuray moran: %.2f%%' % (accuracy_hr_mo * 100))

    print('ssim: %.4f%%' % (ssim))
    print('psnr: %.4f%%' % (psnr))
    print("=====================================================")

    return {
        'sr_accuray': accuracy * 100,
        'lr_accuray': accuracy_lr * 100,
        'hr_accuray': accuracy_hr * 100,
        'ssim': ssim,
        'psnr': psnr
    }


if __name__ == '__main__':
    samples_root = './diff_samples/textzoom'
    dataset_root = '../dataset/TextZoom/test'

    # steps_n = 10
    # difficulties_list = ['easy', 'medium', 'hard']
    # n_samples_list = ['1619', '1411', '1343']


    # filename = "easy_1619_100000.npz"
    filename = "easy_sr_samples_gt_dimss.npz"

    samples_dir = os.path.join(samples_root, filename)
    data_dir = os.path.join(dataset_root, 'easy')
    results = main(data_dir, samples_dir)




