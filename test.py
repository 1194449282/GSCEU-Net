import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import metrics
import archs
import acfnet
import albumentations as albu
from dataset import Dataset
from metrics_standard import iou_score, jaccard_socre, acc_socre, sensitivity_socre, specificity_socre, precision_socre, dice_score,iou_score1,  sensitivity_score1,  precision_score1, dice_score1,specificity_score1,acc_score1

from utils import AverageMeter
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
"""
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="final_skin2018_UNet_woDS",
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    y_true_list=[]
    y_pred_list=[]
    roc_auc=0
    labels_all = np.array([], dtype=int)
    score_all = np.array([], dtype=float)
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = acfnet.Seg_Model(config['num_classes'])
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        #                                    config['deep_supervision']
    )

    model = model.cuda()

    # Data loading code
    test_img_ids = glob(os.path.join('inputs', 'final_skin2018/test', 'images', '*' + config['img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    test_transform = Compose([
        albu.Resize(config['input_h'], config['input_w']),
        albu.Normalize(),
    ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', 'final_skin2018/test', 'images'),
        mask_dir=os.path.join('inputs', 'final_skin2018/test', 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # avg_meter = AverageMeter()
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'specificity': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'jaccard': AverageMeter(),
                  'precision': AverageMeter(),
                  'iou1': AverageMeter(),
                  'dice1': AverageMeter(),
                  'sensitivity1': AverageMeter(),
                  'precision1': AverageMeter(),
                  'specificity1': AverageMeter(),
                  'accuracy1': AverageMeter()
                  }

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()
            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                    output = model(input)

                    # 计算 AUC（Area Under the Curve）
                    # if roc_auc==0:
                    #     roc_auc = auc(fpr1, tpr1)
                    # else:
                    #     roc_auc = (auc(fpr1, tpr1)+roc_auc)/2

                    # y_true_list.append(target.cpu().numpy().flatten()>0.5)
                    # y_pred_list.append(torch.sigmoid(output).data.cpu().numpy().flatten()>0.5)
                    y_pred_list = torch.sigmoid(output).data.cpu().numpy()
                    y_true_list = target.cpu().numpy()

                    predicted_scores = y_pred_list
                    true_labels = y_true_list
                    # 根据不同的阈值计算 FPR 和 TPR
                    thresholds = np.linspace(0, 1, 1000)
                    fpr = []
                    tpr = []

                    for threshold in thresholds:
                        # 根据阈值将预测结果转换为二分类结果
                        predicted_labels = np.where(predicted_scores >= threshold, 1, 0)

                        # 计算真阳性（TP）、假阳性（FP）、真阴性（TN）和假阴性（FN）的数量
                        TP = np.sum((predicted_labels == 1) & (true_labels == 1))
                        FP = np.sum((predicted_labels == 1) & (true_labels == 0))
                        TN = np.sum((predicted_labels == 0) & (true_labels == 0))
                        FN = np.sum((predicted_labels == 0) & (true_labels == 1))

                        # 计算 FPR 和 TPR
                        current_fpr = FP / (FP + TN)
                        current_tpr = TP / (TP + FN)

                        fpr.append(current_fpr)
                        tpr.append(current_tpr)

                    # 计算 AUC（Area Under the Curve）
                    # roc_auc = np.trapz(tpr, fpr)
                    roc_auc = metrics.auc(fpr, tpr)
                    # 绘制ROC曲线
                    plt.figure()
                    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
                    # plt.plot(mean_fpr, mean_tpr, color='darkorange', label='Mean ROC curve (area = %0.2f)' % mean_roc_auc)
                    # plt.plot(fpr, tpr, color='darkorange', label='Mean ROC curve')
                    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC)')
                    plt.legend(loc="lower right")
                    plt.show()

                    iou = iou_score(output, target)
                    dice = dice_score(output, target)
                    accuracy = acc_socre(output, target)
                    sensitivity = sensitivity_socre(output, target)
                    specificity = specificity_socre(output, target)
                    jaccard = jaccard_socre(output, target)
                    precision = precision_socre(output, target)
                    iou1 = iou_score1(output, target)
                    dice1 = dice_score1(output, target)
                    sensitivity1= sensitivity_score1(output, target)
                    precision1 = precision_score1(output, target)
                    specificity1 = specificity_score1(output, target)
                    accuracy1 = acc_score1(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['sensitivity'].update(sensitivity, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['jaccard'].update(jaccard, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))
            avg_meters['iou1'].update(iou1, input.size(0))
            avg_meters['dice1'].update(dice1, input.size(0))
            avg_meters['sensitivity1'].update(sensitivity1, input.size(0))
            avg_meters['precision1'].update(precision1, input.size(0))
            avg_meters['specificity1'].update(specificity1, input.size(0))
            avg_meters['accuracy1'].update(accuracy1, input.size(0))
            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % avg_meters['iou'].avg)
    print('IoU1: %.4f' % avg_meters['iou1'].avg)
    print('dice: %.4f' % avg_meters['dice'].avg)
    print('dice1: %.4f' % avg_meters['dice1'].avg)
    print('accuracy: %.4f' % avg_meters['accuracy'].avg)
    print('accuracy1: %.4f' % avg_meters['accuracy1'].avg)
    print('sensitivity: %.4f' % avg_meters['sensitivity'].avg)
    print('sensitivity1: %.4f' % avg_meters['sensitivity1'].avg)
    print('specificity: %.4f' % avg_meters['specificity'].avg)
    print('specificity1: %.4f' % avg_meters['specificity1'].avg)
    # print('jaccard: %.4f' % avg_meters['jaccard'].avg)
    print('precision: %.4f' % avg_meters['precision'].avg)
    print('precision1: %.4f' % avg_meters['precision1'].avg)


    torch.cuda.empty_cache()



if __name__ == '__main__':
    main()
