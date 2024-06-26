#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Hefei Institute of Intelligent Machinery, Chinese Academy of Sciences All Rights Reserved.
#
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.datasets import ImageFolder
#from utils.transform import get_transform_for_test
from my_utils.transform import get_transform_for_test
from models.CTCNet import ctcnet_tiny
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

data_root = r'E:\test set root path'  # test set root path
test_weights_path = r"best pre-trained model parameters"  # best pre-trained model parameters
num_class = 4  # number of categories
class_names = ['midCTC', 'mixCTC', 'noCTC', 'upCTC']
gpu = "cuda:0"

def test(model, test_path):
    # loading test set and pre-training model parameters
    test_dir = os.path.join(data_root, 'test')
    class_list = list(os.listdir(test_dir))
    class_list.sort()
    transform_test = get_transform_for_test(mean=[0.948078, 0.93855226, 0.9332005],
                                            var=[0.14589554, 0.17054074, 0.18254866])
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True, num_workers=1)
    checkpoint = torch.load(test_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    score_list = []  # storage prediction score
    label_list = []  # storing real labels
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        score_tmp = outputs  # (batchsize, nclass)
        score_list.extend(score_tmp.detach().cpu().numpy())
        label_list.extend(labels.cpu().numpy())

    score_array = np.array(score_list)
    # converting a label to onehot form
    label_tensor = torch.tensor(label_list)
    label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
    label_onehot = torch.zeros(label_tensor.shape[0], num_class)
    label_onehot.scatter_(dim=1, index=label_tensor, value=1)
    label_onehot = np.array(label_onehot)
    print("score_array:", score_array.shape)  # (batchsize, classnum)
    print("label_onehot:", label_onehot.shape)  # torch.Size([batchsize, classnum])

    #call the sklearn library to compute the fpr and tpr corresponding to each category
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(num_class):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
    # micro
    fpr_dict["micro"], tpr_dict["micro"], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"], tpr_dict["micro"])
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    roc_auc_dict["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])

    #plotting roc curves averaged across all categories
    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle='-', linewidth=3)

    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle='-', linewidth=3)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        class_single_name=class_names[i]
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_single_name, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic(ROC) Curve to CTCNet classify')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    # loading models
    net = ctcnet_tiny()
    device = torch.device(gpu)
    net = net.to(device)
    test(net, test_weights_path)