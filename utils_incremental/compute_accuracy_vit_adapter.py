#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
# import torchvision
# from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_pytorch import *
import logging

from datetime import datetime

import torchprofile

def compute_accuracy(tg_model, class_means, evalloader, scale=None, print_info=True, device=None):
    if device is None:
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.cuda.current_device()
    tg_model.eval()

    correct = 0
    correct_icarl = 0
    correct_icarl_5 = 0
    correct_ncm = 0
    # correct_dlu = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            start_time = datetime.now()
            outputs_features, outputs_logits = tg_model(inputs)
            end_time = datetime.now()
            time_difference = end_time - start_time
            print("running time.........", time_difference)

            flops = torchprofile.profile_macs(tg_model, inputs)
            print(f'FLOPs: {flops}')

            outputs = F.softmax(outputs_logits, dim=1)
            #print(outputs.shape)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            if targets.size(0) > 1:
                outputs_feature = np.squeeze(outputs_features.data.cpu().numpy())
            else:
                outputs_feature = outputs_features.data.cpu().numpy()

            #print(outputs_feature.shape)

            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()

            # import pdb
            # pdb.set_trace()
            # top-5
            maxk = max((1,5))
            y_resize = targets.view(-1,1)
            _, pred = score_icarl.topk(maxk, 1, True, True)
            correct_icarl_5 += torch.eq(pred, y_resize).sum().float().item()

            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')
            score_ncm = torch.from_numpy((-sqd_ncm).T).to(device)
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()

            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)

            #compute score for DLU
            # sqd_dlu = cdist(class_means[:, :, 2].T, outputs_feature, 'sqeuclidean')
            #score_dlu = torch.from_numpy((-sqd_dlu).T).to(device)
            #_, predicted_dlu = score_dlu.max(1)
            #correct_dlu += predicted_dlu.eq(targets).sum().item()
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        logging.info("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        logging.info("  top 1 accuracy iCaRL            :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 5 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl_5/total))
        logging.info("  top 5 accuracy iCaRL            :\t\t{:.2f} %".format(100.*correct_icarl_5/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))
        logging.info("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))
        # print("  top 1 accuracy DLU            :\t\t{:.2f} %".format(100. * correct_dlu / total))

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]
