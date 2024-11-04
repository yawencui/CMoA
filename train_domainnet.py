import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import sys
import copy
import argparse
from PIL import Image
import torch
import utils_pytorch
from utils_incremental.compute_features_vit_adapter import compute_features
from utils_incremental.compute_accuracy_vit_adapter import compute_accuracy
from utils_incremental.incremental_train_and_eval_iteration_class_balance_vit_adapter_cs_loss_adaptive_3 import incremental_train_and_eval
from resnet import resnet18
from ViT import ViT
from models.vit_convpass_cs_adaptive_3.network import build_net
import pickle
import os
import random
from dataloader import BaseDataset
from torchvision import models
from ewc_vit import EWC
import logging
#torch.cuda.current_device()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/project_2006532/dataset', type=str)
parser.add_argument('--dataset', default='DomainNet', type=str)
parser.add_argument('--num_classes', default=80, type=int)
parser.add_argument('--nb_cl_fg', default=60, type=int,
                    help='the number of classes in first session')
parser.add_argument('--nb_cl', default=5, type=int,
                    help='Classes per group')
parser.add_argument('--nb_protos', default=5, type=int,
                    help='Number of prototypes per class at the end')
parser.add_argument('--k_shot', default=5, type=int,
                    help='')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str,
                    help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int,
                    help='Epochs for first sesssion')
parser.add_argument('--T', default=2, type=float,
                    help='Temperature for distialltion')
parser.add_argument('--beta', default=0.25, type=float,
                    help='Beta for distialltion')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from checkpoint')
parser.add_argument('--rs_ratio', default=0.0, type=float,
                    help='The ratio for resample')

parser.add_argument('--unlabeled_start_epoch', default=80, type=int,
                    help='the epoch that start to add unlabeled data')
parser.add_argument('--unlabeled_iteration', default=100, type=int,
                    help='the total iteration to add unlabeled data')
parser.add_argument('--update_unlabeled', action='store_true', default=True,
                    help='if using selected unlabled data to update the class_mean')
parser.add_argument('--use_nearest_mean', action='store_true', default=True,
                    help='if using nearest-mean-of-examplars classification for selecting unlabeled data')
parser.add_argument('--unlabeled_num', default=0, type=int,
                    help='The total number for resample')
parser.add_argument('--unlabeled_num_selected', default=75, type=int,
                    help='The number of selected unlabeled data (25, 50, 75, 100)')
parser.add_argument('--random_seed', default=1993, type=int,
                    help='random seed')

parser.add_argument('--method', default='self_train', type=str,
                    choices=['self_train', 'random'],
                    help='the method for adding unlabeled data')
parser.add_argument('--ups', action='store_true', default=False,
                    help='if use UPS to select unlabeled data')

parser.add_argument('--ups_param', default=100, type=int,
                    help='the range for selecting unlabeled data in UPS')

parser.add_argument('--prob_num', default=10, type=int,
                    help='the number of computing prob')

parser.add_argument('--ups_selection', default=4, type=int,
                    help='1 for sorting, 2 for threshold, 3 for only 10 times prob, 4 for choose the best results from prob and ups')

parser.add_argument('--double_selection', action='store_true', default=False,
                    help='if use double selection for unlabeled data')

parser.add_argument('--distillation', action='store_true', default=False,
                    help='if distillation')

parser.add_argument('--protocol', default=4, type=int,
                    help='use which protocol')

parser.add_argument('--freeze_backbone', action='store_true', default=False,
                    help='if freeze backbone')

parser.add_argument('--lwf', action='store_true', default=False,
                    help='if use lwf loss')

parser.add_argument('--ewc', action='store_true', default=False,
                    help='if ewc loss')

parser.add_argument('--contrastive_learning', action='store_true', default=False,
                    help='if contrastive learning')

parser.add_argument('--cl_start', default=3, type=int,
                    help='contrastive_learning starting session')

parser.add_argument('--cl_weight', default=1.0, type=float,
                    help='contrastive learning weight')

parser.add_argument('--sh', default=0, type=int,
                    help='sh number')

parser.add_argument('--adapter_num', default=3, type=int,
                    help='')

parser.add_argument('--dim', default=8, type=int,
                    help='')

parser.add_argument('--cosine_similarity', action='store_true', default=False,
                    help='if computing cosine_similarity loss')

parser.add_argument('--loose_cs', action='store_true', default=False,
                    help='if loose cosine_similarity loss')

parser.add_argument('--cs_para', default=0.2, type=float,
                    help='Beta for distialltion')

parser.add_argument('--cs_start', default=2, type=int,
                    help='cosine_similarity starting session')

args = parser.parse_args()
assert (args.nb_cl_fg % args.nb_cl == 0)
assert (args.nb_cl_fg >= args.nb_cl)
train_batch_size = 64 # Batch size for train 32
test_batch_size = 64  # Batch size for test (original 100) 50
eval_batch_size = 64  # Batch size for eval 32
base_lr = 1e-3 # Initial learning rate
lr_strat = [80, 120]  # Epochs where learning rate gets decreased
lr_factor = 0.1 # Learning rate decrease factor
custom_weight_decay = 1e-4  # Weight Decay
custom_momentum = 0.9  # Momentum
args.ckp_prefix = '{}_nb_cl_fg_{}_nb_cl_{}_nb_protos_{}'.format(args.ckp_prefix, args.nb_cl_fg, args.nb_cl,
                                                                args.nb_protos)
np.random.seed(args.random_seed)  # Fix the random seed
print(args)
#device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.cuda.current_device()
dictionary_size = 30

label2id = utils_pytorch.get_label2id("/scratch/project_2006532/dataset/DomainNet/split_{}/label_name.txt".format(args.protocol))

trainset_data, trainset_targets = utils_pytorch.get_data_file_domainnet("/scratch/project_2006532/dataset/DomainNet/split_{}/train.txt".format(args.protocol),
                                                              "/scratch/project_2006532/dataset/DomainNet/",
                                                              label2id)

X_train_total = np.array(trainset_data)
Y_train_total = np.array(trainset_targets)

order_name = "/scratch/project_2006532/checkpoint/seed_{}_{}_order_run.pkl".format(args.random_seed, args.dataset)
id2label = {index: la for la, index in label2id.items()}
print("Order name:{}".format(order_name))
order = np.array([i for i in id2label.keys()])
print(order)
# np.random.shuffle(order)
order_list = list(order)
print(order_list)

X_valid_cumuls = []
X_protoset_cumuls = []
X_train_cumuls = []
Y_valid_cumuls = []
Y_protoset_cumuls = []
Y_train_cumuls = []

X_valid_cumuls_base = []
Y_valid_cumuls_base = []
X_valid_cumuls_novel = []
Y_valid_cumuls_novel = []

# alpha_dr_herding = np.zeros((int(args.num_classes / args.nb_cl), dictionary_size, args.nb_cl), np.float32)

# The following contains all the training samples of the different classes
# because we want to compare our method with the theoretical case where all the training samples are stored
prototypes = [[] for i in range(args.num_classes)]
for orde in range(args.num_classes):
    prototypes[orde] = X_train_total[np.where(Y_train_total == order[orde])]

# prototypes = np.array(prototypes)

start_session = int(args.nb_cl_fg / args.nb_cl) - 1

alpha_dr_herding = []
#for i in range(int(args.num_classes / args.nb_cl)):
#    if i > start_session:
#        alpha_dr_herding.append(np.zeros((args.nb_cl, args.k_shot), np.float32))
#    else:
#        alpha_dr_herding.append(np.zeros((args.nb_cl, dictionary_size), np.float32))

classes_current = None
classes_base = None

ewc_regularizer = EWC(ewc_lambda=1.0, if_online=False)

class_means_current = []

class_means = np.zeros((768, 80, 2))

if not os.path.isdir('result_log'):
        os.mkdir('result_log')
output_dir = '/scratch/project_2006532/domainnet_result_log'
log_file = 'train_{}.log'.format(args.sh)


logging.basicConfig(level=logging.INFO,
                    # format="%(asctime)s - %(levelname)s - %(filename)s-%(funcName)s-%(lineno)d:%(message)s",
                    # datefmt='%a-%d %b %Y %H:%M:%S',
                    handlers=[logging.FileHandler(os.path.join(output_dir, log_file), 'a', 'utf-8'),
                                logging.StreamHandler()]
                    )

for session in range(start_session, 20):

    #------traning and testing data in the current session------
    train_file = os.path.join(args.data_dir, args.dataset, "split_{}".format(args.protocol), "session_{}.txt".format(session - 10))
    test_file = os.path.join(args.data_dir, args.dataset, "split_{}".format(args.protocol), "test_{}.txt".format(session - 10))

    unlabeled_data = None
    unlabeled_gt = None
    if session > start_session:
        if session == start_session + 1:
            args.epochs = 100
        else:
            args.epochs = 100
        base_lr = 0.0005
        print('the learning rate is {}'.format(base_lr))

    X_train, Y_train = utils_pytorch.get_data_file_domainnet(train_file, "/scratch/project_2006532/dataset/DomainNet/", label2id)
    X_valid,  Y_valid = utils_pytorch.get_data_file_domainnet(test_file, "/scratch/project_2006532/dataset/DomainNet/", label2id)

    change_fc_flag = False
    if session == start_session:
        classes_base = np.unique(Y_train)
        classes_current = np.unique(Y_train)
        num_cls_previous = 0
        num_cls_current = len(classes_current)
        if args.cs_start == 1:
            change_fc_flag = True
    else:
        num_cls_previous = len(np.unique(classes_current))
        classes_current = np.concatenate((classes_current, np.unique(Y_train)))
        classes_current = np.unique(classes_current)
        num_cls_current = len(classes_current)
        if num_cls_current > num_cls_previous:
            change_fc_flag = True
    
    X_valid_cumuls.append(X_valid)
    X_train_cumuls.append(X_train)
    X_valid_cumul = np.concatenate(X_valid_cumuls)
    X_train_cumul = np.concatenate(X_train_cumuls)

    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul = np.concatenate(Y_valid_cumuls)
    Y_train_cumul = np.concatenate(Y_train_cumuls)

    if session == start_session:
        X_valid_ori = X_valid
        Y_valid_ori = Y_valid

        X_valid_cumuls_base = X_valid
        Y_valid_cumuls_base = Y_valid
    else:
        X_protoset = np.concatenate(X_protoset_cumuls)
        Y_protoset = np.concatenate(Y_protoset_cumuls)
        if args.rs_ratio > 0:
            # 1/rs_ratio = (len(X_train)+len(X_protoset)*scale_factor)/(len(X_protoset)*scale_factor)
            scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
            rs_sample_weights = np.concatenate((np.ones(len(X_train)), np.ones(len(X_protoset)) * scale_factor))
            # number of samples per epoch, undersample on the new classes
            # rs_num_samples = len(X_train) + len(X_protoset)
            rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
            print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset), rs_num_samples))
        if args.distillation:
            X_train = np.concatenate((X_train, X_protoset), axis=0)
            Y_train = np.concatenate((Y_train, Y_protoset))

        X_valid_cumuls_novel.append(X_valid)
        Y_valid_cumuls_novel.append(Y_valid)
        X_valid_cumul_novel = np.concatenate(X_valid_cumuls_novel)
        Y_valid_cumul_novel = np.concatenate(Y_valid_cumuls_novel)
   
    print('Batch of classes number {0} arrives ...'.format(session))
    logging.info("Batch of classes number {0} arrives ...".format(session))

    ############################################################

    trainset = BaseDataset("train", 224, label2id)
    trainset.data = X_train
    trainset.targets = Y_train

    #------model------
    if session == start_session:
        #args.rs_ratio = 0.2
        ############################################################
        last_iter = 0
        ############################################################
        if args.resume:
            print('resume the results of first session')
            ckp_name = '/scratch/project_2006532/checkpoint/{}_epochs_{}_iteration_{}_model.pth'.format(args.dataset, args.epochs, session)
            tg_model = torch.load(ckp_name)
            ref_model = None
            args.epochs = 0
        else:
            # tg_model = resnet18(num_classes=args.nb_cl_fg, pretrained=False)
            kwargs = {
            'conv_type': 'conv',
            'num_classes': 60
            }
            tg_model = build_net(arch_name='vit_base_patch16_224', pretrained=True, dim=args.dim, adapter_num=args.adapter_num, **kwargs)
            ref_model = None
    else:
        #args.rs_ratio = 0.99
        last_iter = session
        ############################################################
        ref_model = copy.deepcopy(tg_model)
        # increment classes
        if change_fc_flag:
            in_features = tg_model.head.in_features
            out_features = tg_model.head.out_features
            new_head = nn.Linear(in_features, out_features + args.nb_cl)
            new_head.weight.data[:out_features] = tg_model.head.weight.data
            new_head.bias.data[:out_features] = tg_model.head.bias.data
            tg_model.head = new_head
    
    if args.freeze_backbone:
        for name, p in tg_model.named_parameters():      
            if 'adapter' in name or 'head' in name:
                p.requires_grad = True
                # import pdb; pdb.set_trace()
            else:
                p.requires_grad = False

    if session > start_session and args.rs_ratio > 0 and scale_factor > 1:

        index1 = np.where(rs_sample_weights > 1)[0]
        index2 = np.where(Y_train < session * args.nb_cl)[0]
        assert ((index1 == index2).all())
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=False, sampler=train_sampler, num_workers=4)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                  shuffle=True, num_workers=4)
    testset = BaseDataset("test", 224, label2id)
    testset.data = X_valid_cumul
    testset.targets = Y_valid_cumul
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=4)
    print('Max and Min of train labels: {}, {}'.format(min(Y_train), max(Y_train)))
    print('Max and Min of valid labels: {}, {}'.format(min(Y_valid_cumul), max(Y_valid_cumul)))
    logging.info("Max and Min of train labels: {}, {}".format(min(Y_train), max(Y_train)))
    logging.info("Max and Min of valid labels: {}, {}".format(min(Y_valid_cumul), max(Y_valid_cumul)))

    ##############################################################
    ckp_name = '/scratch/project_2006532/checkpoint/{}_iteration_{}_model.pth'.format(args.ckp_prefix, session)
    print('ckp_name', ckp_name)

    tg_params = tg_model.parameters()
    tg_model = tg_model.to(device)
    if session > start_session:
        ref_model = ref_model.to(device)
        #base_lr = 0.01
        print('the learning rate is {}'.format(base_lr))

    tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
    tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
    tg_model = incremental_train_and_eval(args.cl_start, args.cl_weight, args.cs_para, args.loose_cs, args.adapter_num, args.cosine_similarity, args.contrastive_learning, change_fc_flag, class_means_current, args.ewc, ewc_regularizer, args.lwf, args.distillation, prototypes, args.update_unlabeled, args.epochs, args.method, args.prob_num, args.ups, args.ups_param, args.unlabeled_num, args.unlabeled_iteration, args.unlabeled_num_selected, train_batch_size, args.double_selection, args.ups_selection, tg_model, ref_model, tg_optimizer, tg_lr_scheduler,
                                          trainloader, testloader,
                                          session, start_session,
                                          args.T, args.beta, unlabeled_data, unlabeled_gt, args.nb_cl, trainset, 224,
                                          args.unlabeled_start_epoch, device=device)
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(tg_model, ckp_name)
    if args.resume == False:
        if session == start_session:
            ckp_name = '/scratch/project_2006532/checkpoint/domainnet_vit_{}_epochs_{}_iteration_{}_model.pth'.format(args.dataset, args.epochs, session)
            torch.save(tg_model, ckp_name)

    nb_protos_cl = args.nb_protos
    # tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_features = tg_model.head.in_features
    # Herding
    print('Updating exemplar set...')
    dr_herding = []
    
    begin_iter = 0
    end_iter = 0

    if session == start_session:
        begin_iter = last_iter * args.nb_cl
        end_iter = (session + 1) * args.nb_cl
    else:
        begin_iter = num_cls_previous
        end_iter = num_cls_current
    
    for iter_dico in range(begin_iter, end_iter):
        # Possible exemplars in the feature space and projected on the L2 sphere

        evalset = BaseDataset("test", 224, label2id)
        evalset.data = prototypes[iter_dico]
        evalset.targets = np.zeros(len(evalset))  # zero labels
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        num_samples = len(evalset)
        mapped_prototypes = compute_features(tg_model, evalloader, num_samples, num_features, device=device)
        D = mapped_prototypes.T
        D = D / np.linalg.norm(D, axis=0)

        '''
        # Herding procedure : ranking of the potential exemplars      
        mu = np.mean(D, axis=1)
        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        alpha_dr_herding[index1][index2] = alpha_dr_herding[index1][index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(alpha_dr_herding[index1][index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if alpha_dr_herding[index1][index2][ind_max] == 0:
                alpha_dr_herding[index1][index2][ind_max]= 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]
        '''
        herding = np.zeros(len(prototypes[iter_dico]), np.float32)
        dr_herding.append(herding)
        # Herding procedure : ranking of the potential exemplars
        mu = np.mean(D, axis=1)
        index1 = int(iter_dico / args.nb_cl)
        index2 = iter_dico % args.nb_cl
        dr_herding[index2] = dr_herding[index2] * 0
        w_t = mu
        iter_herding = 0
        iter_herding_eff = 0
        while not (np.sum(dr_herding[index2] != 0) == min(nb_protos_cl, 500)) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if dr_herding[index2][ind_max] == 0:
                dr_herding[index2][ind_max] = 1 + iter_herding
                iter_herding += 1
            w_t = w_t + mu - D[:, ind_max]

        if (iter_dico + 1) % args.nb_cl == 0:
            alpha_dr_herding.append(np.array(dr_herding))
            dr_herding = []

    X_protoset_cumuls = []
    Y_protoset_cumuls = []

    # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
    print('Computing mean-of_exemplars and theoretical mean...')
    # class_means = np.zeros((768, 85, 2))
    
    if session == start_session:
        start_iteration = 0
    elif session > start_session and args.distillation:
        start_iteration = 0
    else:
        start_iteration = num_cls_current / args.nb_cl - 1
   
    end_iteration = num_cls_current / args.nb_cl
    
    for iteration2 in range(int(start_iteration), int(end_iteration)):
        for iter_dico in range(args.nb_cl):
            current_cl = order[range(iteration2*args.nb_cl, (iteration2+1)*args.nb_cl)]

            # Collect data in the feature space for each class
            evalset = BaseDataset("test", 224, label2id)
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalset.targets = np.zeros(evalset.data.shape[0]) #zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=4)
            num_samples = evalset.data.shape[0]
            mapped_prototypes = compute_features(tg_model, evalloader, num_samples, num_features, device=device)
            D = mapped_prototypes.T
            D = D/np.linalg.norm(D,axis=0)
            # Flipped version also
            evalset.data = prototypes[iteration2*args.nb_cl+iter_dico]
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                    shuffle=False, num_workers=4)
            mapped_prototypes2 = compute_features(tg_model, evalloader, num_samples, num_features,device=device)
            D2 = mapped_prototypes2.T
            D2 = D2/np.linalg.norm(D2,axis=0)

            # iCaRL
            alph = alpha_dr_herding[iteration2][iter_dico]
            alph = (alph>0)*(alph<nb_protos_cl+1)*1.
            X_protoset_cumuls.append(prototypes[iteration2*args.nb_cl+iter_dico][np.where(alph==1)[0]])
            Y_protoset_cumuls.append(order[iteration2*args.nb_cl+iter_dico]*np.ones(len(np.where(alph==1)[0])))
            alph = alph/np.sum(alph)
            class_means[:,current_cl[iter_dico],0] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],0] /= np.linalg.norm(class_means[:,current_cl[iter_dico],0])

                # Normal NCM
            '''
            if iteration2 > start_session:
                alph = np.ones(args.k_shot) / args.k_shot
            else:
                alph = np.ones(dictionary_size) / dictionary_size
            '''
            # if iteration2 > start_session:
            alph = np.ones(len(prototypes[iteration2*args.nb_cl+iter_dico])) / len(prototypes[iteration2*args.nb_cl+iter_dico])
            # else:
            #     alph = np.ones(dictionary_size) / dictionary_size

            class_means[:,current_cl[iter_dico],1] = (np.dot(D,alph)+np.dot(D2,alph))/2
            class_means[:,current_cl[iter_dico],1] /= np.linalg.norm(class_means[:,current_cl[iter_dico],1])
    
    # torch.save(class_means, './checkpoint/{}_run_iteration_{}_class_means.pth'.format(args.ckp_prefix, session))

    current_means = class_means[:, order[range(0, num_cls_current)]]

    class_means_current = current_means[:, -5:, 0]
    
    logging.info("Computing cumulative accuracy...")
    print('Computing cumulative accuracy...')
    evalset = BaseDataset("test", 224, label2id)
    evalset.data = X_valid_cumul
    evalset.targets = Y_valid_cumul
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                shuffle=False, num_workers=4)
    cumul_acc = compute_accuracy(tg_model, current_means, evalloader, device=device)

    if session > start_session:

        print('Computing the accuracy of base classes...')
        logging.info("Computing the accuracy of base classes...")
        evalset = BaseDataset("test", 224, label2id)
        evalset.data = X_valid_cumuls_base
        evalset.targets = Y_valid_cumuls_base
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, current_means, evalloader, device=device)

        print('Computing the accuracy of novel classes...')
        logging.info("Computing the accuracy of novel classes...")
        evalset = BaseDataset("test", 224, label2id)
        evalset.data = X_valid_cumul_novel
        evalset.targets = Y_valid_cumul_novel
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                 shuffle=False, num_workers=4)
        cumul_acc = compute_accuracy(tg_model, current_means, evalloader, device=device)

    print('Computing the accuracy of unseen domain...')
    logging.info("Computing the accuracy of unseen domain...")
    unseen_test_file = os.path.join(args.data_dir, args.dataset, "painting_test.txt")
    X_unseen, Y_unseen = utils_pytorch.get_data_file_domainnet_unseen(classes_current, unseen_test_file, "/scratch/project_2006532/dataset/DomainNet/", label2id)

    unseen_set = BaseDataset("test", 224, label2id)
    unseen_set.data = X_unseen
    unseen_set.targets = Y_unseen

    unseen_loader = torch.utils.data.DataLoader(unseen_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=4)
    cumul_acc = compute_accuracy(tg_model, current_means, unseen_loader, device=device)

    X_unseen_base, Y_unseen_base = utils_pytorch.get_data_file_domainnet_unseen(classes_base, unseen_test_file, "/scratch/project_2006532/dataset/DomainNet/", label2id)

    unseen_base_set = BaseDataset("test", 224, label2id)
    unseen_base_set.data = X_unseen_base
    unseen_base_set.targets = Y_unseen_base

    unseen_base_loader = torch.utils.data.DataLoader(unseen_base_set, batch_size=test_batch_size,
                                             shuffle=False, num_workers=4)

    print('Computing the accuracy of unseen domain (base)...')
    logging.info("Computing the accuracy of unseen domain (base)...")
    cumul_acc = compute_accuracy(tg_model, current_means, unseen_base_loader, device=device)