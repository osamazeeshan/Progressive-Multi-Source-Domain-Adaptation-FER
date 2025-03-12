# import comet_ml at the top of your file
from comet_ml import Experiment
import argparse
import os, sys

# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from torch import nn
import random
# import cv2
from scipy import stats
import time
from numpy.linalg import norm

# from thop import profile
# from ptflops import get_model_complexity_info
import re
from collections import Counter
import copy

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.common import *
from utils.distance import compute_distance_matrix
from methods.clusters import *
from utils.visualize import *

from datasets.base_dataset import BaseDataset
from datasets.dataset import TragetRestartableIterator
from models.resnet_model_modified import ResNet50Fc, ResNet18Fc
from losses.coral_loss import CORAL_loss
from losses.mmd_loss import MMD_loss
import torch.nn.functional as F

from models.resnet_fer_modified import *
from pathlib import Path

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score

from utils.tsne_variation import *
# from utils.tsne_github import *

from methods.create_prototypes import create_target_pl_dicts,generate_tar_aug_conf_pl
from torch.utils.data import TensorDataset

from utils.reproducibility import set_default_seed, set_to_deterministic, _get_current_seed
from sklearn.metrics import pairwise_distances_argmin_min

from scipy.spatial.distance import cdist

import config
import itertools

from utils.cometml import comet_init, set_comet_exp_name

# from losses.supcontrast_loss import SupConLoss


# Create an experiment with your api key
# experiment = Experiment(
#     api_key="eow2bmNwSPBKrx657Qfx43lW7",
#     project_name="source-selection-self-paced-msda",
#     workspace="osamazeeshan",
#     log_code=True,
#     disabled=False,
# )


experiment = comet_init(config.COMET_PROJECT_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transfer_loss = 'mmd'
learning_rate = 0.0001
# n_class = 7


prev_srcs_loader_dic = []

# n_class = 2 # for pain Biovid
# n_class = 77 # to train a classifier with N source subject classes

set_default_seed()
set_to_deterministic()

def calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, lamb_threshold, target_file_path, dist_measure):
    src_tar_dist_dic = {}
    for src_sub in source_list_name:
        if src_sub in eliminate_list:
            continue
        subject_list = [src_sub]
        subject_list.append(target_subject)
        print(subject_list)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting, args.pain_tar_label_path)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.MCMASTER_FULL_LABEL_PATH, subject_list, target_subject, args.n_class, args.oracle_setting)
        # srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        # srcs_file_name = 'mcmaster_list_full.txt'   # to train model for McMaster to use as a pre-trained model
        combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, args.batch_size, phase='src')
        tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_tar_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')

        # calculate Source and Target subject distance
        dist = measure_srcs_tar_dist(transfer_model, combine_srcs_loader, tar_loader, dist_measure, args.batch_size)
        src_tar_dist_dic[src_sub] = dist
        # dist_array.append(dist)
    
    src_tar_dist_dic = dict(sorted(src_tar_dist_dic.items(), key=lambda x:x[1], reverse=True))
    src_sub_normalized = normalize_data(list(src_tar_dist_dic.values())) # normalize the distance between 0 and 1 to apply threshold 
    src_tar_dist_dic = dict(itertools.islice(src_tar_dist_dic.items(), np.sum(src_sub_normalized > lamb_threshold))) # select baased on the threshold
    
    # src_tar_dist_dic = dict(itertools.islice(src_tar_dist_dic.items(), 10)) # select based on the closest topk=10
    # torch.argmax(torch.nn.functional.softmax(src_tar_dist_dic, dim=1), dim=1).cpu().numpy()
    print("Normalize: ", src_sub_normalized)
    print(src_tar_dist_dic)
    return src_tar_dist_dic

def calculate_src_sampl_tar_dist(source_list_name, target_subject, transfer_model, target_file_path, dist_measure):
    src_tar_dist_dic = {}
    # for src_sub in source_list_name:
    #     if src_sub in eliminate_list:
    #         continue
    subject_list = source_list_name
    # subject_list.append(target_subject)
    print(subject_list)

    subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

    # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.MCMASTER_FULL_LABEL_PATH, subject_list, target_subject, args.n_class, args.oracle_setting)
    # srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

    # srcs_file_name = 'mcmaster_list_full.txt'   # to train model for McMaster to use as a pre-trained model
    combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, 100, phase='src')
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, 100, phase='tar')

    # calculate dis of every source sample with tar
    close_src_sample_tar_numpy = 'closest_src_samples_tar_'+target_subject+'.npz'
    if not Path(close_src_sample_tar_numpy).is_file():
        measure_src_samples_tar_dist(transfer_model, combine_srcs_loader, tar_loader, dist_measure, close_src_sample_tar_numpy, args.batch_size)
    
    closest_loaded_arrays = np.load(close_src_sample_tar_numpy)
    return closest_loaded_arrays

def get_closest_src_sample_tar(closest_loaded_arrays, relv_sample_count):
    # Loading the arrays
    
    closest_src_samples = closest_loaded_arrays['closest_src_samples']
    closest_src_labels = closest_loaded_arrays['closest_src_labels']
    closest_src_dists = closest_loaded_arrays['closest_src_dists']

    sorted_indices = np.flip(np.argsort(closest_src_dists))
    # get the sorted top N previous src samples indices
    relv_indices = sorted_indices[:relv_sample_count].tolist()
    closest_src_samples_arr = np.array(closest_src_samples)
    closest_src_samples_arr = closest_src_samples_arr[relv_indices]
    closest_src_labels_arr = np.array(closest_src_labels)
    closest_src_labels_arr = closest_src_labels_arr[relv_indices]


    return closest_src_samples_arr, closest_src_labels_arr


def get_top_closest_target_subject(transfer_model, target_file_path, tar_file_name):
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')
    transfer_model, _ = initialize_model(args, os.path.join(config.CURRENT_DIR, config.BIOVID_N_SRC_WEIGHT_FILE), True, args.n_class_N_src, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model

    # target evaluation on trianed source model
    top_srcs_train = test(transfer_model, tar_loader, args.batch_size, True, args.is_pain_dataset)
    # top_srcs_val = test(transfer_model, tar_val_loader, args.batch_size, True, args.is_pain_dataset)
    # top_srcs_test = test(transfer_model, tar_test_loader, args.batch_size, True, args.is_pain_dataset)

    # get subject name from subject Ids
    top_subname_dic = map_subid_to_subname(config.BIOVID_SUBID_TO_SUBNAME_MAPPING, top_srcs_train)
    print(top_subname_dic)

    return top_subname_dic
    
def main(args):

    '''
    BIOVID DATSET:
        - write text file for the numbers of subjects for sources and target
        - create loader for source and target 
        - create src and traget file name
    '''
    NUM_SUBJECTS = 2
    TARGET_SUBJECT = 2

    '''
    Random Sources:
        082714_m_22, 101916_m_40, 083109_m_60, 080209_w_26, 091914_m_46, 083013_w_47, 072514_m_27, 100514_w_51, 082208_w_45, 071614_m_20
        120514_w_56, 072414_m_23, 092009_m_54, 080614_m_24, 082414_m_64, 072609_w_23, 092813_w_24, 102214_w_36, 080709_m_24, 102514_w_40
        111914_w_63, 082814_w_46, 111409_w_63, 111313_m_64, 083114_w_55, 101015_w_43, 101908_m_61, 092808_m_51, 100117_w_36, 110810_m_62

        112909_w_20, 102309_m_61, 110614_m_42, 102008_w_22, 080609_w_27, 082109_m_53, 081714_m_36, 092014_m_56

        last 17 subjects 
        ['071313_m_41', '112809_w_23', '071911_w_24', '100914_m_39', '112610_w_60', '082909_m_47', '101209_w_61', '112310_m_20', 
        '082809_m_26', '091814_m_37', '112016_m_25', '071814_w_23', '101814_m_58', '101309_m_48', '100214_m_50', '101114_w_37', '080714_m_23']

    '''
    '''
    Sources=77
    ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23', '082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42', 
    '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
    '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', 
    '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
    '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
    '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', 
    '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
    '091914_m_46', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24', '071309_w_21']
    
    Random Target:
        081014_w_27 [40]
        101609_m_36 [70]
        112009_w_43 [66]
        091809_w_43 [68]
        071309_w_21 [4]
        073114_m_25 [69]
        080314_w_25 [80]
        073109_w_28 [82]
        100909_w_65 [13]
        081609_w_40 [17]

    '''
    
    # --- Random target subjects
    target_subject_list = ['081014_w_27','101609_m_36','112009_w_43','091809_w_43','071309_w_21','073114_m_25','080314_w_25','073109_w_28','100909_w_65','081609_w_40']

    # --- Balance target subjects
    # target_bal_list = ['082714_m_22','080314_w_25','073114_m_25','073109_w_28','101609_m_36','071313_m_41','091809_w_43','100909_w_65','082208_w_45','111313_m_64']

    '''
    BioVid Random target subjects
    '''
    # source_list_name = ['082208_w_45', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42',
    # '101814_m_58', '112016_m_25', '071313_m_41', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
    # '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '111313_m_64', '071614_m_20', 
    # '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
    # '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
    # '120514_w_56', '083109_m_60', '110909_m_29', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '082714_m_22', 
    # '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
    # '091914_m_46', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']

    '''
    BioVid Balance target subjects
    '''
    # source_list_name = ['081014_w_27', '081714_m_36', '112610_w_60', '101908_m_61', '071709_w_23','082014_w_24', '110810_m_62', '080209_w_26', '101916_m_40', '110614_m_42',
    # '101814_m_58', '112016_m_25', '091914_m_46', '102514_w_40', '100514_w_51', '101114_w_37', '100509_w_43', '082315_w_60', '112310_m_20', '120614_w_61', 
    # '092714_m_64', '101514_w_36', '092813_w_24', '102414_w_58', '102309_m_61', '081617_m_27', '080609_w_27', '083114_w_55', '083109_m_60', '071614_m_20', 
    # '101309_m_48', '071911_w_24', '102316_w_50', '100417_m_44', '083013_w_47', '083009_w_42', '080714_m_23', '101809_m_59', '082909_m_47', '101209_w_61', 
    # '092014_m_56', '072414_m_23', '101015_w_43', '112909_w_20', '111609_m_65', '100117_w_36', '111409_w_63', '080709_m_24', '072714_m_23', '112914_w_51', 
    # '082208_w_45', '071309_w_21', '082714_m_22', '091814_m_37', '071814_w_23', '092509_w_51', '112809_w_23', '100214_m_50', '102214_w_36', '112009_w_43', 
    # '082109_m_53', '092808_m_51', '080309_m_29', '102008_w_22', '111914_w_63', '082809_m_26', '072514_m_27', '082814_w_46', '072609_w_23', '101216_m_40', 
    # '081609_w_40', '100914_m_39', '112209_m_51', '092514_m_50', '092009_m_54', '082414_m_64', '080614_m_24']

    # source_list_name = ['082208_w_45', '081714_m_36', '112610_w_60', '112016_m_25', '091914_m_46', '082315_w_60', '112310_m_20', '120614_w_61']

    '''
    McMaster subjects
    '''
    source_list_name = ['047-jl047', '095-tv095', '048-aa048', '049-bm049', '097-gf097', '052-dr052',  '059-fn059', '124-dn124',   '042-ll042',   
    '106-nm106', '066-mg066','096-bg096', '043-jh043', '101-mg101', '103-jk103', '108-th108', '080-bn080', '120-kz120', '064-ak064', '092-ch092']
    
    unbc_target_subject_list = ['107-hs107', '109-ib109', '121-vw121', '123-jh123', '115-jy115'] 
    
    target_subject = target_subject_list[args.tar_subject]
    print('Selected target subject: ', target_subject)

    # ---------------------------------- ----------------------------------- #

    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
    test_model_name = None

    eliminate_list = []
    selected_sub = 0
    
    count_subs = 1

    relv_feat_arr = []
    relv_src_data_arr = [] 
    relv_src_label_arr = []
    prev_relv_samples_struc = None

    _BEST_VAL_ACC = 0
    is_ignore_model = False 
    prev_src_sub_model = ''

    # comet create experiment name
    set_comet_exp_name(experiment, args.top_k, args.source_combined, len(source_list_name), target_subject)
    target_file_path, target_weight_path, timestamp = create_target_folders(config.CURRENT_DIR, args.weights_folder, target_subject, args.top_timestamp if args.target_evaluation_only else None)


    # Selection of top source subjects w.r.t each target using N classifier 
    if args.train_N_source_classes and selected_sub == 0:
        subject_list = source_list_name
        subject_list.append(target_subject)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

        top_N_srcs_tar = get_top_closest_target_subject(transfer_model, target_file_path, tar_file_name)

    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
    counter_based_N_classifer = 0
    if args.train_source or args.load_source:
        subject_list = source_list_name
        subject_list.append(target_subject)

        subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
        transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model = domain_adaptation(srcs_file_name, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model)
    
    if args.train_w_src_sm:
        source_model_name = 'WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
        transfer_model, optimizer = initialize_model(args, source_model_name, True, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
        
    relv_sample_count = 2000
    closest_src_samples_arr = [] 
    closest_src_labels_arr = []
    if args.train_w_src_sm:
        subject_list = source_list_name
        subject_list.append(target_subject)
        # source_model_name = 'WeightFiles/lab_srcs78_082208w45_081714m36_112610w60_101908m61_071709w23_082014w24_110810m62_080209w26_101916m40_110614m42_____only'
        # transfer_model, optimizer = initialize_model(args, source_model_name, True, args.n_class, pretrained_model_name = None) # pretrained_model_name = args.pretrained_model
        if not args.train_w_rand_src_sm:
            closest_loaded_arrays = dist_measure_dic = calculate_src_sampl_tar_dist(source_list_name, target_subject, transfer_model, target_file_path, args.dist_measure)
        while relv_sample_count < args.top_rev_src_sam:
            if not args.train_w_rand_src_sm:
                closest_src_samples_arr, closest_src_labels_arr = get_closest_src_sample_tar(closest_loaded_arrays, relv_sample_count)

            subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
            srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

            transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
            transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, relv_sample_count, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, None, None, None, None, closest_src_samples_arr, closest_src_labels_arr)
            relv_sample_count = relv_sample_count + 2000

    elif not args.target_evaluation_only:
        while selected_sub < args.top_k:

            # if selected_sub == 0 :  # THIS IS FOR 77 SOURCES ADAPTED TO TARGET '081609_w_40'
            #     src_tar_dist_dic = {'083109_m_60': 0.6030410408973694, '102309_m_61': 0.6049227381861487, '072414_m_23': 0.6055345017176408, '081714_m_36': 0.6074512179081256, '080709_m_24': 0.6074585939039949, '110909_m_29': 0.6077407480196189}
            # else:
            # if selected_sub == 0 :  # THIS IS FOR 77 SOURCES ADAPTED TO TARGET '101609_m_36'
            #     src_tar_dist_dic = {'072414_m_23': 0.604693013888139, '082714_m_22': 0.6072624758817255, '082208_w_45': 0.6078853152180446, '120514_w_56': 0.60897833609399, '112914_w_51': 0.6090267573604147, '081714_m_36': 0.6091837791296152, '102414_w_58': 0.6094778052723134, '080709_m_24': 0.6096465734184765}
            # else:
            
            # if selected_sub == 0 :  # THIS IS FOR 77 SOURCES ADAPTED TO TARGET '101609_m_36'
            #     # src_tar_dist_dic = {'082208_w_45': 0.604693013888139, '092514_m_50': 0.6072624758817255, '101015_w_43': 0.6078853152180446, '092509_w_51': 0.6091837791296152, '101908_m_61': 0.6094778052723134}   
                
            #     src_tar_dist_dic = {'082109_m_53': 0.604693013888139, '082814_w_46': 0.6072624758817255, '071614_m_20': 0.6078853152180446, '100417_m_44': 0.6091837791296152, '100214_m_50': 0.6094778052723134, '082809_m_26': 0.6096465734184765, '072714_m_23': 0.6097465734184765, '111914_w_63': 0.6098465734184765, '083114_w_55': 0.6099465734184765, '082909_m_47': 0.6099965734184765}   
            
            if args.train_N_source_classes and args.train_with_dist_measure:
                if target_subject in source_list_name:
                    source_list_name.remove(target_subject)

                dist_measure_dic = calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, args.cs_threshold, target_file_path, args.dist_measure)
                # dist_measure_dic = {'083109_m_60': 0.6030410408973694, '102309_m_61': 0.6049227381861487, '072414_m_23': 0.6055345017176408}
                N_dist_dic = top_N_srcs_tar

                if N_dist_dic:
                    pick_top_N_src_sb = min(3, len(N_dist_dic))
                    src_subjects = list(set(dist_measure_dic).intersection(dict(list(N_dist_dic.items())[:pick_top_N_src_sb])))     # get common subjects from dist measure and N-Classifier
                else:
                    src_subjects = []

                if len(src_subjects) > 0:
                    print("\n ---------------------- -------------------- -----------------\n")
                    print("Common Source Subjects B/W Dist Measure and N-Classifier): ", src_subjects)
                    print("\n ---------------------- -------------------- -----------------\n")
                    top_N_srcs_tar = {k: v for k, v in top_N_srcs_tar.items() if k not in src_subjects}    # remove key/value pair from N-classifier dic 
                else:
                    pick_top_N_src_sb = min(3, len(N_dist_dic) if N_dist_dic else len(dist_measure_dic))
                    src_subjects = list(N_dist_dic.keys())[:pick_top_N_src_sb] if N_dist_dic else list(dist_measure_dic.keys())[:pick_top_N_src_sb]

                    top_N_srcs_tar = {k: v for k, v in top_N_srcs_tar.items() if k not in src_subjects}    # remove key/value pair from N-classifier dic 
                    print("\n ---------------------- --------------------\n")
                    print("No Common Source Subjects")
                    print("Selecting subjects: ", src_subjects)
                    print("\n ---------------------- --------------------\n")

            elif args.train_N_source_classes and selected_sub == 0 :
                source_list_name.remove(target_subject)
                src_subjects = list(top_N_srcs_tar.keys())
                counter_based_N_classifer = len(src_subjects)
                print("counter_based_N_classifer: ", counter_based_N_classifer)
                # src_tar_dist_dic = {'082109_m_53': 0.604693013888139}
            else:
                src_tar_dist_dic = calculate_src_tar_dist(source_list_name, target_subject, eliminate_list, transfer_model, args.cs_threshold, target_file_path, args.dist_measure)
                # src_tar_dist_dic = {'083109_m_60': 0.6030410408973694, '102309_m_61': 0.6049227381861487, '072414_m_23': 0.6055345017176408}
                # src_tar_dist_dic = {'048-aa048': 0.6030410408973694, '049-bm049': 0.6049227381861487, '097-gf097': 0.6055345017176408}
                
                src_subjects = list(src_tar_dist_dic.keys())
                if selected_sub == 0:
                    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
        
        # src_tar_dist_dic = dict(sorted(src_tar_dist_dic.items(), key=lambda x:x[1]))
        # src_keys_list = [key for key in src_tar_dist_dic.keys()] # extract the selected subjects and add to the eliminate list to avoid these subjects next time
        
            for src_key in src_subjects:
                eliminate_list.append(src_key)
            # eliminate_list = eliminate_list[0] # this is to avoid double list [[]]

            # src_subjects = list(src_tar_dist_dic.keys())
            selected_sub = selected_sub + len(src_subjects)

            '''
                if : combine selected source subjects and adapt to target
                else : selected source subjects will be adpted individually with target  
            '''
            if args.source_combined:
                subject_list = src_subjects
                subject_list.append(target_subject)

                subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
                srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
                transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model) 
            else:
                for src in src_subjects:
                    # subject_list = [src]
                    '''
                        New strategy: using two source subject loaders.
                        First loader contains all the previous subjects, Second loader contain only the newly added source subject
                        - In case of first sub, put same sub in both loaders
                    ''' 
                    # prev_subject_list = None
                    if args.accumulate_prev_source_subs:
                        print("\n *** Accumulating previously adapted source subjects in the first loader *** \n")
                        if count_subs > 1:
                            prev_subject_list.remove(target_subject)
                            prev_subject_list.append(src)
                        else:
                            prev_subject_list = [src]
                        print('Total previous subjects:', len(prev_subject_list))
                        prev_subject_list.append(target_subject)
                    subject_list = [src]
                    subject_list.append(target_subject)

                     # generating file for previous source sub loader 
                    if args.accumulate_prev_source_subs:
                        prev_subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, prev_subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting, args.pain_tar_label_path)
                        prev_srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(prev_subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)

                    subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, args.pain_label_path, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting, args.pain_tar_label_path)
                    srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
                    transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc = domain_adaptation(srcs_file_name, prev_srcs_file_name, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc, closest_src_samples_arr, closest_src_labels_arr)
                    count_subs = count_subs + 1

        print("\n ------------------ ------------------\n")
        print("Source selection based on N classifier: ", counter_based_N_classifer)
        print("\n ------------------ ------------------\n")

        print("\n ------------------ ------------------\n")
        print("Source selection based on threshold: ", counter_based_N_classifer - count_subs)
        print("\n ------------------ ------------------\n")
    else:
        subject_list = [args.top_src_sub]
        subject_list.append(target_subject)

        # subject_list = write_srcs_tar_txt_files_using_list(args.src_train_datasets_path, config.BIOVID_REDUCE_LABEL_PATH, subject_list, target_subject, args.n_class, target_file_path, args.oracle_setting)
        srcs_file_name, tar_file_name, _ = get_srcs_tar_name_using_list(subject_list, args.src_train_datasets_path, target_subject, args.n_class, args.oracle_setting)
        tar_file_name='lab_srcs3_083013w47_112809w23_tar_081609w40.txt'
        srcs_file_name='lab_srcs3_083013w47_112809w23_only.txt'
        transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, _, _, _, _ = domain_adaptation(srcs_file_name, None, tar_file_name, transfer_model, optimizer, target_subject, count_subs, test_model_name, target_file_path, target_weight_path, timestamp, args.dist_measure, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model)

def domain_adaptation(srcs_file_name, prev_srcs_file_name, tar_file_name, transfer_model, optimizer, target_subject, count_subs, 
                      test_model_name, target_file_path, target_weight_path, timestamp, dist_measure, _BEST_VAL_ACC, is_ignore_model, 
                      prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc, closest_src_samples_arr, 
                      closest_src_labels_arr):
    
    combine_srcs_loader, combine_srcs_val_loader, combine_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, srcs_file_name), None, args.batch_size, phase='src')
    tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_tar_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')

    source_model_name = srcs_file_name.split('.')[0]
    tar_model_name = tar_file_name.split('.')[0] + "_oracle" if args.oracle_setting else tar_file_name.split('.')[0] 
    lamb = 0.5 # weight for transfer loss, it is a hyperparameter that needs to be tuned

    # create data loader for current source subject
    if prev_srcs_file_name and not args.apply_replay:
        prev_srcs_loader, prev_srcs_val_loader, curr_srcs_test_loader = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, prev_srcs_file_name), None, args.batch_size, phase='src')
        # tar_loader, tar_val_loader, tar_test_loader  = BaseDataset.load_pain_dataset(args.pain_db_root_path, os.path.join(target_file_path, tar_file_name), None, args.batch_size, phase='tar')
    
    # prev_srcs_loader_dic.extend(combine_srcs_loader)
    # relv_sample_count = len(combine_srcs_loader)*args.batch_size
    relv_sample_count = 2000 # biovid = 2000, cross-dataset = 1000
    # *** *** Selection of relevant source samples --- ****
    if args.accumulate_prev_source_subs and args.apply_replay:

        if prev_relv_samples_struc:
            relv_src_data_arr = [point[0] for point in prev_relv_samples_struc if point[0] is not None]
            relv_src_label_arr = [point[2] for point in prev_relv_samples_struc if point[2] is not None]

            relv_prev_src_subs = TensorDataset(torch.tensor(np.array(relv_src_data_arr)), torch.tensor(relv_src_label_arr)) 
            prev_srcs_loader, prev_srcs_val_loader = BaseDataset.load_target_data(relv_prev_src_subs, args.batch_size, split=False)
        else:
            prev_srcs_loader, prev_srcs_val_loader = combine_srcs_loader, combine_srcs_val_loader,

    if args.train_w_src_sm:
        if len(closest_src_samples_arr) > 0 and len(closest_src_labels_arr) > 0:
            relv_prev_src_subs = TensorDataset(torch.tensor(np.array(closest_src_samples_arr)), torch.tensor(closest_src_labels_arr)) 
            prev_srcs_loader, prev_srcs_val_loader = BaseDataset.load_target_data(relv_prev_src_subs, args.batch_size, split=False)
        elif args.train_w_rand_src_sm:
            prev_srcs_loader = BaseDataset.generate_random_src_samples(combine_srcs_loader, count_subs)
            prev_srcs_val_loader = BaseDataset.generate_random_src_samples(combine_srcs_val_loader, 300)
        tar_model_name = tar_model_name + '_' + str(count_subs)
        
    dataloaders = {
        'tar': tar_loader,
        'tar_val': tar_val_loader,
        'tar_test': tar_test_loader,
        'combine_srcs': combine_srcs_loader,
        'combine_srcs_val': combine_srcs_val_loader,
        'combine_srcs_test': combine_srcs_test_loader,
        'prev_srcs': prev_srcs_loader,
        'prev_srcs_val': prev_srcs_val_loader
    }

    # load last trained model
    if test_model_name is not None:
        # transfer_model.load_state_dict(torch.load(target_weight_path + '/' + test_model_name + '.pkl'))
        if is_ignore_model:
            test_model_name = prev_src_sub_model

        transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)

        if args.load_prev_source_model:
            target_trained_model = torch.load(target_weight_path + '/' + test_model_name  + '_load.pt')
            transfer_model.load_state_dict(target_trained_model['model_state_dict'])
            optimizer.load_state_dict(target_trained_model['optimizer_state_dict'])
    
    if not args.target_evaluation_only:        
        transfer_model, each_sub_train_total_loss = train(dataloaders, 
                                                          transfer_model, 
                                                          optimizer, lamb, 
                                                          source_model_name, 
                                                          tar_model_name, 
                                                          target_subject, 
                                                          target_weight_path, 
                                                          timestamp, 
                                                          dist_measure, 
                                                          args, 
                                                          relv_sample_count)
        
        experiment.log_metric("Source Subjects Total Train Loss", each_sub_train_total_loss, step=count_subs)

    if args.train_model_wo_adaptation: 
        test_model_name = source_model_name 
    elif args.source_free:
        test_model_name = tar_model_name + '_source_free'
    elif args.source_combined:
        test_model_name = tar_model_name + '_source_combined' 
    elif args.train_source:
        test_model_name = source_model_name
    else:
        test_model_name = tar_model_name
    
    # transfer_model.load_state_dict(torch.load(target_weight_path + '/' + test_model_name + '.pkl'))
    transfer_model, optimizer = initialize_model(args, None, args.load_source, args.n_class, pretrained_model_name = None)
    target_trained_model = torch.load(target_weight_path + '/' + test_model_name  + '_load.pt')
    transfer_model.load_state_dict(target_trained_model['model_state_dict'])
    optimizer.load_state_dict(target_trained_model['optimizer_state_dict'])

     # *** *** Selection of relevant source samples --- ****
    if args.accumulate_prev_source_subs and args.apply_replay:        
        prev_relv_samples_struc = create_relv_src_clus_dbscan(dataloaders['combine_srcs'], dataloaders['tar'], transfer_model, prev_relv_samples_struc, relv_sample_count, timestamp + '/' + test_model_name)

    # optimizer.load 'optimizer_state_dict': optimizer.state_dict()
    # acc_test, acc_top2 = test(transfer_model, dataloaders['tar_test'], args.batch_size)
    # if args.train_source:
    if args.train_source and not args.target_evaluation_only: 
        train_loader = dataloaders['combine_srcs']
        val_loader = dataloaders['combine_srcs_val']
        test_loader = dataloaders['combine_srcs_test']
    else:
        train_loader = dataloaders['tar']
        val_loader = dataloaders['tar_val']
        test_loader = dataloaders['tar_test']

    acc = test(transfer_model, train_loader, args.batch_size, False, args.is_pain_dataset)
    acc_val = test(transfer_model, val_loader, args.batch_size, False, args.is_pain_dataset)
    acc_test = test(transfer_model, test_loader, args.batch_size, False, args.is_pain_dataset)

    experiment.log_metric("Val Accuracy", acc_val, step=count_subs)
    experiment.log_metric("Test Accuracy", acc_test, step=count_subs)

    '''
        - Ignore the subject/model which has the lowest accuracy then the best one
        - Store the previous model/subject to initialize the next subject model  
    '''
    # _BEST_VAL_ACC = _BEST_VAL_ACC - 0.05
    # if acc_val.item() > _BEST_VAL_ACC:
    #     _BEST_VAL_ACC = acc_val.item()
    #     is_ignore_model = False
    #     prev_src_sub_model = test_model_name

    #     experiment.log_metric("Val Accuracy", acc_val, step=count_subs)
    #     experiment.log_metric("Test Accuracy", acc_test, step=count_subs)
    # else:
    #     is_ignore_model = True
    #     prev_src_sub_model = temp_src_sub_model

    #     print("\n ------------------ ------------------\n")
    #     print("IGNORE MODEL/SUBJECT: ", test_model_name)
    #     print("\n ------------------ ------------------\n")

    '''
        END
    '''

    # to generate t_sne graph
    # _data_arr, _prob_arr, _label_arr, _gt_arr, tcl_clusters = create_target_pl_dicts(transfer_model, dataloaders['tar_val'], 0.00, args.batch_size, args.is_pain_dataset)
    
    print('Source model: ', source_model_name)
    print('Target: ', target_subject)
    print('Target model: ', tar_model_name)

    print(f'Target Accuracy: {acc}')
    print(f'Target Val Accuracy: {acc_val}')
    print(f'Target Test Accuracy: {acc_test}')
    
    return transfer_model, test_model_name, _BEST_VAL_ACC, is_ignore_model, prev_src_sub_model, relv_feat_arr, relv_src_data_arr, relv_src_label_arr, prev_relv_samples_struc

def initialize_model(args, source_model_name, load_source, n_class, pretrained_model_name = None):
    transfer_model = TransferNet(n_class, transfer_loss=transfer_loss, base_net=args.back_bone).cuda()
    # load multi-source pre-trained model and adapt to target domain 

    optimizer = torch.optim.SGD([
        {'params': transfer_model.base_network.parameters()},
        {'params': transfer_model.bottleneck_layer.parameters(), 'lr': 10 * learning_rate},
        {'params': transfer_model.classifier_layer.parameters(), 'lr': 10 * learning_rate},
    ], lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    if pretrained_model_name is not None and not load_source:
        source_trained_model = torch.load(pretrained_model_name + '_load.pt')

        # change classificaiton layer of the pre-trained model to the num of classes for the defined Architecture
        source_trained_model['model_state_dict']['classifier_layer.3.weight'] = source_trained_model['model_state_dict']['classifier_layer.3.weight'][:n_class,:]
        source_trained_model['model_state_dict']['classifier_layer.3.bias'] = source_trained_model['model_state_dict']['classifier_layer.3.bias'][:n_class]
        
        transfer_model.load_state_dict(source_trained_model['model_state_dict'])

    # -- load trained source model and train only target
    if load_source and source_model_name is not None and not args.single_best:
        source_trained_model = torch.load(source_model_name + '_load.pt')
        transfer_model.load_state_dict(source_trained_model['model_state_dict'])
        optimizer.load_state_dict(source_trained_model['optimizer_state_dict'])

        # for single UDA
        # source_trained_model = torch.load(source_model_name + '.pt')
        # transfer_model.load_state_dict(source_trained_model)
    
    return transfer_model, optimizer


def train(dataloaders, model, optimizer, lamb, source_model_name, tar_model_name, target_sub_name, target_weight_path, timestamp, dist_measure, args, relv_sample_count):
    target_loader, combine_srcs_loader, prev_srcs_loader  = dataloaders['tar'], dataloaders['combine_srcs'], dataloaders['prev_srcs']
    target_val_loader, combine_srcs_val_loader, prev_srcs_val_loader = dataloaders['tar_val'],dataloaders['combine_srcs_val'], dataloaders['prev_srcs_val']
    len_target_loader, len_combine_srcs_loader, len_prev_srcs_loader = len(target_loader), len(combine_srcs_loader), len(prev_srcs_loader)

    # srcs_loader, srcs_val_loader = dataloaders['srcs'], dataloaders['srcs_val'] 
    
    criterion = nn.CrossEntropyLoss()

    with experiment.train():
        experiment.log_parameter("Sources", get_dataset_name_from_path(args.src_train_datasets_path))
        experiment.log_parameter("Training", config.TRAIN_ONLY_TARGET if args.load_source else config.TRAIN_SOURCE_AND_TARGET)
        experiment.log_parameter("Back_bone", args.back_bone)
        experiment.log_parameter("Source epoch", args.source_epochs)
        experiment.log_parameter("Target epoch", args.target_epochs)
        experiment.log_parameter("Learning_rate", "{:.4}".format(float(learning_rate)))
        experiment.log_parameter("Loss", 'CrossEntropyLoss')
        experiment.log_parameter("Test dataset", get_dataset_name_from_path(args.tar_dataset_path))
        experiment.log_parameter("Target model name", tar_model_name)
        experiment.log_parameter("Source model name", source_model_name)
        experiment.log_parameter("TARGET SUB NAME", target_sub_name)
        experiment.log_parameter("Optimizer", optimizer.__module__)
        experiment.log_parameter("Class", args.n_class)
        experiment.log_parameter("Transfer loss", transfer_loss)
        experiment.log_parameter("ImageNet weights", True)
        experiment.log_parameter("Pre-trained Model", args.pretrained_model)
        experiment.log_parameter("Source subjects combine", args.source_combined)
        experiment.log_parameter("Subject selection Top k", args.top_k)
        experiment.log_parameter("Time stamp", timestamp)
        experiment.log_parameter("Distance measure", dist_mapping(dist_measure))
        experiment.log_parameter("Train using N classifier", args.train_N_source_classes)
        experiment.log_parameter("Train using Distance Measure ", args.train_with_dist_measure)
        experiment.log_parameter("Firt adapt to N source subjects", args.first_adapt_N_source_subjects)
        experiment.log_parameter("Load previous source subject model ", args.load_prev_source_model)
        experiment.log_parameter("Accumulate previous source subjects ", args.accumulate_prev_source_subs)
        experiment.log_parameter("Target dataset expanded", args.expand_tar_dataset)
        experiment.log_parameter("SEED ", _get_current_seed())
        experiment.log_parameter("Experiment description", args.experiment_description)
        experiment.log_parameter("Torch_version:", torch.__version__)
        experiment.log_parameter("Cuda_version:", torch.version.cuda)
        experiment.log_parameter("Selected relevant history samples:", relv_sample_count)
        experiment.log_parameter("Early_stop:", args.early_stop)
        experiment.log_parameter("cs_threshold:", args.cs_threshold)
        experiment.log_parameter("apply_replay:", args.apply_replay)        

        # ------------------------------------------ ------------------------------------- #
        # Training of labeled multi-source data 
        #
        # Experiments:
        # ---- 1. Source#1 with Source#2 tested with Source#1
        # ---- 2. Source#1 with Source#2 tested with Source#2
        # ------------------------------------------ ------------------------------------- #
        each_sub_train_total_loss = 0
        if args.train_source and not args.load_source and not args.train_model_wo_adaptation:
            # n_batch = min(len_source1_loader, len_source2_loader)
            # n_batch = min(len_combine_srcs_loader, len_target_loader)
            n_batch = len_combine_srcs_loader
            print('\n ------ Start Training of Source Domain ------ \n')
            # train_multi_model(args.source_epochs, model, source1_loader, source2_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, combine_srcs_val_loader, args.oracle_setting, train_source=True)

            # train_multi_src_model_2(args.source_epochs, model, srcs_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, srcs_val_loader, args.oracle_setting, train_source=True)
            
            model, _ = train_multi_model(args.source_epochs, model, combine_srcs_loader, combine_srcs_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, combine_srcs_val_loader, target_weight_path, args.oracle_setting, train_source=args.train_source)
        
        # ------------------------------------------ ------------------------------------- #
        # Generating robust target labels
        # ------------------------------------------ ------------------------------------- #
        # if args.target_clustering:
        #     # cluster_centroids = create_clusters_from_source_feat(model, combine_srcs_loader)
        #     # load source cluster centroids -- also add check when to use the existing clusters, and when to generate new once 
        #     loaded_data = np.load('source_centroids_256.npz') # define path in the function call
        #     centroids = {key: loaded_data[key] for key in loaded_data} # convert string back to dictionary
        #     n_batch = len_target_loader
        #     tar_model_name = tar_model_name + '_cluster_target' 
        #     generate_robust_target_label(model, target_loader, optimizer, criterion, args.target_epochs, n_batch, args.early_stop, tar_model_name, centroids, target_val_loader)
        #     # test(model, target_val_loader, n_batch)

        # ------------------------------------------ ------------------------------------- #
        # Training of labeled multi-source with unlabeled target model
        #
        # Experiments:
        # ---- 1. Source#1 with Target
        # ---- 2. Source#2 with Target
        # ---- 3. Source#1 + Source#2 with Target
        # ------------------------------------------ ------------------------------------- #
        if args.train_w_src_sm:
            print('\n ------ Start Training of Target Domain with Source Samples ------ \n')
            # train_multi_model(args.source_epochs, model, source1_loader, source2_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, combine_srcs_val_loader, args.oracle_setting, train_source=True)

            # train_multi_src_model_2(args.source_epochs, model, srcs_loader, optimizer, criterion, lamb, n_batch , args.early_stop, source_model_name, srcs_val_loader, args.oracle_setting, train_source=True)
            n_batch = min(len_prev_srcs_loader, len_target_loader)
            model, each_sub_train_total_loss = train_multi_model_only_src_sample(args.target_epochs, model, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch , args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)
           
        if not args.train_source and not args.train_w_src_sm:
            if args.source_combined:
                tar_model_name = tar_model_name + '_source_combined' 
                print('\n ----------------------------------------- -----------------------------------------------\n')
                print('\n ------ Start Training of Target Domain (Combined Source) ------ \n')
                print('\n ----------------------------------------- -----------------------------------------------\n')

            else:
                print('\n ----------------------------------------- -----------------------------------------------\n')
                print('\n ------ Start Training of Target Domain ------ \n')
                print('\n ----------------------------------------- -----------------------------------------------\n')

            n_batch = min(len_combine_srcs_loader, len_target_loader)
            model, each_sub_train_total_loss = train_multi_model(args.target_epochs, model, combine_srcs_loader, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch, args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)
            # model, each_sub_train_total_loss = train_multi_model_all_srcs(args.target_epochs, model, combine_srcs_loader, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch, args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)
            # to include all prev src subjects
            # model, each_sub_train_total_loss = train_multi_model_only_src_sample(args.target_epochs, model, prev_srcs_loader, target_loader, optimizer, criterion, lamb, n_batch , args.early_stop, tar_model_name, target_val_loader, target_weight_path, args.oracle_setting, train_source=False)

        return model, each_sub_train_total_loss
def train_multi_model_only_src_sample(n_epoch, model, data_loader1, data_loader2, optimizer, criterion, lamb, n_batch , early_stop, trained_model_name, val_loader, target_weight_path, oracle_setting, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    clusters_by_label = {}
    feat_memory_bank = {}
    threshold = 0.91

    # tar_loader_for_PL = data_loader2
    # tar_loader = data_loader2
    calculate_tar_pl_ce = False
    train_total_loss_all_epochs = 0

    # SupConCriterion = SupConLoss()
    current_forzen_model = copy.deepcopy(model)

    # train_source = False # remove this line ; ITS ONLY THERE TO PERFORM EXPERIMENTS ON THE MMD LOSS FOR DOMAIN SHIFT FOR GDA FOR DGA-1033 PRESENTATION
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0
        # calculate_tar_pl_ce = False

        if train_source is False and oracle_setting is False:
            if e % 20 == 0: # remove e != 0 and
                # create a copy of a previously trained model to generate target PL
                current_forzen_model = copy.deepcopy(model)

                print("\n**** Threshold Reduced From: ", threshold)
                threshold = threshold - 0.01      
                print(" To: ", threshold)

                # ***** replace tar_loader_for_PL with data_loader2 to load all the samples everytime it calculates target PL
                # _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr, non_conf_label_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)
                # # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader_for_PL, threshold)
                # # if len(_data_arr) <= 0:
                # #     threshold = threshold - 0.06
                # if len(_data_arr) > 0:
                #     for i in range(0,1):
                #         if len(_data_arr) > 0:
                #             # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                #             target_wth_gt_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #             tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                #             _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr_du, non_conf_label_arr_du = create_target_pl_dicts(model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)
                #             # non_conf_data_arr = np.concatenate((non_conf_data_arr, non_conf_data_arr_du), axis=0)
                #             # non_conf_label_arr = np.concatenate((non_conf_label_arr, non_conf_label_arr_du), axis=0)
                            
                #             # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader, threshold)
                #     if len(_data_arr) > 0:
                #         if args.expand_tar_dataset:
                #             tar_exp_data = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size, _label_arr, _prob_arr)
                #             tar_ext_data_arr, tar_ext_gt_arr, _label_arr, _prob_arr = tar_exp_data["_data_arr"],  tar_exp_data["_gt_arr"], tar_exp_data["_label_arr"], tar_exp_data["_prob_arr"] 
                #         else:
                #             tar_ext_data_arr, tar_ext_gt_arr = _data_arr, _gt_arr
                        
                #         # tar_ext_data_arr = np.concatenate((_data_arr, non_conf_data_arr), axis=0)
                #         # tar_ext_gt_arr = np.concatenate((_gt_arr, non_conf_label_arr), axis=0)
                #         # _label_arr = np.concatenate((_label_arr, non_conf_label_arr), axis=0)
                        
                #         target_wth_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_label_arr), torch.tensor(_prob_arr), torch.tensor(_gt_arr))
                #         # target_wth_labels = TensorDataset(torch.tensor(tar_ext_data_arr), torch.tensor(_label_arr), torch.tensor(tar_ext_gt_arr))
                        
                #         tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_labels, args.batch_size, split=False)
                #         calculate_tar_pl_ce = True
                #         # data_loader2 = tar_loader
                #         # n_batch = min(len(data_loader1), len(tar_loader))

                #         target_for_PL = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #         tar_loader_for_PL, _ = BaseDataset.load_target_data(target_for_PL, args.batch_size, split=False)
                

        model.train()
        
        count = 0
        total_mmd = 0
        srcs_avg_features = []
        clusters_by_label = {}
        feat_memory_bank = {}
        feat_memory_bank_tar = {}
        gt_arr = []
        tar_arr = []
        # tar_pl_batch = len(_data_arr)/args.batch_size
        tar_counter = 0
        
        # conf_tar_data_iter = TragetRestartableIterator(tar_loader)
        tar_data_iter = TragetRestartableIterator(data_loader2)
        n_batch = min(len(data_loader1), len(data_loader2))

        for data_domain1, label_domain1 in tqdm(data_loader1, leave=False):
        # for (domain1, tar_domain) in zip(data_loader1, data_loader2):
            # data_source, label_source = src
            # data_target, _ = tar
            # data_source, label_source = data_source.cuda(), label_source.cuda()
            # data_target = data_target.cuda()
            count = count + 1  

            ### --- *** defining for conf target samples
            # domain2 = next(conf_tar_data_iter)

            ### --- *** Current Source Subject
            # data_domain1, label_domain1 = domain1
            data_domain1, label_domain1 = data_domain1.cuda(), label_domain1.cuda()

            ### --- *** Previous Source Subjects
            # prev_data_domain, prev_label_domain = prev_src_domain
            # prev_data_domain, prev_label_domain = prev_data_domain.cuda(), prev_label_domain.cuda()

            ### --- *** Target Subject
            tar_domain = next(tar_data_iter)
            # tar_data_domain, tar_label_domain = tar_domain
            # tar_data_domain, tar_label_domain = tar_data_domain.cuda(), tar_label_domain.cuda()

            # -- Generate target PL in minibatch
            if train_source is False and oracle_setting is False:
                for i in range(0,1):
                    # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                    target_wth_gt_labels = TensorDataset(tar_domain[0], tar_domain[1])
                    tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                    conf_data_arr, _prob_arr, pl_label_arr, _gt_arr, _, _ = create_target_pl_dicts(current_forzen_model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)

                    calculate_tar_pl_ce = True
                    if len(conf_data_arr) <= 0:
                        calculate_tar_pl_ce = False
                        break

             # defining for domain-2
            # if train_source or oracle_setting:
            #     data_domain2, label_domain2 = tar_domain
            #     data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
            # else:
            if calculate_tar_pl_ce:
                data_domain2, label_domain2 = tar_domain
                conf_data_domain2, conf_label_domain2 = torch.tensor(conf_data_arr).cuda(), torch.tensor(pl_label_arr).cuda()
            else:
                data_domain2, label_domain2 = tar_domain
            
            data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
                # print(label_domain2)
                
                # data_domain2, _ = domain2
                # data_domain2 = data_domain2.cuda()

            # macs, params =profile(model, inputs=(data_domain1.float(), data_domain2.float()))

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            data_domain1 = data_domain1.float()
            data_domain2 = data_domain2.float()
            # prev_data_domain = prev_data_domain.float()
            # prev_label_domain = prev_label_domain.to(torch.int64)
            # tar_data_domain = tar_data_domain.float()

            ''' 
            Test-1000: 
                only considered target PL for adaptaion (MMD+PL) and does not converge on tar subject
                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2 if calculate_tar_pl_ce else None)
            '''

            # calculate target PL loss only with conf samples
            clf_loss, transfer_loss = 0, 0
            if calculate_tar_pl_ce and count <= len(tar_loader):
                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, conf_data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)
            else:
                label_source_pred, transfer_loss, _ = model(data_domain1, data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)

                # added prev source loss always
                # prev_label_source_pred, _, _ = model(prev_data_domain, None)
                # prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

            #----- ALL MMD: calculate mmd loss with prev source sub
            # prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
            # prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

            # label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2)
            # clf_loss = criterion(label_source_pred, label_domain1)
            # transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'

            # loss = (clf_loss) + lamb * transfer_loss
            transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + transfer_loss

            
            # loss = (clf_loss) 

            total_mmd += transfer_loss 
            # loss.backward()

            # adding target loss with source loss
            if train_source or oracle_setting:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(data_domain2, data_domain1)
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) + lamb * transfer_loss_domain2

                # loss.backward(retain_graph=True) # param: retain_graph=True if wanted to backpropogate two losses separately
                # loss_domain2.backward()

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce and count <= len(tar_loader) :
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(conf_data_domain2, None)
                    clf_loss_domain2 = criterion(label_pred_domain2, conf_label_domain2)
                    loss = clf_loss_domain2 + loss
                    
                    optimizer.zero_grad()
                    loss.backward()

                    tar_counter = tar_counter + 1
                    train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                else:
                    train_loss_clf_domain2 = 0
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() if clf_loss else 0 + train_loss_clf + train_loss_clf_domain2
            train_loss_transfer = transfer_loss  + train_loss_transfer
            train_loss_total = train_loss_clf + train_loss_transfer
            # train_loss_total = combine_loss.detach().item() + train_loss_total

            # target loss_clf
            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

        acc = test(model, val_loader, args.batch_size, False, args.is_pain_dataset)

        # if e % 5 == 0:
        #     plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, gt_arr, None)
        #     plot_tsne_graph_scr_tar(feat_memory_bank_tar, feat_memory_bank_tar, tar_arr, None)
        
        if train_loss_clf_domain2 > 0:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, train_tar_pl_loss_clf: {train_loss_clf_domain2/len(tar_loader):.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        else:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')

        experiment.log_metric("Source-1 loss:", train_loss_clf/n_batch, epoch=e)
        experiment.log_metric("Total loss:", train_loss_total/n_batch, epoch=e)
        if train_loss_clf_domain2 > 0 and len(tar_loader) > 0:
            experiment.log_metric("Target pl train loss:", train_loss_clf_domain2/len(tar_loader))

        experiment.log_metric("Each epoch top 1 accuracy", acc, epoch=e)
        # experiment.log_metric("Each epoch top 2 accuracy", acc_top2)

        # add up all the epochs total losses
        train_total_loss_all_epochs = train_total_loss_all_epochs + train_loss_total/n_batch

        if best_acc < acc:
            best_acc = acc
            trained_model_name = trained_model_name + ''
            torch.save(model.state_dict(), target_weight_path + '/' + trained_model_name + '.pkl')
            # torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + 'FER_model.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, target_weight_path + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, target_weight_path + '/' + trained_model_name + '_features.pt')
            experiment.log_metric("Val Target Best Accuracy", best_acc, epoch=e)
            stop = 0
        if stop >= early_stop:
            break
    print(total_mmd)
    # _data_arr, _prob_arr, _label_arr, _gt_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)

    # visualize_tsne(clusters_by_label)
    print("Best Val Acc: ", best_acc)
    train_total_loss_all_epochs = train_total_loss_all_epochs/n_epoch
    return model, train_total_loss_all_epochs
        
def train_multi_model(n_epoch, model, data_loader1, prev_src_loader, data_loader2, optimizer, criterion, lamb, n_batch , early_stop, trained_model_name, val_loader, target_weight_path, oracle_setting, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    clusters_by_label = {}
    feat_memory_bank = {}
    threshold = 0.91

    # tar_loader_for_PL = data_loader2
    # tar_loader = data_loader2
    calculate_tar_pl_ce = False
    train_total_loss_all_epochs = 0

    # SupConCriterion = SupConLoss()
    current_forzen_model = copy.deepcopy(model)

    # train_source = False # remove this line ; ITS ONLY THERE TO PERFORM EXPERIMENTS ON THE MMD LOSS FOR DOMAIN SHIFT FOR GDA FOR DGA-1033 PRESENTATION
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0
        # calculate_tar_pl_ce = False

        if train_source is False and oracle_setting is False:
            if e % 10 == 0: # remove e != 0 and
                # create a copy of a previously trained model to generate target PL
                current_forzen_model = copy.deepcopy(model)

                print("\n**** Threshold Reduced From: ", threshold)
                threshold = threshold - 0.01      
                print(" To: ", threshold)

                # ***** replace tar_loader_for_PL with data_loader2 to load all the samples everytime it calculates target PL
                # _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr, non_conf_label_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)
                # # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader_for_PL, threshold)
                # # if len(_data_arr) <= 0:
                # #     threshold = threshold - 0.06
                # if len(_data_arr) > 0:
                #     for i in range(0,1):
                #         if len(_data_arr) > 0:
                #             # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                #             target_wth_gt_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #             tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                #             _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr_du, non_conf_label_arr_du = create_target_pl_dicts(model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)
                #             # non_conf_data_arr = np.concatenate((non_conf_data_arr, non_conf_data_arr_du), axis=0)
                #             # non_conf_label_arr = np.concatenate((non_conf_label_arr, non_conf_label_arr_du), axis=0)
                            
                #             # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader, threshold)
                #     if len(_data_arr) > 0:
                #         if args.expand_tar_dataset:
                #             tar_exp_data = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size, _label_arr, _prob_arr)
                #             tar_ext_data_arr, tar_ext_gt_arr, _label_arr, _prob_arr = tar_exp_data["_data_arr"],  tar_exp_data["_gt_arr"], tar_exp_data["_label_arr"], tar_exp_data["_prob_arr"] 
                #         else:
                #             tar_ext_data_arr, tar_ext_gt_arr = _data_arr, _gt_arr
                        
                #         # tar_ext_data_arr = np.concatenate((_data_arr, non_conf_data_arr), axis=0)
                #         # tar_ext_gt_arr = np.concatenate((_gt_arr, non_conf_label_arr), axis=0)
                #         # _label_arr = np.concatenate((_label_arr, non_conf_label_arr), axis=0)
                        
                #         target_wth_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_label_arr), torch.tensor(_prob_arr), torch.tensor(_gt_arr))
                #         # target_wth_labels = TensorDataset(torch.tensor(tar_ext_data_arr), torch.tensor(_label_arr), torch.tensor(tar_ext_gt_arr))
                        
                #         tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_labels, args.batch_size, split=False)
                #         calculate_tar_pl_ce = True
                #         # data_loader2 = tar_loader
                #         # n_batch = min(len(data_loader1), len(tar_loader))

                #         target_for_PL = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #         tar_loader_for_PL, _ = BaseDataset.load_target_data(target_for_PL, args.batch_size, split=False)
                

        model.train()
        
        count = 0
        total_mmd = 0
        srcs_avg_features = []
        clusters_by_label = {}
        feat_memory_bank = {}
        feat_memory_bank_tar = {}
        gt_arr = []
        tar_arr = []
        # tar_pl_batch = len(_data_arr)/args.batch_size
        tar_counter = 0
        
        # conf_tar_data_iter = TragetRestartableIterator(tar_loader)
        tar_data_iter = TragetRestartableIterator(data_loader2)
        n_batch = min(len(data_loader1), len(prev_src_loader))

        for (domain1, prev_src_domain) in zip(data_loader1, prev_src_loader):
            # data_source, label_source = src
            # data_target, _ = tar
            # data_source, label_source = data_source.cuda(), label_source.cuda()
            # data_target = data_target.cuda()
            count = count + 1  

            ### --- *** defining for conf target samples
            # domain2 = next(conf_tar_data_iter)

            ### --- *** Current Source Subject
            data_domain1, label_domain1 = domain1
            data_domain1, label_domain1 = data_domain1.cuda(), label_domain1.cuda()

            ### --- *** Previous Source Subjects
            prev_data_domain, prev_label_domain = prev_src_domain
            prev_data_domain, prev_label_domain = prev_data_domain.cuda(), prev_label_domain.cuda()

            ### --- *** Target Subject
            tar_domain = next(tar_data_iter)
            # tar_data_domain, _ = tar_domain
            # tar_data_domain = tar_data_domain.cuda()

            # -- Generate target PL in minibatch
            if train_source is False and oracle_setting is False:
                for i in range(0,1):
                    # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                    target_wth_gt_labels = TensorDataset(tar_domain[0], tar_domain[1])
                    tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                    conf_data_arr, _prob_arr, pl_label_arr, _gt_arr, _, _ = create_target_pl_dicts(current_forzen_model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)

                    calculate_tar_pl_ce = True
                    if len(conf_data_arr) <= 0:
                        calculate_tar_pl_ce = False
                        break

            
             # defining for domain-2
            if train_source or oracle_setting:
                data_domain2, label_domain2 = tar_domain
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
            else:
                if calculate_tar_pl_ce:
                    data_domain2, label_domain2 = tar_domain
                    conf_data_domain2, conf_label_domain2 = torch.tensor(conf_data_arr).cuda(), torch.tensor(pl_label_arr).cuda()
                else:
                    data_domain2, label_domain2 = tar_domain
                
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
                # print(label_domain2)
                
                # data_domain2, _ = domain2
                # data_domain2 = data_domain2.cuda()

            # macs, params =profile(model, inputs=(data_domain1.float(), data_domain2.float()))

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            data_domain1 = data_domain1.float()
            data_domain2 = data_domain2.float()
            prev_data_domain = prev_data_domain.float()
            prev_label_domain = prev_label_domain.to(torch.int64)
            # tar_data_domain = tar_data_domain.float()

            # store feature vector and label into an feature_memory_bank
            # domain1_feature = model.forward_features(data_domain1)
            # domain2_feature = model.forward_features(data_domain2)
            
            # # # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((domain1_feature.cpu().detach().numpy(), label_domain1.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((domain1_feature.cpu().detach().numpy(), label_domain1.cpu().detach().numpy()))
            
            # feat_memory_bank = np.concatenate((feat_memory_bank, domain1_feature.cpu().detach().numpy()), axis=0) if len(feat_memory_bank) > 0 else domain1_feature.cpu().detach().numpy()
            # gt_arr.extend(label_domain1.detach().cpu().numpy())

            # feat_memory_bank_tar = np.concatenate((feat_memory_bank_tar, domain2_feature.cpu().detach().numpy()), axis=0) if len(feat_memory_bank_tar) > 0 else domain2_feature.cpu().detach().numpy()
            # tar_arr.extend(np.full(16, 5))
            # tar_arr.extend(label_domain2.detach().cpu().numpy())

            # for i in range(0, n_class):
            #     indices = [index for index, label in enumerate(label_domain1) if label == i]
            #     indices2 = [index for index, label in enumerate(label_domain2) if label == i]
            #     intra_class_feat_d1 = data_domain1[indices]
            #     intra_class_feat_d2 = data_domain2[indices2]
            #     intra_label = label_domain1[indices]

            #     if len(data_domain2[indices2]) > 0 and len(data_domain1[indices]) > 0:
            #         label_source_pred, transfer_loss, domain1_feature = model(intra_class_feat_d1, intra_class_feat_d2)
            #         clf_loss = criterion(label_source_pred, intra_label)
            #     elif len(data_domain1[indices]) > 0:
            #         label_source_pred, transfer_loss, domain1_feature = model(intra_class_feat_d1, None)
            #         clf_loss = criterion(label_source_pred, intra_label)
            #     else:
            #         clf_loss = 0
            #         transfer_loss = None

            #     transfer_loss = transfer_loss.detach().item() if transfer_loss is not None and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            #     loss = loss + (clf_loss) + transfer_loss


            # tensor = torch.randn(1, 3, 100, 100)
            # start_t = time.time()
            # model(tensor.cuda().float(), tensor.cuda().float())
            # print("Training Time:")
            # print(start_t)
            # end_t = time.time() - start_t
            # print(end_t)   

            # optimizer.zero_grad()
            ''' 
            Test-1000: 
                only considered target PL for adaptaion (MMD+PL) and does not converge on tar subject
                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2 if calculate_tar_pl_ce else None)
            '''

            # calculate target PL loss only with conf samples
            clf_loss, prev_clf_loss, transfer_loss = 0, 0, 0
            if calculate_tar_pl_ce and count <= len(tar_loader) :
                # calculate mmd loss with prev source sub
                prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
                prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, conf_data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)
                transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            else:
                label_source_pred, prev_transfer_loss, _ = model(data_domain1, prev_data_domain)
                clf_loss = criterion(label_source_pred, label_domain1)

                # added prev source loss always
                # prev_label_source_pred, _, _ = model(prev_data_domain, None)
                # prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

            #----- ALL MMD: calculate mmd loss with prev source sub
            # prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
            # prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

            # label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2)
            # clf_loss = criterion(label_source_pred, label_domain1)
            # transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'

            # loss = (clf_loss) + lamb * transfer_loss
               
            prev_transfer_loss = prev_transfer_loss.detach().item() if prev_transfer_loss and prev_transfer_loss.detach().item() == prev_transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + prev_clf_loss + transfer_loss + (lamb*prev_transfer_loss)

            
            # loss = (clf_loss) 

            total_mmd += transfer_loss 
            # loss.backward()

            # adding target loss with source loss
            if train_source or oracle_setting:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(data_domain2, data_domain1)
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) + lamb * transfer_loss_domain2

                # loss.backward(retain_graph=True) # param: retain_graph=True if wanted to backpropogate two losses separately
                # loss_domain2.backward()

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce and count <= len(tar_loader) :
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(conf_data_domain2, None)
                    clf_loss_domain2 = criterion(label_pred_domain2, conf_label_domain2)
                    loss = clf_loss_domain2 + loss
                    
                    optimizer.zero_grad()
                    loss.backward()

                    tar_counter = tar_counter + 1
                    train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                else:
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() if clf_loss else 0 + prev_clf_loss.detach().item() if prev_clf_loss else 0 + train_loss_clf
            train_loss_transfer = transfer_loss + prev_transfer_loss + train_loss_transfer
            train_loss_total = train_loss_clf + train_loss_transfer
            # train_loss_total = combine_loss.detach().item() + train_loss_total

            # target loss_clf
            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

                # train_loss_clf_domain2 = (clf_loss.detach().item() + clf_loss_domain2.detach().item())/2 + train_loss_clf_domain2
                # # train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                # train_loss_transfer_domain2 = 0
                # train_loss_total_domain2 = combine_loss.detach().item() + train_loss_total_domain2

                # ONLY FOR SOURCE_COMBINE
                # train_loss_clf_domain2 = clf_loss.detach().item() + train_loss_clf_domain2
                # # train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                # train_loss_transfer_domain2 = 0
                # train_loss_total_domain2 = train_loss_total_domain2
        
        # clusters_by_label = make_clusters(feat_memory_bank)
        # cluster_centers = calculate_centroid(clusters_by_label)
        # visualize_feat_PCA_all(clusters_by_label)
        # visualize_tsne(clusters_by_label)
        # visualize_feat_single_PCA(clusters_by_label[0])
        # visualize_feat_single_PCA(clusters_by_label[1])
        # visualize_feat_single_PCA(clusters_by_label[3])
        # visualize_feat_single_PCA(clusters_by_label[4])
        # visualize_feat_single_PCA(clusters_by_label[5])
        # visualize_feat_single_PCA(clusters_by_label[6])
        # train_acc, train_acc_top2 = test(model, tar_loader, args.batch_size, args.is_pain_dataset)
        # acc, acc_top2 = test_4_param(model, tar_loader, args.batch_size, args.is_pain_dataset)
        acc = test(model, val_loader, args.batch_size, False, args.is_pain_dataset)

        # if e % 5 == 0:
        #     plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, gt_arr, None)
        #     plot_tsne_graph_scr_tar(feat_memory_bank_tar, feat_memory_bank_tar, tar_arr, None)
        
        if train_loss_clf_domain2 > 0:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, train_tar_pl_loss_clf: {train_loss_clf_domain2/len(tar_loader):.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        else:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')

        experiment.log_metric("Source-1 loss:", train_loss_clf/n_batch, epoch=e)
        experiment.log_metric("Total loss:", train_loss_total/n_batch, epoch=e)
        if train_loss_clf_domain2 > 0 and len(tar_loader) > 0:
            experiment.log_metric("Target pl train loss:", train_loss_clf_domain2/len(tar_loader))

        experiment.log_metric("Each epoch top 1 accuracy", acc, epoch=e)
        # experiment.log_metric("Each epoch top 2 accuracy", acc_top2)

        # add up all the epochs total losses
        train_total_loss_all_epochs = train_total_loss_all_epochs + train_loss_total/n_batch

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), target_weight_path + '/' + trained_model_name + '.pkl')
            # torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + 'FER_model.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, target_weight_path + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, target_weight_path + '/' + trained_model_name + '_features.pt')
            experiment.log_metric("Val Target Best Accuracy", best_acc, epoch=e)
            stop = 0
        if stop >= early_stop:
            break
    print(total_mmd)
    # _data_arr, _prob_arr, _label_arr, _gt_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)

    # visualize_tsne(clusters_by_label)
    print("Best Val Acc: ", best_acc)
    train_total_loss_all_epochs = train_total_loss_all_epochs/n_epoch
    return model, train_total_loss_all_epochs

def train_multi_model_all_srcs(n_epoch, model, data_loader1, prev_src_loader, data_loader2, optimizer, criterion, lamb, n_batch , early_stop, trained_model_name, val_loader, target_weight_path, oracle_setting, train_source):
    best_acc = 0
    stop = 0
    srcs_avg_features = []
    clusters_by_label = {}
    feat_memory_bank = {}
    threshold = 0.91

    # tar_loader_for_PL = data_loader2
    # tar_loader = data_loader2
    calculate_tar_pl_ce = False
    train_total_loss_all_epochs = 0

    # SupConCriterion = SupConLoss()
    current_forzen_model = copy.deepcopy(model)

    # train_source = False # remove this line ; ITS ONLY THERE TO PERFORM EXPERIMENTS ON THE MMD LOSS FOR DOMAIN SHIFT FOR GDA FOR DGA-1033 PRESENTATION
    for e in range(n_epoch):
        stop += 1
        train_loss_clf, train_loss_transfer, train_loss_total = 0, 0, 0
        train_loss_clf_domain2, train_loss_transfer_domain2, train_loss_total_domain2 = 0, 0, 0
        # calculate_tar_pl_ce = False

        if train_source is False and oracle_setting is False:
            if e % 20 == 0: # remove e != 0 and
                # create a copy of a previously trained model to generate target PL
                current_forzen_model = copy.deepcopy(model)

                print("\n**** Threshold Reduced From: ", threshold)
                threshold = threshold - 0.01      
                print(" To: ", threshold)

                # ***** replace tar_loader_for_PL with data_loader2 to load all the samples everytime it calculates target PL
                # _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr, non_conf_label_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)
                # # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader_for_PL, threshold)
                # # if len(_data_arr) <= 0:
                # #     threshold = threshold - 0.06
                # if len(_data_arr) > 0:
                #     for i in range(0,1):
                #         if len(_data_arr) > 0:
                #             # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                #             target_wth_gt_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #             tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                #             _data_arr, _prob_arr, _label_arr, _gt_arr, non_conf_data_arr_du, non_conf_label_arr_du = create_target_pl_dicts(model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)
                #             # non_conf_data_arr = np.concatenate((non_conf_data_arr, non_conf_data_arr_du), axis=0)
                #             # non_conf_label_arr = np.concatenate((non_conf_label_arr, non_conf_label_arr_du), axis=0)
                            
                #             # _data_arr, _prob_arr, _label_arr, _gt_arr = generate_tar_aug_conf_pl(model, tar_loader, threshold)
                #     if len(_data_arr) > 0:
                #         if args.expand_tar_dataset:
                #             tar_exp_data = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size, _label_arr, _prob_arr)
                #             tar_ext_data_arr, tar_ext_gt_arr, _label_arr, _prob_arr = tar_exp_data["_data_arr"],  tar_exp_data["_gt_arr"], tar_exp_data["_label_arr"], tar_exp_data["_prob_arr"] 
                #         else:
                #             tar_ext_data_arr, tar_ext_gt_arr = _data_arr, _gt_arr
                        
                #         # tar_ext_data_arr = np.concatenate((_data_arr, non_conf_data_arr), axis=0)
                #         # tar_ext_gt_arr = np.concatenate((_gt_arr, non_conf_label_arr), axis=0)
                #         # _label_arr = np.concatenate((_label_arr, non_conf_label_arr), axis=0)
                        
                #         target_wth_labels = TensorDataset(torch.tensor(_data_arr), torch.tensor(_label_arr), torch.tensor(_prob_arr), torch.tensor(_gt_arr))
                #         # target_wth_labels = TensorDataset(torch.tensor(tar_ext_data_arr), torch.tensor(_label_arr), torch.tensor(tar_ext_gt_arr))
                        
                #         tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_labels, args.batch_size, split=False)
                #         calculate_tar_pl_ce = True
                #         # data_loader2 = tar_loader
                #         # n_batch = min(len(data_loader1), len(tar_loader))

                #         target_for_PL = TensorDataset(torch.tensor(_data_arr), torch.tensor(_gt_arr))
                #         tar_loader_for_PL, _ = BaseDataset.load_target_data(target_for_PL, args.batch_size, split=False)
                

        model.train()

        count = 0
        total_mmd = 0
        srcs_avg_features = []
        clusters_by_label = {}
        feat_memory_bank = {}
        feat_memory_bank_tar = {}
        gt_arr = []
        tar_arr = []
        # tar_pl_batch = len(_data_arr)/args.batch_size
        tar_counter = 0
        
        # conf_tar_data_iter = TragetRestartableIterator(tar_loader)
        src_data_iter = TragetRestartableIterator(data_loader1)
        tar_data_iter = TragetRestartableIterator(data_loader2)
        n_batch = len(prev_src_loader)

        for prev_data_domain, prev_label_domain in tqdm(prev_src_loader, leave=False):
            count = count + 1  

            ### --- *** defining for conf target samples
            # domain2 = next(conf_tar_data_iter)

            ### --- *** Current Source Subject
            domain1 = next(src_data_iter)
            data_domain1, label_domain1 = domain1
            data_domain1, label_domain1 = data_domain1.cuda(), label_domain1.cuda()

            ### --- *** Previous Source Subjects
            prev_data_domain, prev_label_domain = prev_data_domain.cuda(), prev_label_domain.cuda()

            ### --- *** Target Subject
            tar_domain = next(tar_data_iter)
            # tar_data_domain, _ = tar_domain
            # tar_data_domain = tar_data_domain.cuda()

            # -- Generate target PL in minibatch
            if train_source is False and oracle_setting is False:
                for i in range(0,1):
                    # _data_arr, tar_exp_gt_labels, _, _ = BaseDataset.expand_target_dataset(_data_arr, _gt_arr, data_loader1, args.batch_size) if args.expand_tar_dataset else _data_arr, _gt_arr, _, _
                    target_wth_gt_labels = TensorDataset(tar_domain[0], tar_domain[1])
                    tar_loader, tar_val_loader = BaseDataset.load_target_data(target_wth_gt_labels, args.batch_size, split=False)
                    conf_data_arr, _prob_arr, pl_label_arr, _gt_arr, _, _ = create_target_pl_dicts(current_forzen_model, tar_loader, threshold, args.batch_size, args.is_pain_dataset)

                    calculate_tar_pl_ce = True
                    if len(conf_data_arr) <= 0:
                        calculate_tar_pl_ce = False
                        break

             # defining for domain-2
            if train_source or oracle_setting:
                data_domain2, label_domain2 = tar_domain
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
            else:
                if calculate_tar_pl_ce:
                    data_domain2, label_domain2 = tar_domain
                    conf_data_domain2, conf_label_domain2 = torch.tensor(conf_data_arr).cuda(), torch.tensor(pl_label_arr).cuda()
                else:
                    data_domain2, label_domain2 = tar_domain
                
                data_domain2, label_domain2 = data_domain2.cuda(), label_domain2.cuda()
                # print(label_domain2)
                
                # data_domain2, _ = domain2
                # data_domain2 = data_domain2.cuda()

            # macs, params =profile(model, inputs=(data_domain1.float(), data_domain2.float()))

            # for training the custom dataset (PAIN DATASETS); I have added .float() otherwise removed it when using build-in dataset
            data_domain1 = data_domain1.float()
            data_domain2 = data_domain2.float()
            prev_data_domain = prev_data_domain.float()

            ''' 
            Test-1000: 
                only considered target PL for adaptaion (MMD+PL) and does not converge on tar subject
                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2 if calculate_tar_pl_ce else None)
            '''

            # calculate target PL loss only with conf samples
            clf_loss, prev_clf_loss, transfer_loss = 0, 0, 0
            if calculate_tar_pl_ce and count <= len(tar_loader) :
                # calculate mmd loss with prev source sub
                prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
                prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

                label_source_pred, transfer_loss, domain1_feature = model(data_domain1, conf_data_domain2)
                clf_loss = criterion(label_source_pred, label_domain1)
                transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'
            else:
                label_source_pred, prev_transfer_loss, _ = model(data_domain1, prev_data_domain)
                clf_loss = criterion(label_source_pred, label_domain1)

            # calculate mmd loss with prev source sub
            # prev_label_source_pred, prev_transfer_loss, _ = model(prev_data_domain, data_domain1)
            # prev_clf_loss = criterion(prev_label_source_pred, prev_label_domain)

            # label_source_pred, transfer_loss, domain1_feature = model(data_domain1, data_domain2)
            # clf_loss = criterion(label_source_pred, label_domain1)
            # transfer_loss = transfer_loss.detach().item() if transfer_loss and transfer_loss.detach().item() == transfer_loss.detach().item() else 0 # to avoid 'NaN'

            # loss = (clf_loss) + lamb * transfer_loss
               
            prev_transfer_loss = prev_transfer_loss.detach().item() if prev_transfer_loss and prev_transfer_loss.detach().item() == prev_transfer_loss.detach().item() else 0 # to avoid 'NaN'
            loss = (clf_loss) + prev_clf_loss + transfer_loss + (lamb*prev_transfer_loss)

            total_mmd += transfer_loss 
            # loss.backward()

            # adding target loss with source loss
            if train_source or oracle_setting:
                label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(data_domain2, data_domain1)
                clf_loss_domain2 = criterion(label_pred_domain2, label_domain2)
                loss_domain2 = (clf_loss_domain2) + lamb * transfer_loss_domain2

                # loss.backward(retain_graph=True) # param: retain_graph=True if wanted to backpropogate two losses separately
                # loss_domain2.backward()

                """
                    combine source-1 and source-2 loss
                """
                # combine_loss = loss_domain2 + loss
                combine_loss = clf_loss_domain2 + loss
                optimizer.zero_grad()
                combine_loss.backward()

                if domain1_feature.shape == domain2_feature.shape:
                    srcs_avg_features.append(torch.mean(torch.stack([domain1_feature, domain2_feature]), dim=0))

            else:
                if calculate_tar_pl_ce and count <= len(tar_loader) :
                    label_pred_domain2, transfer_loss_domain2, domain2_feature  = model(conf_data_domain2, None)
                    clf_loss_domain2 = criterion(label_pred_domain2, conf_label_domain2)
                    loss = clf_loss_domain2 + loss
                    
                    optimizer.zero_grad()
                    loss.backward()

                    tar_counter = tar_counter + 1
                    train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                else:
                    optimizer.zero_grad()
                    loss.backward()
                
            optimizer.step()
            train_loss_clf = clf_loss.detach().item() if clf_loss else 0 + prev_clf_loss.detach().item() if prev_clf_loss else 0 + train_loss_clf
            train_loss_transfer = transfer_loss + prev_transfer_loss + train_loss_transfer
            train_loss_total = train_loss_clf + train_loss_transfer
            # train_loss_total = combine_loss.detach().item() + train_loss_total

            # target loss_clf
            if train_source:
                train_loss_clf_domain2 = clf_loss_domain2.detach().item() + train_loss_clf_domain2
                train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                train_loss_total_domain2 = loss_domain2.detach().item() + train_loss_total_domain2

                # train_loss_clf_domain2 = (clf_loss.detach().item() + clf_loss_domain2.detach().item())/2 + train_loss_clf_domain2
                # # train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                # train_loss_transfer_domain2 = 0
                # train_loss_total_domain2 = combine_loss.detach().item() + train_loss_total_domain2

                # ONLY FOR SOURCE_COMBINE
                # train_loss_clf_domain2 = clf_loss.detach().item() + train_loss_clf_domain2
                # # train_loss_transfer_domain2 = transfer_loss_domain2.detach().item() + train_loss_transfer_domain2
                # train_loss_transfer_domain2 = 0
                # train_loss_total_domain2 = train_loss_total_domain2
        
        # clusters_by_label = make_clusters(feat_memory_bank)
        # cluster_centers = calculate_centroid(clusters_by_label)
        # visualize_feat_PCA_all(clusters_by_label)
        # visualize_tsne(clusters_by_label)
        # visualize_feat_single_PCA(clusters_by_label[0])
        # visualize_feat_single_PCA(clusters_by_label[1])
        # visualize_feat_single_PCA(clusters_by_label[3])
        # visualize_feat_single_PCA(clusters_by_label[4])
        # visualize_feat_single_PCA(clusters_by_label[5])
        # visualize_feat_single_PCA(clusters_by_label[6])
        # train_acc, train_acc_top2 = test(model, tar_loader, args.batch_size, args.is_pain_dataset)
        # acc, acc_top2 = test_4_param(model, tar_loader, args.batch_size, args.is_pain_dataset)
        acc = test(model, val_loader, args.batch_size, False, args.is_pain_dataset)

        # if e % 5 == 0:
        #     plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, gt_arr, None)
        #     plot_tsne_graph_scr_tar(feat_memory_bank_tar, feat_memory_bank_tar, tar_arr, None)
        
        if train_loss_clf_domain2 > 0:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, train_tar_pl_loss_clf: {train_loss_clf_domain2/len(tar_loader):.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')
        else:
            print(f'Epoch: [{e:2d}/{n_epoch}], train_src_loss_clf: {train_loss_clf/n_batch:.4f}, transfer_loss: {train_loss_transfer/n_batch:.4f}, total_Loss: {train_loss_total/n_batch:.4f}, acc: {acc:.4f}')

        experiment.log_metric("Source-1 loss:", train_loss_clf/n_batch, epoch=e)
        experiment.log_metric("Total loss:", train_loss_total/n_batch, epoch=e)
        if train_loss_clf_domain2 > 0 and len(tar_loader) > 0:
            experiment.log_metric("Target pl train loss:", train_loss_clf_domain2/len(tar_loader))

        experiment.log_metric("Each epoch top 1 accuracy", acc, epoch=e)
        # experiment.log_metric("Each epoch top 2 accuracy", acc_top2)

        # add up all the epochs total losses
        train_total_loss_all_epochs = train_total_loss_all_epochs + train_loss_total/n_batch

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), target_weight_path + '/' + trained_model_name + '.pkl')
            # torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + 'FER_model.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, target_weight_path + '/' + trained_model_name + '_load.pt')
            # save source feature maps
            if len(srcs_avg_features) > 0:
                torch.save(srcs_avg_features, target_weight_path + '/' + trained_model_name + '_features.pt')
            experiment.log_metric("Val Target Best Accuracy", best_acc, epoch=e)
            stop = 0
        if stop >= early_stop:
            break
    print(total_mmd)
    # _data_arr, _prob_arr, _label_arr, _gt_arr = create_target_pl_dicts(model, tar_loader_for_PL, threshold, args.batch_size, args.is_pain_dataset)

    # visualize_tsne(clusters_by_label)
    print("Best Val Acc: ", best_acc)
    train_total_loss_all_epochs = train_total_loss_all_epochs/n_epoch
    return model, train_total_loss_all_epochs

def measure_src_samples_tar_dist(model, srcs_sub_loader, tar_sub_loader, dist_measure, tar_name, batch_size):
    model.eval()

    mmd = MMD_loss()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    total_dist = 0
    tar_mean = 0.0
    concat_features = []
    closest_src_samples = []
    closest_src_labels = []
    closest_src_dists = []
    relv_sample_count = 50000
    with torch.no_grad():
        tar_data_iter = TragetRestartableIterator(tar_sub_loader)
        n_batch = len(srcs_sub_loader)

        for src_data, src_gt in tqdm(srcs_sub_loader, leave=False):
            # count = count + 1  

            target = next(tar_data_iter)

            # src_data, src_gt = sources
            src_data, src_gt = src_data.cuda(), src_gt.cuda()
            src_data = src_data.float()
            src_feat = model.forward_features(src_data)

            tar_data, _ = target
            tar_data = tar_data.cuda()
            tar_data = tar_data.float()
            tar_feat = model.forward_features(tar_data)

            # concat_features = torch.cat([concat_features, tar_feat], dim=0) if len(concat_features) > 0 else tar_feat 

            if dist_measure == config.MMD_SIMILARITY:
                mmd_dist = mmd(src_feat, tar_feat)
                total_dist = total_dist + mmd_dist.cpu().numpy()
            elif dist_measure == config.COSINE_SIMILARITY:
                cosine_dist = cosine_sim(src_feat, tar_feat)
                # sim = cosine_sim(src_feat, tar_feat[0].unsqueeze(0).expand(src_feat.shape[0], -1))

            closest_src_samples.extend(src_data.detach().cpu().numpy())
            closest_src_labels.extend(src_gt.detach().cpu().numpy())
            closest_src_dists.extend(cosine_dist.detach().cpu().numpy())


    np.savez(tar_name, closest_src_samples=closest_src_samples, closest_src_labels=closest_src_labels, closest_src_dists=closest_src_dists)

    # batches = min(len(srcs_sub_loader), len(tar_sub_loader))
    # total_dist = total_dist / batches
    # print(total_dist)
    # return closest_src_samples, closest_src_labels

    # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

def measure_srcs_tar_dist(model, srcs_sub_loader, tar_sub_loader, dist_measure, batch_size):
    model.eval()

    mmd = MMD_loss()
    cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    total_dist = 0
    tar_mean = 0.0
    concat_features = []
    with torch.no_grad():
        for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
            src_data, src_gt = sources
            src_data, src_gt = src_data.cuda(), src_gt.cuda()
            src_data = src_data.float()
            src_feat = model.forward_features(src_data)

            tar_data, _ = target
            tar_data = tar_data.cuda()
            tar_data = tar_data.float()
            tar_feat = model.forward_features(tar_data)

            concat_features = torch.cat([concat_features, tar_feat], dim=0) if len(concat_features) > 0 else tar_feat 

            if dist_measure == config.MMD_SIMILARITY:
                mmd_dist = mmd(src_feat, tar_feat)
                total_dist = total_dist + mmd_dist.cpu().numpy()
            elif dist_measure == config.COSINE_SIMILARITY:
                cosine_dist = cosine_sim(src_feat, tar_feat)
                total_dist = total_dist + (np.sum(cosine_dist.cpu().detach().numpy()))/batch_size
            else:
                tensor1_probs = F.softmax(src_feat, dim=1)
                tensor2_probs = F.softmax(tar_feat, dim=1)
                total_dist = total_dist + F.kl_div(tensor1_probs.log(), tensor2_probs, reduction='batchmean')

    # print(cosine_dist)
    batches = min(len(srcs_sub_loader), len(tar_sub_loader))
    total_dist = total_dist / batches
    print(total_dist)
    return total_dist

    # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

# def create_relv_src_dic(srcs_sub_loader, tar_sub_loader, model, relv_feat_dic, relv_sample_dic, relv_label_dic, relv_sample_count):
#     model.eval()

#     concat_tar_features = []
#     concat_src_features = torch.tensor(relv_feat_dic.tolist()).to(device) if len(relv_feat_dic) > 0 else []
#     relv_src_data = relv_sample_dic.tolist() if len(relv_sample_dic) > 0 else []
#     relv_src_label = relv_label_dic.tolist() if len(relv_label_dic) > 0 else []
#     with torch.no_grad():
#         for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
#             src_data, src_label = sources
#             src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
#             src_feat = model.forward_features(src_data)
#             concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
#             # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data
#             relv_src_data.extend(src_data.detach().cpu().numpy())
#             relv_src_label.extend(src_label.detach().cpu().numpy())

#             tar_data, _ = target
#             tar_data = tar_data.cuda().float()
#             tar_feat = model.forward_features(tar_data)
#             concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

#     mean_feature = torch.mean(concat_tar_features, dim=0)
#     src_dist = torch.norm(concat_src_features - mean_feature, dim=1)
#     sorted_src_dist, sorted_src_indices = torch.sort(src_dist)
#     relv_indices = sorted_src_indices[:relv_sample_count]

#     # include current samples into a relv dic
#     # relv_sample_dic = torch.cat([relv_sample_dic, src_dist], dim=0) if len(src_dist) > 0 else src_dist

#     relv_src_data_arr = np.array(relv_src_data)
#     relv_src_data_arr = relv_src_data_arr[relv_indices.tolist()]
#     relv_src_label_arr = np.array(relv_src_label)
#     relv_src_label_arr = relv_src_label_arr[relv_indices.tolist()]

#     relv_feat_arr = np.array(concat_src_features.cpu().numpy())
#     relv_feat_arr = relv_feat_arr[relv_indices.tolist()]

#     # relv_src_samples = {i.item(): relv_src_data[i].cpu().numpy() for i in relv_indices}

#     return relv_feat_arr, relv_src_data_arr, relv_src_label_arr

# def create_relv_src_clusters(srcs_sub_loader, tar_sub_loader, model, relv_feat_dic, relv_sample_dic, relv_label_dic, relv_sample_count, tar_name, is_before_adapt=False):
#     model.eval()

#     concat_tar_features = []
#     curr_src_features = []
#     prev_data_feat = []
#     concat_src_features = torch.tensor(relv_feat_dic.tolist()).to(device) if len(relv_feat_dic) > 0 else []
#     relv_src_data = relv_sample_dic.tolist() if len(relv_sample_dic) > 0 else []
#     relv_src_label = relv_label_dic.tolist() if len(relv_label_dic) > 0 else []

#     # count = 0
#     # for prev_src in prev_srcs_loader_dic:
#     #     if len(prev_srcs_loader_dic) == count:
#     #         prev_data_iter = TragetRestartableIterator(prev_src)
#     #     count = count + 1

#     with torch.no_grad():
#         for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
#             src_data, src_label = sources
#             src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
#             src_feat = model.forward_features(src_data)
#             curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 
#             concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
#             # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data
#             relv_src_data.extend(src_data.detach().cpu().numpy())
#             relv_src_label.extend(src_label.detach().cpu().numpy())

#             # prev_data = model.forward_features(prev_data_iter)
#             # prev_data_feat = torch.cat([prev_data_feat, prev_data], dim=0) if len(prev_data_feat) > 0 else prev_data 

#             tar_data, _ = target
#             tar_data = tar_data.cuda().float()
#             tar_feat = model.forward_features(tar_data)
#             concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

#     # Define the number of clusters
#     n_clusters = 2

#     # Initialize KMeans
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
#     concat_tar_features = concat_tar_features.cpu().numpy()
#     concat_src_features = concat_src_features.cpu().numpy() # combining all the relevant + new src sb
#     curr_src_features = curr_src_features.cpu().numpy()

#     prev_src_feat = relv_feat_dic if len(relv_feat_dic) > 0 else curr_src_features[:relv_sample_count]


#     # if is_only_tsne:
#     # tsne_graph(concat_tar_features, curr_src_features.cpu().numpy())
#     # return relv_feat_dic, relv_sample_dic, relv_label_dic

#     # combined_feat = np.vstack([concat_tar_features, concat_src_features])

#     # Fit the model and get the cluster labels
#     kmeans.fit(concat_tar_features)
#     labels = kmeans.labels_
#     # Get the cluster centers
#     centroids = kmeans.cluster_centers_

#     print("\nTarget Centroid:", centroids)


#     # Calculate/Re-calculating the distances of relv src dic with the target cluster centroid
#     closest_centroids, distances = pairwise_distances_argmin_min(concat_src_features, centroids)
#     # Sort distances and get feature indices
#     sorted_indices = np.argsort(distances)
#     # get the sorted top N previous src samples indices
#     relv_indices = sorted_indices[:relv_sample_count].tolist()
#     relv_src_data_arr = np.array(relv_src_data)
#     relv_src_data_arr = relv_src_data_arr[relv_indices]
#     relv_src_label_arr = np.array(relv_src_label)
#     relv_src_label_arr = relv_src_label_arr[relv_indices]

#     relv_feat_arr = np.array(concat_src_features)
#     relv_feat_arr = relv_feat_arr[relv_indices]

#     # relv_src_samples = {i.item(): relv_src_data[i].cpu().numpy() for i in relv_indices}

#     # Print the distances
#     # for i, distance in enumerate(distances):
#     #     print(f"Data point {i} is {distance:.4f} units away from its cluster centroid.")

#     # combined_data = np.vstack([concat_tar_features, centroids, concat_src_features])
#     # combined_data = np.vstack([concat_tar_features, prev_data_feat, concat_src_features])
#     # combined_data = np.vstack([concat_tar_features, prev_src_feat, curr_src_features, centroids])
#     # updated_relv_data = np.vstack([concat_tar_features, relv_feat_arr, curr_src_features, centroids])

#     plot_tsne_cluster(n_clusters, labels, True, tar_name, concat_tar_features, prev_src_feat, curr_src_features, centroids)
#     plot_tsne_cluster(n_clusters, labels, False, tar_name, concat_tar_features, relv_feat_arr, curr_src_features, centroids)
    

#     # Combine original features, second features, and centroids for t-SNE
#     # all_data = np.vstack([combined_feat, centroids])

#     # Reduce dimensionality using t-SNE
#     # tsne = TSNE(n_components=2, random_state=42)
#     # combined_2d = tsne.fit_transform(combined_data)

# ## ----------------------------------- -----------------------------------------##
# # Separate the transformed data back into original features, second features, and centroids
#     # original_features_2d = combined_2d[:concat_tar_features.shape[0]]
#     # second_features_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + concat_src_features.shape[0]]
#     # centroids_2d = combined_2d[concat_tar_features.shape[0] + concat_src_features.shape[0]:]

#     # # Plot the t-SNE visualization
#     # plt.figure(figsize=(12, 8))

#     # # Plot the original features with circles
#     # plt.scatter(original_features_2d[:, 0], original_features_2d[:, 1], c=labels[:concat_tar_features.shape[0]], marker='o', alpha=0.6, label='Original Features')

#     # # Plot the second features with squares
#     # plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c=labels[concat_tar_features.shape[0]:], marker='s', alpha=0.6, label='Second Features')

#     # # Plot the centroids with a distinct marker
#     # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')
# # ------------------------------------ --------------------------------------##
#     # features_2d = tsne.fit_transform(concat_tar_features.cpu().numpy())

#     # Separate the transformed features and centroids
#     # *** Giving equal size to every features BCZ of same data in all three feature map. Note: change this in case of selecting diff prev top-N
#     # features_2d = combined_2d[:concat_tar_features.shape[0]]
#     # features_2d = combined_2d[:combined_feat.shape[0]]
#     # centroids_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + centroids.shape[0]]
#     # second_features_2d = combined_2d[concat_tar_features.shape[0] + centroids.shape[0]:]

#     # prev_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0]]
#     # second_features_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0] + concat_tar_features.shape[0]]
#     # centroids_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]+concat_tar_features.shape[0]:]
    

#     # Plot the t-SNE visualization
#     # plt.figure(figsize=(10, 8))
#     # for i in range(n_clusters):
#     #     # Select points belonging to the current cluster
#     #     cluster_points = features_2d[labels == i]
#     #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Target Cluster {i}', alpha=0.6)

#     # Plot the centroids
#     # Plot the centroids in a distinct color and marker
#     # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, label='Centroids')

#     # plt.scatter(prev_2d[:, 0], prev_2d[:, 1], c='green', marker='o', alpha=0.3, label='Previous Relevent Features')

#     # Plot the second set of features in a different color
#     # plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c='gray', marker='o', alpha=0.3, label='Second Features')

#     # plt.title('t-SNE Visualization of Clusters')
#     # plt.xlabel('t-SNE Component 1')
#     # plt.ylabel('t-SNE Component 2')
#     # plt.legend()
#     # tsne_folder = 'relv_samples_clusters/'+ tar_name.split('/')[0]
#     # if not os.path.exists(tsne_folder):
#     #     os.makedirs(tsne_folder)

#     # file_name = tsne_folder+'/'+tar_name.split('/')[1]+'_BE.png' if is_before_adapt else tsne_folder+'/'+tar_name.split('/')[1]+'AF_.png'
#     # plt.savefig(file_name)
#     # plt.show()
    
#     return relv_feat_arr, relv_src_data_arr, relv_src_label_arr

# def create_relv_src_clus_cent(srcs_sub_loader, tar_sub_loader, model, prev_relv_samples_struc, relv_sample_count, tar_name, is_before_adapt=False):
#     model.eval()

#     concat_tar_features = []
#     curr_src_features = []
#     curr_src_data = []
#     curr_src_label = []
#     updated_relv_samples_struc = None
#     # concat_src_features = torch.tensor(relv_feat_dic.tolist()).to(device) if len(relv_feat_dic) > 0 else []

#     with torch.no_grad():
#         for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
#             src_data, src_label = sources
#             src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
#             src_feat = model.forward_features(src_data)
#             curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 
#             # concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
#             # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data
#             curr_src_data.extend(src_data.detach().cpu().numpy())
#             curr_src_label.extend(src_label.detach().cpu().numpy())

#             # relv_src_data.extend(src_data.detach().cpu().numpy())
#             # relv_src_label.extend(src_label.detach().cpu().numpy())

#             # prev_data = model.forward_features(prev_data_iter)
#             # prev_data_feat = torch.cat([prev_data_feat, prev_data], dim=0) if len(prev_data_feat) > 0 else prev_data 

#             tar_data, _ = target
#             tar_data = tar_data.cuda().float()
#             tar_feat = model.forward_features(tar_data)
#             concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 

#     # Define the number of clusters
#     n_clusters = 2

#     concat_tar_features = concat_tar_features.cpu().numpy()
#     # concat_src_features = concat_src_features.cpu().numpy() # combining all the relevant + new src sb
#     curr_src_features = curr_src_features.cpu().numpy()

#     # if is_only_tsne:
#     # tsne_graph(concat_tar_features, curr_src_features.cpu().numpy())
#     # return relv_feat_dic, relv_sample_dic, relv_label_dic

#     # combined_feat = np.vstack([concat_tar_features, concat_src_features])

#     # Fit the model and get the cluster labels
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
#     kmeans.fit(concat_tar_features)
#     labels = kmeans.labels_ 
#     centroids = kmeans.cluster_centers_

#     # print("\nTarget Centroid:", centroids)
#     kmeans_curr_src = KMeans(n_clusters=n_clusters, random_state=42, algorithm='lloyd')
#     kmeans_curr_src.fit(curr_src_features)
#     labels_curr_src = kmeans_curr_src.labels_
#     centroids_curr_src = kmeans_curr_src.cluster_centers_

#     # Step 3: Calculate distances with every sample in new subject with its own centroids
#     B_distances = cdist(curr_src_features, centroids_curr_src, 'euclidean')
#     B_min_distances = B_distances.min(axis=1) # it picks the smallest distance of every point to the centroids 
#     # Include features, given labels (B_L), and K-means labels in B_results
#     B_results = [(curr_src_data[i], curr_src_features[i], curr_src_label[i], B_min_distances[i]) for i in range(len(curr_src_data))]
#     B_sorted = sorted(B_results, key=lambda x: x[3])  # Sort by distance (now at index 4)

#     # calculate the distance with the target centroid 
#     distances_from_A = cdist([result[1] for result in B_sorted[:len(B_sorted)]], centroids, 'euclidean')
#     distances_from_A = distances_from_A.min(axis=1)
#     new_relv_samples_struc = [(B_sorted[i][0], B_sorted[i][1], B_sorted[i][2], distances_from_A[i]) for i in range(len(B_sorted))]

#     '''
#         Add up based on the distances from target centroid Prev Samples + New Samples

#         prev_relv_samples_struc: store prev closest samples
#         new_relv_samples_struc: store new src samples
#         updated_relv_samples_struc: store updated list after combining with new_relv_samples_struc
#     '''
#     if prev_relv_samples_struc:
#         # only taking first 500 samples that are most closer to src centroid and check if its closer to the target subject or not
#         updated_prev_struc = prev_relv_samples_struc + new_relv_samples_struc[:500]
#         updated_relv_samples_struc = sorted(updated_prev_struc, key=lambda x: x[3])

#         prev_src_feat =[point[1] for point in prev_relv_samples_struc if point[1] is not None]
#     else:
#         updated_relv_samples_struc = new_relv_samples_struc[:500]
#         prev_src_feat =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

#     if len(updated_relv_samples_struc) < relv_sample_count:
#         relv_sample_count = len(updated_relv_samples_struc) 

#     relv_feat_arr =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

#     plot_tsne_cluster(n_clusters, labels, True, tar_name, concat_tar_features, np.array(prev_src_feat), curr_src_features, centroids)
#     plot_tsne_cluster(n_clusters, labels, False, tar_name, concat_tar_features, np.array(relv_feat_arr[:relv_sample_count]), curr_src_features, centroids)
    
#     return updated_relv_samples_struc[:relv_sample_count]

# def create_relv_src_clus_dbscan(srcs_sub_loader, tar_sub_loader, model, prev_relv_samples_struc, relv_sample_count, tar_name, is_before_adapt=False):
#     model.eval()

#     concat_tar_features = []
#     curr_src_features = []
#     curr_src_data = []
#     curr_src_label = []

#     # count = 0
#     # for prev_src in prev_srcs_loader_dic:
#     #     if len(prev_srcs_loader_dic) == count:
#     #         prev_data_iter = TragetRestartableIterator(prev_src)
#     #     count = count + 1

#     with torch.no_grad():
#         for (sources, target) in zip(srcs_sub_loader, tar_sub_loader):
#             src_data, src_label = sources
#             src_data, src_label = src_data.cuda().float(), src_label.cuda().float()
#             src_feat = model.forward_features(src_data)
#             curr_src_features = torch.cat([curr_src_features, src_feat], dim=0) if len(curr_src_features) > 0 else src_feat 
#             # concat_src_features = torch.cat([concat_src_features, src_feat], dim=0) if len(concat_src_features) > 0 else src_feat 
#             # relv_src_data = torch.cat([relv_src_data, src_data], dim=0) if len(relv_src_data) > 0 else src_data

#             curr_src_data.extend(src_data.detach().cpu().numpy())
#             curr_src_label.extend(src_label.detach().cpu().numpy())

#             # prev_data = model.forward_features(prev_data_iter)
#             # prev_data_feat = torch.cat([prev_data_feat, prev_data], dim=0) if len(prev_data_feat) > 0 else prev_data 

#             tar_data, _ = target
#             tar_data = tar_data.cuda().float()
#             tar_feat = model.forward_features(tar_data)
#             concat_tar_features = torch.cat([concat_tar_features, tar_feat], dim=0) if len(concat_tar_features) > 0 else tar_feat 
    
#     concat_tar_features = concat_tar_features.cpu().numpy()
#     # concat_src_features = concat_src_features.cpu().numpy() # combining all the relevant + new src sb
#     curr_src_features = curr_src_features.cpu().numpy()
    
#     tar_centroids_dbscan, tar_dbscan_clusters, tar_labels_dbscan = apply_dbscan(concat_tar_features)
#     src_centroids_dbscan, src_dbscan_clusters, src_labels_dbscan = apply_dbscan(curr_src_features)
        
#     # Step 3: Calculate distances with every sample in new subject with its own centroids
#     B_distances = cdist(curr_src_features, src_centroids_dbscan, 'euclidean')
#     B_min_distances = B_distances.min(axis=1) # it picks the smallest distance of every point to the centroids 
#     # Include features, given labels (B_L), and K-means labels in B_results
#     B_results = [(curr_src_data[i], curr_src_features[i], curr_src_label[i], B_min_distances[i]) for i in range(len(curr_src_data))]
#     B_sorted = sorted(B_results, key=lambda x: x[3])  # Sort by distance (now at index 4)

#     # calculate the distance with the target centroid 
#     distances_from_A = cdist([result[1] for result in B_sorted[:len(B_sorted)]], tar_centroids_dbscan, 'euclidean')
#     distances_from_A = distances_from_A.min(axis=1)
#     new_relv_samples_struc = [(B_sorted[i][0], B_sorted[i][1], B_sorted[i][2], distances_from_A[i]) for i in range(len(B_sorted))]

#     '''
#         Add up based on the distances from target centroid Prev Samples + New Samples

#         prev_relv_samples_struc: store prev closest samples
#         new_relv_samples_struc: store new src samples
#         updated_relv_samples_struc: store updated list after combining with new_relv_samples_struc
#     '''
#     if prev_relv_samples_struc:
#         # only taking first 500 samples that are most closer to src centroid and check if its closer to the target subject or not
#         updated_prev_struc = prev_relv_samples_struc + new_relv_samples_struc[:500] # biovid=500  , UNBC=100
#         updated_relv_samples_struc = sorted(updated_prev_struc, key=lambda x: x[3])

#         prev_src_feat =[point[1] for point in prev_relv_samples_struc if point[1] is not None]
#     else:
#         updated_relv_samples_struc = new_relv_samples_struc[:500]
#         prev_src_feat =[point[1] for point in updated_relv_samples_struc if point[1] is not None]

#     if len(updated_relv_samples_struc) < relv_sample_count:
#         relv_sample_count = len(updated_relv_samples_struc) 

#     relv_feat_arr =[point[1] for point in updated_relv_samples_struc if point[1] is not None]
#     print("** ** Updated Prev Dic using DBSCAN ** **")

#     plot_dbscan_tsne(tar_dbscan_clusters, src_dbscan_clusters, True, tar_name, concat_tar_features, np.array(prev_src_feat), curr_src_features, tar_centroids_dbscan)
#     plot_dbscan_tsne(tar_dbscan_clusters, src_dbscan_clusters, False, tar_name, concat_tar_features, np.array(relv_feat_arr[:relv_sample_count]), curr_src_features, tar_centroids_dbscan)
    
#     return updated_relv_samples_struc[:relv_sample_count]

# def apply_dbscan(input_features):
#     dbscan_eps = 5.5
#     dbscan_minsam = 4
#     while True:
#         dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minsam)
#         dbscan_clusters = dbscan.fit_predict(input_features)
#         if max(dbscan_clusters) > -1:
#             break
#         else:
#             dbscan_eps = dbscan_eps - 1.0
#             dbscan_minsam = dbscan_minsam - 1
#             print("EPS reduce by 1:: ", dbscan_eps)
#             print("Min Sample reduce by 1: ", dbscan_minsam)

#     labels_dbscan = set(dbscan_clusters)
#     labels_dbscan.discard(-1)
#     centroids_dbscan = []
#     for label in labels_dbscan:
#         cluster_points = input_features[dbscan_clusters == label]
#         centroid = np.mean(cluster_points, axis=0)
#         centroids_dbscan.append(centroid)
    
#     return centroids_dbscan, dbscan_clusters, labels_dbscan

# def plot_dbscan_tsne(tar_labels, src_labels, is_before_adapt, tar_name, concat_tar_features, prev_src_feat, curr_src_features, centroids):
#     combined_data = np.vstack([concat_tar_features, prev_src_feat, curr_src_features, centroids])

#     # Reduce dimensionality using t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     combined_2d = tsne.fit_transform(combined_data)

#     # Separate the transformed features and centroids
#     # *** Giving equal size to every features BCZ of same data in all three feature map. Note: change this in case of selecting diff prev top-N
#     features_2d = combined_2d[:concat_tar_features.shape[0]]
#     prev_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0]]
#     second_features_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0] + concat_tar_features.shape[0]]
#     centroids_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]+concat_tar_features.shape[0]:]
    

#     # Plot the t-SNE visualization
#     plt.figure(figsize=(10, 8))

#     plt.scatter(features_2d[:, 0], features_2d[:, 1], c=tar_labels, cmap='viridis')
#     # Plot the centroids
#     # Plot the centroids in a distinct color and marker
#     plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, label='Centroids')

#     plt.scatter(prev_2d[:, 0], prev_2d[:, 1], c='green', marker='o', alpha=0.3, label='Previous Relevent Features')

#     # Plot the second set of features in a different color
#     if is_before_adapt:
#         plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c='gray', marker='o', alpha=0.3, label='Second Features')

#     plt.title('t-SNE Visualization of Clusters')
#     plt.xlabel('t-SNE Component 1')
#     plt.ylabel('t-SNE Component 2')
#     plt.legend()
#     tsne_folder = 'relv_samples_clusters/'+ tar_name.split('/')[0]
#     if not os.path.exists(tsne_folder):
#         os.makedirs(tsne_folder)

#     file_name = tsne_folder+'/'+tar_name.split('/')[1]+'_BE.png' if is_before_adapt else tsne_folder+'/'+tar_name.split('/')[1]+'AF_.png'
#     plt.savefig(file_name)


def plot_tsne_cluster(n_clusters, labels, is_before_adapt, tar_name, concat_tar_features, prev_src_feat, curr_src_features, centroids):
    combined_data = np.vstack([concat_tar_features, prev_src_feat, curr_src_features, centroids])
    
    # Combine original features, second features, and centroids for t-SNE
    # all_data = np.vstack([combined_feat, centroids])

    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_data)
## ----------------------------------- -----------------------------------------##
# Separate the transformed data back into original features, second features, and centroids
    # original_features_2d = combined_2d[:concat_tar_features.shape[0]]
    # second_features_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + concat_src_features.shape[0]]
    # centroids_2d = combined_2d[concat_tar_features.shape[0] + concat_src_features.shape[0]:]

    # # Plot the t-SNE visualization
    # plt.figure(figsize=(12, 8))

    # # Plot the original features with circles
    # plt.scatter(original_features_2d[:, 0], original_features_2d[:, 1], c=labels[:concat_tar_features.shape[0]], marker='o', alpha=0.6, label='Original Features')

    # # Plot the second features with squares
    # plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c=labels[concat_tar_features.shape[0]:], marker='s', alpha=0.6, label='Second Features')

    # # Plot the centroids with a distinct marker
    # plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroids')
# ------------------------------------ --------------------------------------##
    # features_2d = tsne.fit_transform(concat_tar_features.cpu().numpy())

    # Separate the transformed features and centroids
    # *** Giving equal size to every features BCZ of same data in all three feature map. Note: change this in case of selecting diff prev top-N
    features_2d = combined_2d[:concat_tar_features.shape[0]]
    # features_2d = combined_2d[:combined_feat.shape[0]]
    # centroids_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0] + centroids.shape[0]]
    # second_features_2d = combined_2d[concat_tar_features.shape[0] + centroids.shape[0]:]

    prev_2d = combined_2d[concat_tar_features.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0]]
    second_features_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]:concat_tar_features.shape[0]+prev_src_feat.shape[0] + concat_tar_features.shape[0]]
    centroids_2d = combined_2d[concat_tar_features.shape[0] + prev_src_feat.shape[0]+concat_tar_features.shape[0]:]
    

    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 8))
    # for i in range(n_clusters):
    #     # Select points belonging to the current cluster
    #     cluster_points = features_2d[labels == i]
    #     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Target Cluster {i}', alpha=0.6)

    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis')
    # Plot the centroids
    # Plot the centroids in a distinct color and marker
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=200, label='Centroids')

    plt.scatter(prev_2d[:, 0], prev_2d[:, 1], c='green', marker='o', alpha=0.3, label='Previous Relevent Features')

    # Plot the second set of features in a different color
    if is_before_adapt:
        plt.scatter(second_features_2d[:, 0], second_features_2d[:, 1], c='gray', marker='o', alpha=0.3, label='Second Features')

    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    tsne_folder = 'relv_samples_clusters/'+ tar_name.split('/')[0]
    if not os.path.exists(tsne_folder):
        os.makedirs(tsne_folder)

    file_name = tsne_folder+'/'+tar_name.split('/')[1]+'_BE.png' if is_before_adapt else tsne_folder+'/'+tar_name.split('/')[1]+'AF_.png'
    plt.savefig(file_name)

def create_clusters_from_source_feat(model, combine_srcs_loader):
    model.train()
    clusters_by_label = {}
    feat_memory_bank = {}

    loaded_data = np.load('source_features.npz')
    source_feat = {key: loaded_data[key] for key in loaded_data}
    clusters_by_label = make_clusters(source_feat['arr_0'])
    arr=[]
    arr = np.row_stack(clusters_by_label[5])
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[1])), axis=0)
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[2])), axis=0)
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[3])), axis=0)
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[4])), axis=0)
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[5])), axis=0)
    # arr = np.concatenate((arr, np.row_stack(clusters_by_label[6])), axis=0)
    visualize_feat_clusters(clusters_by_label)
    # larger_cluster = k_means(arr)
    # visualize_feat_single_PCA(larger_cluster)

    # model = models.resnet18(pretrained=True).cuda()
    # model.train()
    with torch.no_grad():
        for domain1 in tqdm(combine_srcs_loader, leave=False):   
            data, target = domain1
            data, target = data.cuda(), target.cuda()
            features = model.forward_features(data)
            feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

    clusters_by_label = make_clusters(feat_memory_bank)
    cluster_centroids = calculate_centroid(clusters_by_label)
    visualize_feat_PCA(clusters_by_label)
    visualize_tsne(clusters_by_label)

    # convert dic to string to store clusters into an npy file 
    data = {str(k): v for k, v in cluster_centroids.items()}
    np.savez('source_centroids.npz', **data)

    return cluster_centroids

def generate_robust_target_label(model, target_loader, optimizer, criterion, n_epoch, n_batch, early_stop, trained_model_name, centroids, tar_val_loader):
    loss_clf = 0
    stop = 0
    best_acc = 0
    batch_count = 0
    mod_cluster = False
    feat_memory_bank = {}
    clusters_weights = {}
    # visualize_feat_PCA_all(centroids)
    # scattered_graph(centroids)
    target_kmeans = create_target_clusters(model, target_loader, centroids, None)

    model.train()
    for e in range(n_epoch):
        stop += 1
        batch_count = 0
        feat_memory_bank = {}
        true_labels = []
        pseudo_labels = []
        pred_labels = []
        for domain1 in tqdm(target_loader, leave=False):    
            batch_count +=1
            data, gt_labels = domain1
            data, gt_labels = data.cuda(), gt_labels.cuda()

            features = model.forward_features(data)
            cluster_preds = target_kmeans.predict(features.cpu().detach().numpy())
            pseudo_labels.extend(cluster_preds.tolist())
            true_labels.extend(gt_labels.tolist())
            # psuedo_labels, center = k_means_clustering(features, centroids, args.batch_size, n_class, mod_cluster, clusters_weights)
            # l, c = spherical_kmeans(features)
            # new_clusters = fer_clusters(c, centroids, args.batch_size, n_class)
            # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), psuedo_labels))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), psuedo_labels))

            optimizer.zero_grad()
            output, _, _ = model(data, None)
            pred_labels.extend(torch.argmax(torch.nn.functional.softmax(output, dim=1), dim=1).cpu().numpy())
            loss = criterion(output, torch.from_numpy(cluster_preds).cuda().long())
            loss.backward()
            optimizer.step()

            loss_clf = loss.detach().item() + loss_clf
        
        acc, acc_top2 = test(model, target_loader, args.batch_size, False, args.is_pain_dataset)
        # print(f'Epoch: [{e:2d}/{n_epoch}], train_loss_clf: {loss_clf/n_batch:.4f}, acc: {acc:.4f}')
        cluster_acc = accuracy_score(true_labels, pseudo_labels)
        pred_acc = accuracy_score(true_labels, pred_labels)
        print(f"Epoch {e + 1}, cluster prediction accuracy: {cluster_acc:.3f}, prediction accuracy: {pred_acc:.3f}, test accuracy: {acc:.3f}")

        target_kmeans = create_target_clusters(model, target_loader, None, target_kmeans.cluster_centers_)
        # clusters_by_feat = make_clusters(feat_memory_bank)

        # # assume you have a list of class labels
        # class_labels = [0, 1, 2, 3, 4, 5, 6]
        # class_weights = {0: np.count_nonzero(feat_memory_bank[:, -1] == 0), 1: np.count_nonzero(feat_memory_bank[:, -1] == 1), 2: np.count_nonzero(feat_memory_bank[:, -1] == 2), 3: np.count_nonzero(feat_memory_bank[:, -1] == 3), 4: np.count_nonzero(feat_memory_bank[:, -1] == 4), 5: np.count_nonzero(feat_memory_bank[:, -1] == 5), 6: np.count_nonzero(feat_memory_bank[:, -1] == 6)}
        
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 0))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 1))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 2))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 3))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 4))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 5))
        # print(np.count_nonzero(feat_memory_bank[:, -1] == 6))
        # # find the class with the smallest number of samples
        # # smallest_class = min(class_weights, key=class_weights.get)
        # # # calculate the weight of each class
        # # clusters_weights = {label: class_weights[smallest_class] / class_weights[label] for label in class_labels}

        # # calculate total number of samples
        # total_samples = sum(class_weights.values())
        # # calculate weights for each class based on number of samples
        # clusters_weights = {k: v/total_samples for k, v in class_weights.items()}
        # print(clusters_weights)

        # centroids = calculate_centroid(clusters_by_feat)
        # mod_cluster = True
        
        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + '.pkl')
            # torch.save(model.state_dict(), config.CURRENT_DIR + '/' + trained_model_name + 'FER_model.pt')
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, config.CURRENT_DIR + '/' + trained_model_name + '_load.pt')
            experiment.log_metric("Best Accuracy", best_acc)
            stop = 0
        if stop >= early_stop:
            break

def select_relvnt_src_samples(model, target_loader, init_centroids, kmeans_centroids):
    # Define pre-trained ResNet-18 model
    # model = models.resnet18(pretrained=True)
    model.eval()
    # Extract features for all images
    feat_array = []
    target_imgs = torch.randn(1500, 3, 100, 100)
    with torch.no_grad():
        for data, _ in target_loader:
            data = data.cuda()
            features = model.forward_features(data)
            target_imgs = data if len(feat_array) == 0 else torch.cat((target_imgs, data), dim=0)
            feat_array.append(features.squeeze().cpu().detach().numpy())
            # feat_array.append(features.squeeze())

    # Reshape and stack features into a 2D array
    feat_array = np.vstack(feat_array)

    # # Scale the features using StandardScaler
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # feat_array = torch.cat(feat_array)

    # Initialize centroids as the average of the first 6 feature vectors and 4 additional random feature vectors
    # cent = np.concatenate([feat_array[:6].mean(dim=0).cpu().numpy(), feat_array[6:12].cpu().numpy()], axis=0)

    loaded_data = np.load('source_features.npz')
    source_feat = {key: loaded_data[key] for key in loaded_data}
    clusters_by_label = make_clusters(source_feat['arr_0'])
    # clusters_by_label[7] = feat_array

    # visualize_feat_single_PCA(clusters_by_label[0], clusters_by_label[7])
    # visualize_feat_clusters(clusters_by_label, 'tsne')

    src_centroids, target_points = cal_pca_source_mean_target(clusters_by_label)

    # visualize_feat_clusters_PCA(clusters_by_label)

    # visualize_feat_single_PCA(feat_array)

    # centriods = []
    # if kmeans_centroids is None:
    #     centriods.append(init_centroids['0'])
    #     centriods.append(init_centroids['1'])
    #     centriods.append(init_centroids['2'])
    #     centriods.append(init_centroids['3'])
    #     centriods.append(init_centroids['4'])
    #     centriods.append(init_centroids['5'])
    #     centriods.append(init_centroids['6'])
    # else:
    #     centriods = kmeans_centroids

    # Perform K-means clustering
    # target_kmeans = KMeans(n_clusters=7, n_init=20, max_iter=100, random_state=0)
    # target_kmeans = KMeans(n_clusters=2, n_init=1, init=src_centroids, random_state=0, max_iter=10)/
    target_kmeans = KMeans(n_clusters=2, n_init=1, init=5, random_state=0, max_iter=10)
    # kmeans = KMeans(n_clusters=6, n_init=1, init=[0.3387582214158091, 0.3932915386636692, 0.31122543205110037, 0.3974165048609186, 0.3509254715473257, 0.3367280794814298, 0.38326681757189773], random_state=0)
    target_kmeans.fit_predict(target_points)
    labels = target_kmeans.labels_

    target_cluster = making_target_clusters(target_points, labels)
    # clusters_by_label = make_clusters(source_feat['arr_0'])
    # visualize_feat_clusters(clusters_by_label, 'pca', target_cluster, target_kmeans.cluster_centers_)

    # Print the cluster labels
    print(labels)
    return target_kmeans

def create_target_clusters(model, target_loader, init_centroids, kmeans_centroids):
    # Define pre-trained ResNet-18 model
    # model = models.resnet18(pretrained=True)
    model.eval()
    # Extract features for all images
    feat_array = []
    target_imgs = torch.randn(1500, 3, 100, 100)
    with torch.no_grad():
        for data, _ in target_loader:
            data = data.cuda()
            features = model.forward_features(data)
            target_imgs = data if len(feat_array) == 0 else torch.cat((target_imgs, data), dim=0)
            feat_array.append(features.squeeze().cpu().detach().numpy())
            # feat_array.append(features.squeeze())

    # Reshape and stack features into a 2D array
    feat_array = np.vstack(feat_array)

    # # Scale the features using StandardScaler
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # feat_array = torch.cat(feat_array)

    # Initialize centroids as the average of the first 6 feature vectors and 4 additional random feature vectors
    # cent = np.concatenate([feat_array[:6].mean(dim=0).cpu().numpy(), feat_array[6:12].cpu().numpy()], axis=0)

    loaded_data = np.load('source_features.npz')
    source_feat = {key: loaded_data[key] for key in loaded_data}
    clusters_by_label = make_clusters(source_feat['arr_0'])
    # clusters_by_label[7] = feat_array

    # visualize_feat_single_PCA(clusters_by_label[0], clusters_by_label[7])
    visualize_feat_clusters(clusters_by_label, 'tsne')

    src_centroids, target_points = cal_pca_source_mean_target(clusters_by_label)

    # visualize_feat_clusters_PCA(clusters_by_label)

    # visualize_feat_single_PCA(feat_array)

    # centriods = []
    # if kmeans_centroids is None:
    #     centriods.append(init_centroids['0'])
    #     centriods.append(init_centroids['1'])
    #     centriods.append(init_centroids['2'])
    #     centriods.append(init_centroids['3'])
    #     centriods.append(init_centroids['4'])
    #     centriods.append(init_centroids['5'])
    #     centriods.append(init_centroids['6'])
    # else:
    #     centriods = kmeans_centroids

    # Perform K-means clustering
    # target_kmeans = KMeans(n_clusters=7, n_init=20, max_iter=100, random_state=0)
    target_kmeans = KMeans(n_clusters=7, n_init=1, init=src_centroids, random_state=0, max_iter=10)
    # kmeans = KMeans(n_clusters=6, n_init=1, init=[0.3387582214158091, 0.3932915386636692, 0.31122543205110037, 0.3974165048609186, 0.3509254715473257, 0.3367280794814298, 0.38326681757189773], random_state=0)
    target_kmeans.fit_predict(target_points)
    labels = target_kmeans.labels_

    target_cluster = making_target_clusters(target_points, labels)
    clusters_by_label = make_clusters(source_feat['arr_0'])
    visualize_feat_clusters(clusters_by_label, 'pca', target_cluster, target_kmeans.cluster_centers_)
    # Print the cluster labels
    print(labels)
    return target_kmeans

def generate_tsne(model, srcs_loader, tar_loader, temp_model):
    model.eval()
    temp_model.eval()
    correct = 0
    corr_acc_top2 = 0
    len_target_dataset = len(tar_loader.dataset)
    srcs_data_all = None
    src_tar_all = None
    tar_data_all = None
    tar_tar_all = None

    tar_gt_all = None

    with torch.no_grad():
        for data, target in srcs_loader:
            data, target = data.cuda(), target.cuda()
            data = data.float()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            # calculate Top 2 Percent accuracy
            # acc_top2 = Accuracy(top_k=1).to(device)
            # corr_acc_top2 += acc_top2(s_output, target)

            # to plot T-SNE graph
            s_output = s_output.detach().cpu().numpy()

            data = model.forward_tsne(data)
            data_de = data.detach().cpu().numpy()
            
            if (srcs_data_all is None):
                srcs_data_all = data_de
            else:
                srcs_data_all = np.concatenate((srcs_data_all, data_de))

            target_de = pred.detach().cpu().numpy()
            # target_de = target.detach().cpu().numpy()
            if (src_tar_all is None):
                src_tar_all = target_de
            else:
                src_tar_all = np.concatenate((src_tar_all, target_de))

        for data, target in tar_loader:
            data, target = data.cuda(), target.cuda()
            data = data.float()
            s_output = temp_model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            # calculate Top 2 Percent accuracy
            # acc_top2 = Accuracy(top_k=2).to(device)
            # corr_acc_top2 += acc_top2(s_output, target)

            # to plot T-SNE graph
            s_output = s_output.detach().cpu().numpy()

            data = temp_model.forward_tsne(data)
            data_de = data.detach().cpu().numpy()
            
            if (tar_data_all is None):
                tar_data_all = data_de
            else:
                tar_data_all = np.concatenate((tar_data_all, data_de))

            target_de = pred.detach().cpu().numpy()
            target_gt = target.detach().cpu().numpy()
            # target_de = target.detach().cpu().numpy()
            if (tar_tar_all is None):
                tar_tar_all = target_de
                tar_gt_all = target_gt
            else:
                tar_tar_all = np.concatenate((tar_tar_all, target_de))
                tar_gt_all = np.concatenate((tar_gt_all, target_gt))

    # plot_tsne_graph_scr_tar(srcs_data_all, tar_data_all, src_tar_all, tar_tar_all)
    plot_tsne_scr_tar_sub(srcs_data_all, tar_data_all, src_tar_all, tar_tar_all)

    print("TSNE!")

def test(model, target_test_loader, batch_size, top_N_tar_evaluate=False, is_pain_dataset=False):
    model.eval()
    correct = 0
    corr_acc_top1 = 0
    corr_acc_top2 = 0
    len_target_dataset = len(target_test_loader) if is_pain_dataset else len(target_test_loader.dataset) 
    data_all = None
    tar_all = None
    acc_top1 = 0
    acc_top2 = 0

    store_pred = []

    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.cuda(), target.cuda()
            s_output = model.predict(data.float())
            pred = torch.max(s_output, 1)[1]

            if top_N_tar_evaluate:
                store_pred.extend(pred.tolist())
            else:
                correct += torch.sum(pred == target)

            # input = torch.randn(1, 3, 224, 224)
            # macs, params =profile(model, inputs=(input.float(),))

            # macs, params = get_model_complexity_info(model, (3,100,100), as_strings=True, print_per_layer_stat=True, verbose=True)
            # # Extract the numerical value
            # flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
            # # Extract the unit
            # flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

            # print('Computational complexity: {:<8}'.format(macs))
            # print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
            # print('Number of parameters: {:<8}'.format(params))

            # calculate Top 1 Percent accuracy
                
            if not top_N_tar_evaluate:
                acc_top1 = Accuracy(task='multiclass', num_classes=args.n_class, top_k=1).to(device)
                corr_acc_top1 += acc_top1(s_output, target)

             # calculate Top 2 Percent accuracy
            # acc_top2 = Accuracy(top_k=2).to(device)
            # corr_acc_top2 += acc_top2(s_output, target)

            # to plot T-SNE graph
            # s_output = s_output.detach().cpu().numpy()

            # data = model.forward_tsne(data)
            # data_de = data.detach().cpu().numpy()
            
            # if (data_all is None):
            #     data_all = data_de
            # else:
            #     data_all = np.concatenate((data_all, data_de))

            # target_de = pred.detach().cpu().numpy()
            # # target_de = target.detach().cpu().numpy()
            # if (tar_all is None):
            #     tar_all = target_de
            # else:
            #     tar_all = np.concatenate((tar_all, target_de))

    
    if top_N_tar_evaluate:
        item_counts = Counter(store_pred)
        print(item_counts)
        return item_counts
    
    acc = correct.double() / len_target_dataset
    # batch_samples = len_target_dataset / int(batch_size)
    batch_samples = len_target_dataset if len_target_dataset == len(target_test_loader) else len_target_dataset / int(batch_size)
    
    acc_top1 = corr_acc_top1/batch_samples
    # acc_top2 = corr_acc_top2/batch_samples
    acc_top2 = 0
        
    # plot_tsne_graph_scr_tar(data_all, None, tar_all)  
    return acc_top1
    
    # if not top_N_tar_evaluate:
    #     acc = correct.double() / len_target_dataset
    #     # batch_samples = len_target_dataset / int(batch_size)
    #     batch_samples = len_target_dataset if len_target_dataset == len(target_test_loader) else len_target_dataset / int(batch_size)
        
    #     acc_top1 = corr_acc_top1/batch_samples
    #     # acc_top2 = corr_acc_top2/batch_samples
    #     acc_top2 = 0
    # else:
    #     item_counts = Counter(store_pred)
    #     print(item_counts)

    # # plot_tsne_graph_scr_tar(data_all, None, tar_all)  

    # if top_N_tar_evaluate:
    #     return item_counts
    # return acc_top1, acc_top2

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net, transfer_loss='coral', use_bottleneck=False, bottleneck_width=256, width=1024): #1024
        super(TransferNet, self).__init__()
        if base_net == 'resnet50':
            self.base_network = ResNet50Fc()
        elif base_net == 'resnet18':
            self.base_network = ResNet18Fc()
        else:
            # Your own basenet
            return
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        bottleneck_list = [nn.Linear(self.base_network.output_num()
        , bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)

        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(width, num_class)]  

        # classifier_layer_list = [nn.Linear(4608, width), nn.ReLU(), nn.Dropout(0.5),
        #                          nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.00005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

    def forward(self, source, target):
        source = self.base_network(source)
        target = self.base_network(target) if target is not None else None
        source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target) if target is not None else None
        transfer_loss = self.adapt_loss(source, target, self.transfer_loss) if target is not None else None
        # transfer_loss = 0
        return source_clf, transfer_loss, source

    # def forward(self, source):
    #     source = self.base_network(source)
    #     # target = self.base_network(target) if target is not None else None
    #     source_clf = self.classifier_layer(source)
    #     if self.use_bottleneck:
    #         source = self.bottleneck_layer(source)
    #         # target = self.bottleneck_layer(target) if target is not None else None
    #     # transfer_loss = self.adapt_loss(source, target, self.transfer_loss) if target is not None else None
    #     transfer_loss = 0
    #     return source_clf, transfer_loss, source
    
    def forward_features(self, domain):
        domain_fea = self.base_network(domain)
        # target = self.base_network(target) if target is not None else None
        # source_clf = self.classifier_layer(source)
        if self.use_bottleneck:
            domain_fea = self.bottleneck_layer(domain_fea)
            # target = self.bottleneck_layer(target) if target is not None else None
        # transfer_loss = self.adapt_loss(source, target, self.transfer_loss) if target is not None else None
        return domain_fea

    def forward_tsne(self, dataset):
        dataset = self.base_network(dataset)
        if self.use_bottleneck:
            dataset = self.bottleneck_layer(dataset)
        return dataset

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL_loss.coral_loss(X, Y)
            # loss = coral_loss(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss

def create_save_model_name(dataset, type, oracle_setting=False):
    save_model_name = ''
    for i in range(len(dataset)):
        save_model_name += ('_' + type + '_')  if i == len(dataset) - 1 else ('_')
        if dataset[i] == config.RAF_DB_PATH or dataset[i] == config.RAF_DB_TEST_PATH:
            save_model_name += config.DATASET_RAF
        elif dataset[i] == config.AFFECTNET_TRAIN_PATH or dataset[i] == config.AFFECTNET_TEST_PATH:
            save_model_name += config.DATASET_AFFECTNET
        elif dataset[i] == config.FER_DB_TRAIN_PATH or dataset[i] == config.FER_DB_TEST_PATH:
            save_model_name += config.DATASET_FER
        elif dataset[i].find("subject_jaffe") != -1:
            save_model_name += config.DATASET_JAFFE + dataset[i].split('/')[-2]  if i == 0 else dataset[i].split('/')[-2]
        else:
            save_model_name += 'only'
    save_model_name = save_model_name + "_oracle" if oracle_setting else save_model_name
    return save_model_name

def get_dataset_name_from_path(dataset_path):
    if (dataset_path == config.RAF_DB_PATH or dataset_path == config.RAF_DB_TEST_PATH):
        return config.DATASET_RAF
    elif (dataset_path == config.FER_DB_TRAIN_PATH or dataset_path == config.FER_DB_TEST_PATH):
        return config.DATASET_FER
    elif (dataset_path == config.AFFECTNET_TRAIN_PATH or dataset_path == config.AFFECTNET_TEST_PATH):
        return config.DATASET_AFFECTNET
    elif dataset_path.find("subject_jaffe") != -1: 
        return config.DATASET_JAFFE
    elif (dataset_path == config.BIOVID_SUBS_PATH):
        return config.DATASET_BIOVID
    else:
        return "Anonymous"


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range

def plot_tsne_graph(X, Y, label_domain1):
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2

    # features = np.concatenate((X, Y))
    # b, c, nx, ny = X.shape
    # features = X.reshape((b,nx*ny))
    X = X[:300]
    label_domain1 = label_domain1[:300]
    tsne = TSNE(n_components, n_iter=2000, perplexity = 5, verbose=1, random_state=25663214, init="pca")
    tsne_result = tsne.fit_transform(X)
    tsne_result.shape
    
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne_result[:, 0]
    ty = tsne_result[:, 1]

    df = pd.DataFrame()
    df["y"] = label_domain1.tolist()
    df["comp-1"] = tsne_result[:,0]
    df["comp-2"] = tsne_result[:,1]

    # print(df.y.tolist())
    # sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
    #             palette=sns.color_palette("hls", 7),
    #             data=df).set(title="FER data T-SNE projection") 
    
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors_per_class = [0,1,2,3,4,5,6]
    color=['brown','blue','green','purple', 'pink', 'red', 'magenta', 'gray']
    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(label_domain1) if l == label]
    
        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
    
        # convert the class color to matplotlib format
        # color = np.array(colors_per_class[label], dtype=np.float) / 255
    
        # add a scatter plot with the corresponding color and label
        ax.scatter(current_tx, current_ty, label=label, color=color[label])
    
    # build a legend using the labels we set previously
    ax.legend(loc='best')
    
    # finally, show the plot
    # plt.savefig('tsne_source_raf_fer.png')
    plt.savefig('tsne_tar_affectnet.png')
    plt.show()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a network on FER')
    arg_parser.add_argument('--src1_dataset_path', type=str, default=config.BIOVID_SUB_1_PATH)
    arg_parser.add_argument('--src2_dataset_path', type=str, default=config.BIOVID_SUB_2_PATH)

    arg_parser.add_argument('--src_train_datasets_path', type=str, default=config.MCMASTER_PATH) # BIOVID_SUBS_PATH or MCMASTER_PATH
    arg_parser.add_argument('--tar_datasets_path', type=str, default=config.BIOVID_SUBS_PATH) # BIOVID_SUBS_PATH or MCMASTER_PATH

    arg_parser.add_argument('--tar_dataset_path', type=str, default=config.BIOVID_SUB_3_PATH)
    arg_parser.add_argument('--test_tar_dataset_path', type=str, default=config.BIOVID_SUB_3_PATH)
    arg_parser.add_argument('--pain_db_root_path', type=str, default=config.MCMASTER_PATH) # BIOVID_PATH or MCMASTER_PATH

    arg_parser.add_argument('--pain_tar_db_root_path', type=str, default=config.BIOVID_PATH) # BIOVID_PATH or MCMASTER_PATH
    arg_parser.add_argument('--pain_label_path', type=str, default=config.MCMASTER_TWO_LABEL_PATH) # BIOVID_REDUCE_LABEL_PATH or MCMASTER_TWO_LABEL_PATH
    arg_parser.add_argument('--pain_tar_label_path', type=str, default=config.BIOVID_REDUCE_LABEL_PATH) # BIOVID_REDUCE_LABEL_PATH or MCMASTER_TWO_LABEL_PATH

    arg_parser.add_argument('--pretrained_model', type=str, default='mcmaster_trained_model')
    arg_parser.add_argument('--batch_size', type=int, default=16)    # 1 only because JAFFE has small dataset
    arg_parser.add_argument('--source_epochs', type=int, default=20)
    arg_parser.add_argument('--target_epochs', type=int, default=50)
    arg_parser.add_argument('--early_stop', type=int, default=30)
    arg_parser.add_argument('--single_best', type=str, default=False)
    arg_parser.add_argument('--train_model_wo_adaptation', type=str, default=False)
    arg_parser.add_argument('--oracle_setting', type=str, default=False)
    arg_parser.add_argument('--source_free', type=str, default=False)
    arg_parser.add_argument('--target_clustering', type=bool, default=False)

    arg_parser.add_argument('--load_source', type=str, default=False)
    arg_parser.add_argument('--train_N_source_classes', type=bool, default=False)
    arg_parser.add_argument('--train_with_dist_measure', type=bool, default=True)
    arg_parser.add_argument('--first_adapt_N_source_subjects', type=bool, default=False)
    arg_parser.add_argument('--load_prev_source_model', type=bool, default=True)
    arg_parser.add_argument('--accumulate_prev_source_subs', type=bool, default=True)
    arg_parser.add_argument('--apply_replay', type=bool, default=True)

    arg_parser.add_argument('--tar_subject', type=int, help='Target subject number', default=0)
    arg_parser.add_argument('--expand_tar_dataset', type=bool, default=False)
    # arg_parser.add_argument('--experiment_description', type=str, default='Select CS closest source samples and use for target adaptation.')

    arg_parser.add_argument('--experiment_description', type=str, default='Cross-dataset. Src=Unbc. Select CS closest source samples and use for target adaptation.')

    # arg_parser.add_argument('--experiment_description', type=str, default='1srcload. include all prev src subs (pretrained). *Add tar PL in minibatch. *MMD Conf tar samples.**') # create src and target clusters k-means *Add prev src CE 

    arg_parser.add_argument('--target_evaluation_only', type=bool, default=False)
    arg_parser.add_argument('--top_src_sub', type=str, default='072414_m_23,082714_m_22,110909_m_29')
    arg_parser.add_argument('--top_timestamp', type=str, default='1721256232')
    arg_parser.add_argument('--train_source', type=str, default=False)
    arg_parser.add_argument('--weights_folder', type=str, default=config.WEIGHTS_FOLDER)
    arg_parser.add_argument('--source_combined', type=str, default=False)
    arg_parser.add_argument('--is_pain_dataset', type=bool, default=True)
    arg_parser.add_argument('--dist_measure', type=str, default=config.COSINE_SIMILARITY)
    arg_parser.add_argument('--top_k', type=int, default=19)
    arg_parser.add_argument('--n_class', type=int, default=2) # 6 for unbc and 2 for pain Biovid -- n_class = 77: to train a classifier with N source subject classes
    arg_parser.add_argument('--n_class_N_src', type=int, default=77)
    arg_parser.add_argument('--top_rev_src_sam', type=int, default=31000)
    arg_parser.add_argument('--train_w_rand_src_sm', type=bool, default=False)
    arg_parser.add_argument('--train_w_src_sm', type=bool, default=False)
    arg_parser.add_argument('--cs_threshold', type=float, default=0.80)
    
    arg_parser.add_argument('--back_bone', default="resnet18", type=str)
    args = arg_parser.parse_args()

    #-- BioVid
    # 0. 081014_w_27 [40]
    # 1. 101609_m_36 [70]
    # 2. 112009_w_43 [66]
    # 3. 091809_w_43 [68]
    # 4. 071309_w_21 [4]
    # 5. 073114_m_25 [69]
    # 6. 080314_w_25 [80]
    # 7. 073109_w_28 [82]
    # 8. 100909_w_65 [13]
    # 9. 081609_w_40 [17]
    
    #-- UNBC
    # 0. 107-hs107 
    # 1. 109-ib109
    # 2. 121-vw121
    # 3. 123-jh123
    # 4. 115-jy115

    main(args)
