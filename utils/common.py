import os
import numpy as np
import torch
import math
from tqdm import tqdm
from torchmetrics.functional import accuracy
from torchmetrics import F1Score
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import random
import mmap
import calendar
import time
import config
import csv
from collections import defaultdict

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def do_epoch(model, dataloader, criterion, device, optim=None):
    """
    it will run for single epoch; visit all the samples in the database 
    """
    total_loss = 0
    total_accuracy = 0

    for x, y_true in tqdm(dataloader, leave=False):
        x, y_true = x.to(device), y_true.to(device)
        
        # img = x[0].cpu().squeeze()
        # # label = y_true[0]
        # plt.imshow(img, cmap="gray")
        # plt.show()

        y_pred = model(x.float(), x.float()) # for training the custom dataset; I have added .float() otherwise removed it when using build-in dataset 
        loss = criterion(y_pred, y_true)

        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()

        total_loss += loss.item()
        f1 = F1Score(num_classes=6).to(device)
        
        # normalize_soft = torch.nn.Softmax(y_pred)
        # y_true = torch.where(y_true == 0, -1, y_true)
        # total_accuracy += (y_pred.max(1)[1] == y_true).float().mean().item()
        total_accuracy += f1(y_pred.max(1)[1], y_true)
    mean_loss = total_loss / len(dataloader)
    mean_accuracy = total_accuracy / len(dataloader)

    return mean_loss, mean_accuracy

def knn(tar_feature, src_features):
    nn_feature_bank = []
    
    tar_mean = torch.mean(tar_feature)
    tar_mean.item
    for src_feat in src_features:
        src_mean = torch.mean(src_feat)
        # cosine_sim = cosine_similarity(tar_feature, src_feat)
        dist = abs(tar_mean.detach().item() - src_mean.detach().item())
        nn_feature_bank.append(dist)

    return nn_feature_bank

def get_srcs_tar_name(random_list, subs_path, target_sub, n_class, oracle = False):
    all_subs_list = os.listdir(subs_path)
     # let target subject map to random id
    tar_sub = random_list[target_sub-1]

    srcs_file_name = 'lab_srcs' + str(len(random_list)-1) + '_cl' + str(n_class) if n_class > 2 else 'lab_srcs' + str(len(random_list)-1) 
    rand_list_count = random_list[:10] if len(random_list) > 10 else random_list
    for index in rand_list_count:
        sub_folder = all_subs_list[index]
        split_folder = sub_folder.split("_")
        srcs_file_name = srcs_file_name + "_" + split_folder[1] + split_folder[2] if index != tar_sub else srcs_file_name
        # srcs_file_name = srcs_file_name + "_" + sub_folder if index != tar_sub else srcs_file_name

    split_folder = all_subs_list[tar_sub].split("_")
    tar_file_name = srcs_file_name + "_tar_" + split_folder[1] + split_folder[2] + "_oracle.txt" if oracle else srcs_file_name + "_tar_" + split_folder[1] + split_folder[2] + ".txt"
    # srcs_file_name = srcs_file_name + "_____only.txt" if len(random_list) > 10 else srcs_file_name + "_only.txt"

    # to create file 
    if len(random_list) > 10 and oracle:
        srcs_file_name = srcs_file_name + "_____only_oracle_tar" + all_subs_list[random_list[target_sub-1]] + ".txt"   # bcz org tar is the second last value in oracle setting
    elif len(random_list) > 10:
        srcs_file_name = srcs_file_name + "_____only.txt"
    elif oracle:
        srcs_file_name = srcs_file_name + "_oracle_tar" + all_subs_list[random_list[target_sub-1]] + ".txt"     # bcz org tar is the second last value in oracle setting
    else:
        srcs_file_name + "_only.txt"

    return srcs_file_name, tar_file_name, all_subs_list

def write_srcs_tar_txt_files(subs_num, subs_path, label_file_path, random_list, target_list_fix, target_sub, n_class, oracle):
    
    # random_list = random.sample(range(0, 86), subs_num) if random_list == None else random_list
    all_possible_numbers = list(range(0, 87))
    random_list = random.sample([number for number in all_possible_numbers if number not in target_list_fix], subs_num) if random_list == None else random_list
    srcs_file_name, tar_file_name, all_subs_list = get_srcs_tar_name(random_list, subs_path, target_sub, n_class, oracle)
    # print(random_list)
     # let target subject map to random id
    target_sub = random_list[target_sub-1]

    file_read = open(label_file_path, 'r')
    lines = file_read.readlines()
    
    srcs_write_file = open(srcs_file_name, "w+")
    tar_write_file = open(tar_file_name, "w+")
    print([all_subs_list[index] for index in random_list if index < len(all_subs_list)])
    for line in lines:
        # read subject folder from LABEL TEXT file and write into a seperate text file
        for index in random_list:
            sub_folder = all_subs_list[index]
            if sub_folder in line:
                 srcs_write_file.write(line) if index != target_sub else tar_write_file.write(line)

    srcs_write_file.close()
    tar_write_file.close()

    return random_list




   # -------------------------  picking source and target subjects using subjects names ------------------ #  

def get_srcs_tar_name_using_list(all_sub_list_name, subs_path, target_subject, n_class, oracle = False):
    # all_subs_list = os.listdir(subs_path)
     # let target subject map to random id
    # tar_sub = random_list[target_sub-1]

    # selected_subs = [element for element in all_subs_list if element in source_list_name]
    # source_list_name.append(target_subject) 

    srcs_file_name = 'lab_srcs' + str(len(all_sub_list_name)) + '_cl' + str(n_class) if n_class > 2 else 'lab_srcs' + str(len(all_sub_list_name))
    srcs_file_name = srcs_file_name + "_McMaster" if 'McMaster' in subs_path else srcs_file_name
    rand_list_count = all_sub_list_name[:10] if len(all_sub_list_name) > 10 else all_sub_list_name
    for subject in rand_list_count:
        # sub_folder = all_subs_list[index]
        if 'McMaster' in subs_path:
            split_folder = subject.split("-")
            srcs_file_name = srcs_file_name + "_" + split_folder[0] if subject != target_subject else srcs_file_name
        else:
            split_folder = subject.split("_")
            srcs_file_name = srcs_file_name + "_" + split_folder[0] + split_folder[1] + split_folder[2] if subject != target_subject else srcs_file_name
        # srcs_file_name = srcs_file_name + "_" + sub_folder if index != tar_sub else srcs_file_name

    # split_folder = all_subs_list[tar_sub].split("_")
    split_folder = target_subject.split("-") if 'McMaster' in subs_path else target_subject.split("_")

    if 'McMaster' in subs_path or '-' in target_subject:
        tar_file_name = srcs_file_name + "_tar_" + split_folder[0] + "_oracle.txt" if oracle else srcs_file_name + "_tar_" + split_folder[0] + ".txt"
    else:
        tar_file_name = srcs_file_name + "_tar_" + split_folder[0] + split_folder[1] + split_folder[2] + "_oracle.txt" if oracle else srcs_file_name + "_tar_" + split_folder[0] + split_folder[1] + split_folder[2] + ".txt"

    # srcs_file_name = srcs_file_name + "_____only.txt" if len(random_list) > 10 else srcs_file_name + "_only.txt"

    # to create file 
    if len(all_sub_list_name) > 10:
        srcs_file_name = srcs_file_name + "_____only.txt"
    else:
        srcs_file_name = srcs_file_name + "_only.txt"

    return srcs_file_name, tar_file_name, all_sub_list_name

def write_srcs_tar_txt_files_using_list(subs_path, label_file_path, all_sub_list_name, target_subject, n_class, target_weight_path, oracle, pain_tar_label_path=None):
    
    # random_list = random.sample(range(0, 86), subs_num) if random_list == None else random_list
    all_possible_numbers = list(range(0, 87))
    # random_list = random.sample([number for number in all_possible_numbers if number not in target_list_fix], subs_num) if random_list == None else random_list
    srcs_file_name, tar_file_name, all_subs_list = get_srcs_tar_name_using_list(all_sub_list_name, subs_path, target_subject, n_class, oracle)
    # print(random_list)
     # let target subject map to random id
    # target_sub = random_list[target_sub-1]

    file_read = open(label_file_path, 'r')
    lines = file_read.readlines()
    
    srcs_file_path = os.path.join(target_weight_path, srcs_file_name)
    tar_file_path = os.path.join(target_weight_path, tar_file_name)

    if pain_tar_label_path:
        tar_file_read = open(pain_tar_label_path, 'r')
        tar_lines = tar_file_read.readlines()
        write_label_files(all_sub_list_name, srcs_file_path, lines)
        write_label_files(all_sub_list_name, tar_file_path, tar_lines)
        return all_sub_list_name

    if not os.path.exists(srcs_file_path) or not os.path.exists(tar_file_path):
        srcs_write_file = open(srcs_file_path, "w+")
        tar_write_file = open(tar_file_path, "w+")

        # print([all_subs_list[index] for index in random_list if index < len(all_subs_list)])
        for line in lines:
            # read subject folder from LABEL TEXT file and write into a seperate text file
            for sub_folder in all_sub_list_name:
                if sub_folder in line:
                    # if 'McMaster' in label_file_path:
                    #     mcmaster_data = line.split(" ")
                    #     if mcmaster_data[1] == '0' or mcmaster_data[1] == '4' or mcmaster_data[1] == '5':
                    #         write_txt = '1' if mcmaster_data[1] == '4' or mcmaster_data[1] == '5' else mcmaster_data[1]
                    #         srcs_write_file.write(mcmaster_data[0] + " " + write_txt + "\n") if sub_folder != target_subject else tar_write_file.write(mcmaster_data[0] + " " + write_txt + "\n")
                    # else:
                    srcs_write_file.write(line) if sub_folder != target_subject else tar_write_file.write(line)

        srcs_write_file.close()
        tar_write_file.close()

    return all_sub_list_name

def write_label_files(all_sub_list_name, file_path, lines):
    if not os.path.exists(file_path):
        write_file = open(file_path, "w+")

        # print([all_subs_list[index] for index in random_list if index < len(all_subs_list)])
        for line in lines:
            # read subject folder from LABEL TEXT file and write into a seperate text file
            for sub_folder in all_sub_list_name:
                if sub_folder in line:
                    write_file.write(line)

        write_file.close()
        write_file.close()

def create_target_folders(root_path, folder_name, target_name, timestamp):
    folder_path = os.path.join(root_path, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    # dest_path = os.path.join(folder_path, config.ALL_SOURCES_FOLDER) if train_source else 
    dest_path = os.path.join(folder_path, str(target_mapping(target_name)) + "-" +target_name)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    target_files_path = os.path.join(dest_path, "files")
    if not os.path.exists(target_files_path):
        os.makedirs(target_files_path, exist_ok=True)

    if timestamp is None:
        timestamp = calendar.timegm(time.gmtime())
        target_timestamp_path = os.path.join(dest_path, str(timestamp))
        os.makedirs(target_timestamp_path, exist_ok=True)
    else:
        target_timestamp_path = os.path.join(dest_path, str(timestamp))
    target_weights_path = os.path.join(target_timestamp_path, "weights")
    if not os.path.exists(target_weights_path):
        os.makedirs(target_weights_path, exist_ok=True)
    
    return target_files_path, target_weights_path, str(timestamp)

def target_mapping(target_name):
    if target_name == "081014_w_27":
        return 1
    elif target_name == "101609_m_36":
        return 2
    elif target_name == "112009_w_43":
        return 3
    elif target_name == "091809_w_43":
        return 4
    elif target_name == "071309_w_21":
        return 5
    elif target_name == "073114_m_25":
        return 6
    elif target_name == "080314_w_25":
        return 7
    elif target_name == "073109_w_28":
        return 8
    elif target_name == "100909_w_65":
        return 9
    elif target_name == "081609_w_40":
        return 10
    else:
        return 0
    
def dist_mapping(dist_meaure):
    if dist_meaure == config.COSINE_SIMILARITY:
        return 'Cosine Similarity'
    elif dist_meaure == config.MMD_SIMILARITY:
        return 'MMD Distance'

def map_subid_to_subname(map_id_dir, subid_dic):
    srcname_dic = {}
    src_dic = {}

    # retrieve subid in order of its occurrence
    subid_keys = [key for key, _ in subid_dic.most_common()]

    with open(map_id_dir, 'r') as file:
    # Read each line in the file
        for line in file:
            # Split the line by comma to get the values
            values = line.strip().split(',')
            src_dic[values[1]] = values[0]

    for subid in subid_keys:
        srcname_dic[src_dic.get(str(subid))] = subid
    return srcname_dic

def analyze_unbc(pain_label_path):
    # Initialize a nested defaultdict to store counts
    counts = defaultdict(lambda: defaultdict(int))

    # Read the data from a file named 'data.txt'
    with open(pain_label_path, 'r') as file:
        for line in file:
            # Split the line into path and number
            parts = line.strip().split()
            if len(parts) > 2:
                path, label, _, _ = parts
                subname = path.split('/')[0] # Get the first letter of the path
                try:
                    label = int(label)
                    if -1 <= label <= 5:
                        counts[subname][label] += 1
                except ValueError:
                    # Skip lines where the number is not a valid integer
                    continue

    # Print the results
    for letter in sorted(counts.keys()):
        print(f"{letter}:")
        for number in range(6):  # 0 to 5
            print(f"  {number}: {counts[letter][number]}")
        print()  # Empty line for better readability

def create_two_label_unbc(input_file, output_file):
    # Read the input file and process lines
    with open(output_file, 'w') as writefile:
        with open(input_file, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) > 2:
                    path, label, _, _ = parts
                    try:
                        label = int(label)
                        if label == 0:
                            lines = path.split(' ')[0] + ' 0'
                        elif label in [4, 5]:
                            lines = path.split(' ')[0] + ' 1'
                            
                        writefile.write(f"{lines}\n")
                    except ValueError:
                        continue

    print(f"Processing complete. Results written to {output_file}")