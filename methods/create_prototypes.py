# import comet_ml at the top of your file
from comet_ml import Experiment

import argparse

import os, sys
import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm, trange
from torch import nn
import random
import cv2

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils.common import set_requires_grad, loop_iterable, knn
from utils.distance import compute_distance_matrix
from methods.clusters import *
# from utils.visualize import *

from datasets.base_dataset import BaseDataset
from models.resnet_model_modified import ResNet50Fc, ResNet18Fc
from losses.coral_loss import CORAL_loss
from losses.mmd_loss import MMD_loss
import torch.nn.functional as F

from models.resnet_fer_modified import *

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from utils.tsne_variation import *
# from utils.tsne_github import *

from torch.utils.data import TensorDataset
import torchvision.transforms as transforms

import config

# Create an experiment with your api key
experiment = Experiment(
    api_key="eow2bmNwSPBKrx657Qfx43lW7",
    project_name="Multi Source Adaptation Experiments",
    workspace="osamazeeshan",
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transfer_loss = 'mmd'
learning_rate = 0.0001
n_class = 7

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)   # 6
    m = y.size(0)   # 16
    d = x.size(1)   # 512
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x.detach().cpu() - y.detach().cpu(), 2).sum(2)

class CustomHorizontalFlip:
    def __call__(self, image):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            # transforms.RandomAdjustSharpness(0.5),
            # transforms.RandomRotation(90),
            # transforms.ColorJitter(brightness=.5, hue=.3),
        ]) (image)
        # return transforms.RandomHorizontalFlip()(image)

class CustomAugmentation:
    def __call__(self, image):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAdjustSharpness(2),#0.5
            transforms.RandomRotation(90),
            # transforms.ColorJitter(brightness=.5, hue=.3),
        ]) (image)
        # return transforms.RandomHorizontalFlip()(image)

# ------------------ ------------------------------------------------------------------ #
# Creating target Tpl and Tcl dictionaries, containing data + labels + class_prob 
# ------------------ ------------------------------------------------------------------ #

def create_target_pl_dicts(model, target_domain, threshold, batch, is_pain_dataset=False):
    model.eval()

    correct = 0
    correct_aug = 0
    conf_correct = 0
    conf_index = 0
    # print("\n**** Creating Tpl and Tcl dictionaries **** \n")
    # len_inter_dataset = len(target_domain.dataset)
    len_inter_dataset = len(target_domain)*batch if is_pain_dataset else len(target_domain.dataset) 
    # print("Target PL dataset len: ", len_inter_dataset)
    # rank_data_tensor = torch.randn(len_inter_dataset, len_inter_dataset, 3, 100, 100)
    
    data_arr = []
    prob_arr = []
    label_arr = []
    gt_arr = []
    correct_pred_arr = []

    non_conf_data_arr = []
    non_conf_label_arr = []

    conf_data_arr = []
    conf_pred_arr = []
    conf_label_arr = []
    augmentation_transform = CustomHorizontalFlip()
    with torch.no_grad():
        for data, target in target_domain:
            data, target = data.cuda(), target.cuda()
            data = data.float()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            augmented_images = [augmentation_transform(tensor) for tensor in data]

            augmented_batch_tensor = torch.stack(augmented_images)
            augmented_batch_tensor=augmented_batch_tensor.cuda()
            s_output_aug = model.predict(augmented_batch_tensor)
            pred_aug = torch.max(s_output_aug, 1)[1]
            correct_aug += torch.sum(pred_aug == target)

            softmax_pred_aug = get_target_pred_val(s_output_aug.detach().cpu().numpy(), pred_aug.detach().cpu().numpy())
            # softmax_pred_aug_2 = get_target_pred_val(s_output_aug_2.detach().cpu().numpy(), pred_aug_2.detach().cpu().numpy())
            softmax_pred = get_target_pred_val(s_output.detach().cpu().numpy(), pred.detach().cpu().numpy())

            soft_pred_avg = np.add(softmax_pred, softmax_pred_aug)/2
            
            np_prob = np.array(soft_pred_avg)

            correct_pred_indx = torch.nonzero(pred == target).squeeze().detach().cpu().numpy()
            correct_pred_arr.extend(np_prob[correct_pred_indx]) if len(np.unique(correct_pred_indx)) > 1 else correct_pred_arr.append(np_prob[correct_pred_indx])

            conf_indxs = np.where(np_prob > threshold)[0]

            non_conf_indxs = np.where(np_prob < threshold)[0]
            # print(len(conf_indxs))
            if len(conf_indxs) > 0:
                conf_prob = np_prob[conf_indxs]
                conf_pred_arr.extend(conf_prob)

                np_data = np.array(data.detach().cpu().numpy())

                # store non_conf_data to set label=-1
                non_conf_data = np_data[non_conf_indxs]
                non_conf_data_arr.extend(non_conf_data)
                non_conf_label_arr.extend(np.full(len(non_conf_indxs), -1)) 
                
                conf_data = np_data[conf_indxs]
                conf_data_arr.extend(conf_data)

                # take labels from the prediction that has the highest softmax prob 
                soft_np = np.array(softmax_pred)
                soft_aug_np = np.array(softmax_pred_aug)
                # soft_aug_np_2 = np.array(softmax_pred_aug_2)
                
                np_label = np.array(pred.detach().cpu().numpy())
                np_gt_label = np.array(target.detach().cpu().numpy())
                aug_np_label = np.array(pred_aug.detach().cpu().numpy())
                # aug_np_label_2 = np.array(pred_aug_2.detach().cpu().numpy())
                conf_label = []
                for conf_indx in conf_indxs:
                    if soft_np[conf_indx] > soft_aug_np[conf_indx] :
                        label = np_label[conf_indx] 
                    else :
                        label = aug_np_label[conf_indx]

                    # if soft_np[conf_indx] > soft_aug_np[conf_indx] and soft_np[conf_indx] > soft_aug_np_2[conf_indx] :
                    #     label = np_label[conf_indx] 
                    # elif soft_aug_np[conf_indx] > soft_aug_np_2[conf_indx] :
                    #     label = aug_np_label[conf_indx]
                    # else :
                    #     label = aug_np_label_2[conf_indx]

                    conf_label.append(label)
                    conf_label_arr.append(label)
                    
                gt_arr.extend(np_gt_label[conf_indxs])
                conf_label = np.array(conf_label)

                # np_label = np.array(pred.detach().cpu().numpy())
                # conf_label = np_label[conf_indxs]
                # conf_label_arr.extend(conf_label)

                # storing features and labels to create Tcl clusters
                # conf_data_ten = torch.from_numpy(conf_data).cuda()
                # conf_label_ten = torch.from_numpy(conf_label).cuda()
                # # conf_data_feat = model.forward_features(conf_data_ten)
                # conf_data_feat = model.forward_tsne(conf_data_ten)
                
                # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((conf_data_feat.cpu().detach().numpy(), conf_label_ten.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((conf_data_feat.cpu().detach().numpy(), conf_label_ten.cpu().detach().numpy()))
                
                # feat_memory_bank = np.concatenate((feat_memory_bank, conf_data_feat.cpu().detach().numpy())) if len(feat_memory_bank) > 0 else (conf_data_feat.cpu().detach().numpy())


                # calculate prediction for confident pseudo labels
                conf_target = target[conf_indxs]
                conf_correct += torch.sum(torch.tensor(conf_label).detach().cpu() == conf_target.detach().cpu())
                conf_index += len(conf_indxs)
            else: # it will contains data that were not part of conf PL, we assign -1 to these samples that were only used in MMD and not in tar CE  
                non_conf_data_arr.extend(data.detach().cpu().numpy())
                non_conf_label_arr.extend(np.full(batch, -1))   

            # creating Tpl; which contains all the data 
            data_arr.extend(data.detach().cpu().numpy())
            prob_arr.extend(softmax_pred_aug)
            label_arr.extend(pred.detach().cpu().numpy())
            
            # gt_arr.extend(target.detach().cpu().numpy())

            # np_data = np.array(data.detach().cpu().numpy())
            # conf_label_ten = torch.from_numpy(np_data).cuda()
            # conf_data_feat = model.forward_features(conf_label_ten)
            # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((conf_data_feat.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((conf_data_feat.cpu().detach().numpy(), target.cpu().detach().numpy()))

    conf_data_arr = np.asarray(conf_data_arr)
    conf_pred_arr = np.asarray(conf_pred_arr)
    conf_label_arr = np.asarray(conf_label_arr)

    # creating clusters from features and label information
    # clusters_by_label = make_clusters(feat_memory_bank)
    # cluster_centroids = calculate_centroid(clusters_by_label)
    # visualize_feat_PCA(clusters_by_label)
    # visualize_tsne(clusters_by_label)

    # feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

    data_arr = np.asarray(data_arr)
    prob_arr = np.asarray(prob_arr)
    label_arr = np.asarray(label_arr)
    gt_arr = np.asarray(gt_arr)

    non_conf_data_arr = np.asarray(non_conf_data_arr)
    non_conf_label_arr = np.asarray(non_conf_label_arr)

    # visualize_feat_clusters(clusters_by_label, 'tsne')
    # plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, gt_arr, None)

    # for tsne
    # plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, conf_label_arr, conf_label_arr)

    # visualize_feat_clusters(clusters_by_label, 'tsne')
    # plot_tsne_graph_scr_tar(feat_memory_bank, feat_memory_bank, label_arr, None)

    # ranked_data = np.asarray(data_arr)[(-np.asarray(prob_arr)).argsort()]
    # ranked_labels = np.asarray(label_arr)[(-np.asarray(prob_arr)).argsort()]

    # print("Threshold: ", threshold)
    # print("\n** -- **\n")
    # print("Total target PL samples: ", conf_index)
    # print("Total correctly classified: ", conf_correct)
    conf_acc = 0
    acc = correct.double() / len_inter_dataset
    if conf_correct != 0:
        conf_acc = conf_correct.double() / conf_index
    # print("Acc: ", acc)
    # print("Conf Acc: ", conf_acc)

    return conf_data_arr, conf_pred_arr, conf_label_arr, gt_arr, non_conf_data_arr, non_conf_label_arr

def get_target_pred_val(s_output, target):
    tar_values = []
    softmax_pred = torch.nn.functional.softmax(s_output, axis=1)
    for i in range(0, len(target)):
        tar_values.append(softmax_pred[i][target[i]])
    return tar_values

def generate_tar_aug_conf_pl(model, target_domain, threshold):
    model.eval()
    print("\n**** Generating Augmented Confident Target Pseudo-labels **** \n")
    
    data_arr = []
    prob_arr = []
    label_arr = []
    gt_arr = []
    # correct_pred_arr = []

    conf_data_arr = []
    conf_pred_arr = []
    conf_label_arr = []

    horizontal_flip_transform = CustomHorizontalFlip()
    with torch.no_grad():
        for data, target in target_domain:
            data, target = data.cuda(), target.cuda()
            data = data.float()
            s_output = model.predict(data)
            pred = torch.max(s_output, 1)[1]

            augmented_images = [horizontal_flip_transform(tensor) for tensor in data]

            augmented_batch_tensor = torch.stack(augmented_images)
            augmented_batch_tensor=augmented_batch_tensor.cuda()
            s_output_aug = model.predict(augmented_batch_tensor)
            pred_aug = torch.max(s_output_aug, 1)[1]

            softmax_pred_aug = get_target_pred_val(s_output_aug.detach().cpu().numpy(), pred_aug.detach().cpu().numpy())
            softmax_pred = get_target_pred_val(s_output.detach().cpu().numpy(), pred.detach().cpu().numpy())
            soft_pred_avg = np.add(softmax_pred, softmax_pred_aug)/2
            
            np_prob = np.array(soft_pred_avg)

            conf_indxs = np.where(np_prob > threshold)[0]
            if len(conf_indxs) > 0:
                conf_prob = np_prob[conf_indxs]
                conf_pred_arr.extend(conf_prob)

                np_data = np.array(data.detach().cpu().numpy())
                conf_data = np_data[conf_indxs]
                conf_data_arr.extend(conf_data)

                # take labels from the prediction that has the highest softmax prob 
                soft_np = np.array(softmax_pred)
                soft_aug_np = np.array(softmax_pred_aug)
                
                np_label = np.array(pred.detach().cpu().numpy())
                np_gt_label = np.array(target.detach().cpu().numpy())
                aug_np_label = np.array(pred_aug.detach().cpu().numpy())
                conf_label = []
                for conf_indx in conf_indxs:
                    if soft_np[conf_indx] > soft_aug_np[conf_indx] :
                        label = np_label[conf_indx] 
                    else :
                        label = aug_np_label[conf_indx]

                    conf_label.append(label)
                    conf_label_arr.append(label)
                    
                gt_arr.extend(np_gt_label[conf_indxs])
                conf_label = np.array(conf_label)
                
            # creating Tpl; which contains all the data 
            data_arr.extend(data.detach().cpu().numpy())
            prob_arr.extend(softmax_pred_aug)
            label_arr.extend(pred.detach().cpu().numpy())

    """
    The np.asarray() function in NumPy is used to convert an input array-like object (such as a list, 
    tuple, or ndarray) into an ndarray. It creates a new ndarray if the input is not already an ndarray, 
    and it returns the input as is if it is already an ndarray.
    """
    conf_data_arr = np.asarray(conf_data_arr)
    conf_pred_arr = np.asarray(conf_pred_arr)
    conf_label_arr = np.asarray(conf_label_arr)

    data_arr = np.asarray(data_arr)
    prob_arr = np.asarray(prob_arr)
    label_arr = np.asarray(label_arr)
    gt_arr = np.asarray(gt_arr)

    return conf_data_arr, conf_pred_arr, conf_label_arr, gt_arr


'''
This function replicate the method from paper: Progressive Feature
to generate targer PL and compare with our ACPL method
'''
def cos_distance(vector1,vector2):  
    dot_product = 0.0;  
    normA = 0.0;  
    normB = 0.0;  
    for a,b in zip(vector1,vector2):  
        dot_product += a*b  
        normA += a**2  
        normB += b**2  
    if normA == 0.0 or normB==0.0:  
        return None  
    else:  
        return dot_product / ((normA*normB)**0.5)

def create_src_prototypes(model, combine_srcs_loader):
    clusters_by_label = {}
    feat_memory_bank = {}
    with torch.no_grad():
        for domain1 in tqdm(combine_srcs_loader, leave=False):   
            data, target = domain1
            data, target = data.cuda(), target.cuda()
            features = model.forward_features(data.float())
            feat_memory_bank = np.concatenate((feat_memory_bank, np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))), axis=0) if len(feat_memory_bank) > 0 else np.column_stack((features.cpu().detach().numpy(), target.cpu().detach().numpy()))

    clusters_by_label = make_clusters(feat_memory_bank)
    src_feat_class_centroids = calculate_centroid(clusters_by_label)

    data = {str(k): v for k, v in src_feat_class_centroids.items()}
    np.savez('source_centroids.npz', **data)

    return src_feat_class_centroids

def generate_prototypical_tar_PL(model, combine_srcs_loader, target_loader, batch_size):
    # src_feat_class_centroids = create_src_prototypes(model, combine_srcs_loader)
    src_feat_class_centroids = np.load('source_centroids.npz')
    centroids = {key: src_feat_class_centroids[key] for key in src_feat_class_centroids}

    len_inter_dataset = len(target_loader)*batch_size
    threshold = 0.42

    data_arr = []
    prob_arr = []
    label_arr = []
    gt_arr = []

    conf_correct = 0
    conf_index = 0
    sim_list_all = []

    model.eval()
        
    for data, gt_label in tqdm(target_loader, leave=False):
        data, gt_label = data.cuda(), gt_label.cuda()

        # optimizer.zero_grad()
        _, _, features = model(data.float(), None)
        sim_list = []
        for samp_indx in range(len(data)):
            sim_list = []
            sim_list.append(cos_distance(centroids['0'], features[samp_indx]))
            sim_list.append(cos_distance(centroids['1'], features[samp_indx]))

            max_sim = max(sim_list)
            conf_label = sim_list.index(max_sim)
            sim_list_all.append(max_sim.detach().cpu().numpy())
            # print(max_sim)
            if max_sim.cpu().detach().numpy() > threshold:
                data_arr.append(data[samp_indx].detach().cpu().numpy())
                label_arr.append(conf_label)
                gt_arr.append(gt_label[samp_indx].detach().cpu().numpy())
                prob_arr.append(max_sim.detach().cpu().numpy())

                conf_correct += torch.sum(torch.tensor(conf_label).detach().cpu() == gt_label[samp_indx].detach().cpu())
                conf_index += 1

    data_arr = np.asarray(data_arr)
    prob_arr = np.asarray(prob_arr)
    label_arr = np.asarray(label_arr)
    gt_arr = np.asarray(gt_arr)

    sim_list_all = np.asarray(sim_list_all)
    print(len(sim_list_all[sim_list_all > threshold]))

    conf_acc = conf_correct.double() / conf_index
    # print(sim_list_all)
    print("Conf Acc: ", conf_acc)
    
    return data_arr, prob_arr, label_arr, gt_arr

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
        
        acc, acc_top2 = test(model, target_loader, args.batch_size)
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