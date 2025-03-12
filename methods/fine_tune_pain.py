# import comet_ml at the top of your file
from comet_ml import Experiment

import argparse
from ast import arg
import os, sys


import numpy as np
import torch
from tqdm import tqdm, trange
from torch import nn

"""
This will add all the root directory path in python env variable
parent methods can be imported from child classes 

*** NEED TO INCLUDE PATH IN __init__.py FILE ***

"""
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from datasets.base_dataset import BaseDataset
from models.sample_model import Net
from torchvision.models import resnet18, resnet50
from models.resnet_model_modified import ResNet50Fc
from utils.common import set_requires_grad, loop_iterable, do_epoch
import config

from utils.cometml import init_comet, log_parameters, log_metrics
from networks.transfer_net import TransferNet

# Create comet_ml experimentby defning Project Name
comet_exp = init_comet("fine-tune on pain dataset")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA_LAUNCH_BLOCKING=1
print(torch.__version__)
print(torch.cuda.device_count())

transfer_loss = 'coral'
learning_rate = 0.0001
n_class = 7

def main(args):

    # pain_model = Net(maxPool = 7).to(device)

    # pain_model = resnet50(pretrained=True)
    pain_model = TransferNet(n_class, transfer_loss=transfer_loss, base_net='resnet50').cuda()
    # pain_model = torch.nn.Sequential(*(list(source_model.children())[:-1]))

    # n_features = pain_model.fc.in_features
    # pain_model.fc = nn.Linear(n_features, 7)

    # pain_model.load_state_dict(torch.load(args.trained_model))

    """ 
        Custom Network: TransferNet
        update classifier last layer with 6 classes ONLY when using custom TransferNet

        pain_model.base_network._ResNet50Fc__in_features: is the input feature from ResNet50 model to the Sequential layer(s)
        pain_model.classifier_layer[3].in_features: is the input feature inside the Sequential Linear layer 
    """
    classifier_layer_list = [nn.Linear(pain_model.base_network._ResNet50Fc__in_features, pain_model.classifier_layer[3].in_features), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(pain_model.classifier_layer[3].in_features, 6)]
    pain_model.classifier_layer = nn.Sequential(*classifier_layer_list)

    # n_features = pain_model.fc.in_features
    # fc = torch.nn.Linear(n_features, 6)
    # pain_model.fc = fc

    pain_model.to(device)
    
    clf = pain_model

    train_loader, val_loader = BaseDataset.load_pain_dataset(args.dataset_path, args.label_path_train, args.label_path_val, args.batch_size)

    optim = torch.optim.Adam(pain_model.parameters())
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.BCEWithLogitsLoss()

    best_accuracy = 0
    with comet_exp.train():
        """
        Log comet_ml parameters:
            comet_exp
            network
            pretrained_model
            fine_tune_dataset
            source_dataset
            target_dataset
            batch size
            iterations
            epochs
        """
        log_parameters(comet_exp, 'ResNet50', 'ImageNet', config.PAIN_BIOVID, 'RAF and FER', None, args.batch_size, args.iterations, args.epochs)

        for epoch in range(1, args.epochs+1):
            pain_model.train()
            for iter in range(1, args.batch_size+1):
                train_loss, train_accuracy = do_epoch(pain_model, train_loader, criterion, device, optim=optim)
                tqdm.write(f'Iteration {iter:03d}:' f'train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} ')

                log_metrics(comet_exp, "Train loss:", "Train accuracy:", train_loss, train_accuracy)

            pain_model.eval()
            with torch.no_grad():
                val_loss, val_accuracy = do_epoch(pain_model, val_loader, criterion, device, optim=None)
            
            log_metrics(comet_exp, "Val loss:", "Val accuracy:", val_loss, val_accuracy)

            tqdm.write(f'EPOCH {epoch:03d}:' f'val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}')

            if val_accuracy > best_accuracy:
                print('Saving model...')
                best_accuracy = val_accuracy
                torch.save(pain_model.state_dict(), 'trained_models/pain_McMaster_model.pt')

            lr_schedule.step(val_loss)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Adapt to target domain')
    arg_parser.add_argument('--trained_model', help='A model in trained_models')
    arg_parser.add_argument('--dataset_path', type=str, default=config.BIOVID_PATH)
    arg_parser.add_argument('--label_path_train', type=str, default=config.BIOVID_LABEL_PATH)
    arg_parser.add_argument('--label_path_val', type=str, default=None)
    arg_parser.add_argument('--batch_size', type=int, default=16)
    arg_parser.add_argument('--iterations', type=int, default=10)
    arg_parser.add_argument('--epochs', type=int, default=10)
    arg_parser.add_argument('--k_disc', type=int, default=1)
    arg_parser.add_argument('--k_clf', type=int, default=10)
    args = arg_parser.parse_args()
    main(args)