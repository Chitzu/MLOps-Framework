import json
import torchvision
from data.base_dataset import BaseDataset
import numpy as np
import torch
import torchvision.transforms as transforms
import pandas as pd

class DataManager:
    def __init__(self, config):
        self.config = config

    def get_train_eval_test_dataloaders(self):
        np.random.seed(707)

        train_transform = transforms.Compose([
                        transforms.Resize((512, 512)),  
                        transforms.ToTensor()       
        ])

        test_transform = transforms.Compose([
                        transforms.Resize((512, 512)),  
                        transforms.ToTensor()      
         ])

        train_df = 'data/train.csv'
        test_df = 'data/test.csv'

        trainset = BaseDataset(train_df, train_transform)
        testset = BaseDataset(test_df, test_transform)

        dataset_size = len(trainset)
        train_split = self.config['train_size']

        train_size = int(train_split * dataset_size)
        valid_size = int(dataset_size - train_size)
        test_size = len(testset)

        indices = list(range(dataset_size))
        indices_test = list(range(test_size))
        np.random.shuffle(indices)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size: (train_size + valid_size)] 
        test_indices = indices_test[:]      

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)

        train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=trainset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        test_loader = torch.utils.data.DataLoader(dataset=testset,
                                                  batch_size=self.config['batch_size'],
                                                  sampler=test_sampler,
                                                  pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader, test_loader


