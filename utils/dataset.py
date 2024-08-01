import os
import csv
import numpy as np
import torch
import torch.utils.data as Data
from os.path import join, exists
from .data.data import get_data, get_data_for_test
import sys

sys.path.append('')

def read_labels_from_csv(file_path):
    labels = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_name = row['file']
            label = int(row['label'])
            labels[file_name] = label
    return labels

def get_labels_for_files(file_path, labels):
    def get_label_from_path(labels, file_path):
        file_name, _ = os.path.splitext(os.path.basename(file_path))
        prefix = file_name[:5]

        for key, value in labels.items():
            if key.startswith(prefix):
                return value

    results = []
    for filename in os.listdir(file_path):
        if os.path.isfile(os.path.join(file_path, filename)):
            file_full_path = os.path.join(file_path, filename)
            label = get_label_from_path(labels, file_full_path)
            results.append((label))

    return results

labels = read_labels_from_csv('')

class Dataset:
    def __init__(self, name, add_root='./datasets'):
        self.add_root = add_root
        self.name = name
        self.add = []
        self.if_inited = True
        self.init_add()
        assert self.if_inited

    def init_add(self):
        if self.name == 'AVEC-org':
            self.add.append(join(self.add_root, self.name))
        else:
            print("", self.name, "")
        return

    def load_data_training_all_interface(self, block_size, batch_size, split_name):
        assert split_name in ['train', 'val', 'test']
        samples = None
        samples_diff = None
        label = read_labels_from_csv('')

        for add_a in self.add:
            value = get_labels_for_files(join(add_a, split_name), label)
            value = np.array(value)
            a_samples, a_samples_diff = get_data(join(add_a, split_name), block_size)
            if samples is None:
                samples = a_samples
                samples_diff = a_samples_diff
                if split_name == 'val':
                    break
            else:
                samples = np.concatenate((samples, a_samples), axis=0)
                samples_diff = np.concatenate((samples_diff, a_samples_diff), axis=0)

        samples = torch.tensor(samples, dtype=torch.float32)
        samples_diff = torch.tensor(samples_diff, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.long)
        value = value.unsqueeze(0).expand(samples.size(0), -1, -1)

        dataset_A = Data.TensorDataset(samples, value)
        dataset_B = Data.TensorDataset(samples_diff, value)
        dataset_iter_A = Data.DataLoader(dataset_A, batch_size, shuffle=True)
        dataset_iter_B = Data.DataLoader(dataset_B, batch_size, shuffle=True)

        return dataset_iter_A, dataset_iter_B

    def load_data_training_interface(self, block_size, batch_size, split_name, branch_selection):
        assert split_name in ['train', 'val', 'test']
        assert branch_selection in ['g1', 'g2']
        samples = None

        label = read_labels_from_csv('')

        for add_a in self.add:
            if branch_selection == 'g1':
                value = get_labels_for_files(join(add_a, split_name), label)
                value = np.array(value)
                a_samples, _ = get_data(join(add_a, split_name), block_size)
            else:
                _, a_samples = get_data(join(add_a, split_name), block_size)

            if samples is None:
                samples = a_samples
                if split_name == 'val':
                    break
            else:
                samples = np.concatenate((samples, a_samples), axis=0)

        samples = torch.tensor(samples, dtype=torch.float32)
        value = torch.tensor(value, dtype=torch.long)
        value = value.unsqueeze(0).expand(samples.size(0), -1, -1)

        dataset = Data.TensorDataset(samples, value)
        dataset_iter = Data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True)

        return dataset_iter

    def load_data_train_all(self, block_size, batch_size):
        return self.load_data_training_all_interface(block_size, batch_size, 'train')

    def load_data_val_all(self, block_size, batch_size):
        return self.load_data_training_all_interface(block_size, batch_size, 'val')

    def load_data_test_all(self, block_size, batch_size):
        return self.load_data_training_all_interface(block_size, batch_size, 'test')

    def load_data_train_g1(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'train', 'g1')

    def load_data_val_g1(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'val', 'g1')

    def load_data_test_g1(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'test', 'g1')

    def load_data_train_g2(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'train', 'g2')

    def load_data_val_g2(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'val', 'g2')

    def load_data_test_g2(self, block_size, batch_size):
        return self.load_data_training_interface(block_size, batch_size, 'test', 'g2')
