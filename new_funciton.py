import os.path
import random

import numpy
import numpy as np
import shutil
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt


def signal_generator(k1, k2, t, B):
    signal = k1 * np.exp(-k2 * t) + B
    return signal


def add_gauss_noise(signal, scale):
    noise = np.random.normal(1, scale, signal.shape)
    return noise


def add_random_noise(signal, len):
    noise_local_1 = random.randint(50,890)
    noise_local_2 = random.randint(50,890)
    if noise_local_2 == noise_local_1:
        noise_local_2 = random.randint(50, int(signal.shape[-1]-50))
    noise = np.zeros([len])
    noise[noise_local_1] = 10000 * random.uniform(0, 1)
    noise[noise_local_2] = 20000 * random.uniform(0, 1)
    return noise

def add_power_frequency(signal):
    t = np.arange(0, 1, 1/signal.shape[-1])
    har1 = 1000 * np.sin(np.pi * 50 * t + np.random.normal(0.1, 1, size=1))
    har2 = 1500 * np.sin(np.pi * 150 * t + np.random.normal(0.1, 1, size=1))
    har3 = 2000 * np.sin(np.pi * 100 * t + np.random.normal(0.1, 1, size=1))
    return har1 + har2 + har3

def data_generator(len):
    range_ = 0
    k1 = random.randint(30000, 120000)
    k2 = round(random.uniform(1.4, 6.4), 1)
    B = random.randint(2000, 4000)
    t = np.linspace(1,1024, len)
    scale = random.randint(1000, 1500)
    signal = signal_generator(k1, k2, t/100, B)
    gauss_noise = add_gauss_noise(signal, scale)
    random_noise = add_random_noise(signal, len)
    harmonic_noise = add_power_frequency(signal)
    noised_signal = torch.zeros([len])
    noised_signal += signal
    noised_signal[range_:] += gauss_noise[range_:]
    noised_signal += random_noise
    noised_signal[range_:] += harmonic_noise[range_:]
    return noised_signal, signal


def data_set(data_name, label_name, num, len):
    data_name_dir = f"{data_name}"
    label_name_dir = f"{label_name}"
    if os.path.exists(data_name_dir):
        shutil.rmtree(data_name_dir)
    if os.path.exists(label_name_dir):
        shutil.rmtree(label_name_dir)
    os.makedirs(data_name_dir)
    os.makedirs(label_name_dir)
    for i in range(num):
        data, label = data_generator(len)
        # data_ = []
        # label_ = []
        # data_.append(data)
        # label_.append(label)
        # data_ = np.array(data_)
        # label_ = np.array(label_)
        file_data_path = os.path.join(data_name_dir, f"data_{i}.npy")
        file_label_path = os.path.join(label_name_dir, f"label_{i}.npy")
        np.save(file_data_path, data)
        np.save(file_label_path, label)

def data_load(data_name):
    folder_path = f'{data_name}'
    file_list = os.listdir(folder_path)
    file_list.sort()
    data_list = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
            data_list.append(data)
    return data_list

def batch_normalize(data, label):
    min_val = (label).min()
    max_val = (label).max()
    data = (data - min_val) / (max_val - min_val)
    label = (label - min_val) / (max_val - min_val)
    scale = [min_val, max_val]
    return torch.Tensor(data), torch.Tensor(label), torch.Tensor(scale)

def normalize(data, label):
    batch_size, _ = data.shape
    data = torch.Tensor(data)
    label = torch.Tensor(label)
    scale = torch.zeros((batch_size,2))
    for i in range(batch_size):
        min_val_=min(label[i]).item()
        max_val_=max(label[i]).item()
        data[i] = (data[i] - min_val_) / (max_val_-min_val_)
        label[i] = (label[i] - min_val_) / (max_val_-min_val_)
        print(min_val_)
        scale[i][0] = min_val_
        scale[i][1] = max_val_

    return torch.Tensor(data), torch.Tensor(label), scale


def reverse(data,label,pre_label,scale):
    batch_size, _, _ = data.shape
    data_ = torch.zeros(batch_size,1,1024)
    label_ = torch.zeros(batch_size,1,1024)
    pre_label_ = torch.zeros(batch_size,1,1024)

    for i in range(data.shape[0]):
        min_val = scale[i][0]
        max_val = scale[i][1]
        data_[i] = data[i] * (max_val - min_val) + min_val
        label_[i] = label[i] * (max_val - min_val) + min_val
        pre_label_[i] = pre_label[i] * (max_val - min_val) + min_val
    return data_, label_, pre_label_


def split_normalize(data, label):
    # 创建新变量避免修改原始数据
    data_early = torch.zeros((data.shape[0], 512))
    label_early = torch.zeros((data.shape[0], 512))
    data_later = torch.zeros((data.shape[0], 512))
    label_later = torch.zeros((data.shape[0], 512))

    scale1 = []
    scale2 = []

    for i in range(data.shape[0]):
        # ========= 早期部分处理 =========
        # 使用.clone()创建数据副本
        early_data = data[i][0:512].clone()
        early_label = label[i][0:512].clone()

        # 早期归一化
        min_val1 = early_label.min()
        max_val1 = early_label.max()
        data_early[i] = (early_data - min_val1) / (max_val1 - min_val1)
        label_early[i] = (early_label - min_val1) / (max_val1 - min_val1)
        scale1.append((min_val1.item(), max_val1.item()))

        # ========= 晚期部分处理 =========
        # 创建新的副本用于后期处理
        later_data = data[i][512:].clone().float()
        later_label = label[i][512:].clone().float()

        # 对数转换（保持原始数据不变）
        later_data = torch.log10(later_data)
        later_label = torch.log10(later_label)

        # 晚期归一化
        min_val2 = later_label.min()
        max_val2 = later_label.max()
        data_later[i] = (later_data - min_val2) / (max_val2 - min_val2)
        label_later[i] = (later_label - min_val2) / (max_val2 - min_val2)
        scale2.append((min_val2.item(), max_val2.item()))

    return (data_early, label_early), (data_later, label_later), torch.Tensor(scale1), torch.Tensor(scale2)

def split_reverse(early, later, scale1, scale2):
    data = torch.zeros((64,1024))
    label = torch.zeros((64,1024))
    for i in range(early[0].shape[0]):
        min_val1 = scale1[i][0]
        max_val1 = scale1[i][1]
        min_val2 = scale2[i][0]
        max_val2 = scale2[i][1]
        early[0][i] = early[0][i] * (max_val1-min_val1) + min_val1
        early[1][i] = early[1][i] * (max_val1-min_val1) + min_val1


        later[0][i] = later[0][i] * (max_val2-min_val2) + min_val2
        later[1][i] = later[1][i] * (max_val2-min_val2) + min_val2
        later[0][i] = 10 ** later[0][i]
        later[1][i] = 10 ** later[1][i]
        data[i] = torch.cat((early[0][i], later[0][i]))
        label[i] = torch.cat((early[1][i], later[1][i]))
    return torch.Tensor(data), torch.Tensor(label)

def data_cat(early, later, scale1, scale2):
    for i in range(early.shape[0]):
        min_val1 = scale1[0]
        max_val1 = scale1[1]
        min_val2 = scale2[0]
        max_val2 = scale2[1]
        early = early * (max_val1-min_val1) + min_val1
        later = later * (max_val2-min_val2) + min_val2
        later = 10 ** later
        return torch.cat((early, later), dim=1)
    # def Standardization(data, label, new_label):



def loss_function1(pre_label, true_label):
    batch = true_label.shape[0]
    seq_len = true_label.shape[-1]

    MSE = torch.nn.MSELoss()

    tv_loss = 0.1 * torch.mean(torch.abs(pre_label[:, 1:] - pre_label[:, :-1]))
    laplacian = pre_label[:, 2:] - 2*pre_label[:,1:-1] + pre_label[:, : -2]
    lap_loss = 0.1 * torch.mean(torch.square(laplacian))
    loss = MSE(pre_label, true_label)

    return loss + tv_loss + lap_loss


def loss_function1_later(pre_label, true_label):
    batch = true_label.shape[0]
    seq_len = true_label.shape[-1]

    weights = torch.ones(batch, seq_len).to('cuda:0')

    weights[:, 400:800] = weights[:, 400:800] * 256



    pre_label = pre_label.view(batch, seq_len)
    tv_loss = torch.mean(torch.abs(pre_label[:, 1:] - pre_label[:, :-1]))
    laplacian = pre_label[:, 2:] - 2 * pre_label[:, 1:-1] + pre_label[:, : -2]
    lap_loss = torch.mean(torch.square(laplacian))
    loss = torch.mean(weights * ((true_label  - pre_label) ** 2))

    return loss + 0.1 * lap_loss + 0.01 * tv_loss

def loss_function2(pre_label,true_label):
    batch = true_label.shape[0]
    seq_len = true_label.shape[-1]

    pre_label = pre_label.view(batch, seq_len)

    loss = torch.nn.MSELoss()

    Loss = loss(pre_label, true_label)

    return Loss


def exponential_time_weight(seq_len, decay_factor=0.99):
    position = torch.arange(seq_len, dtype=torch.float32)
    weight = decay_factor ** (seq_len - position)
    weight = weight / torch.max(weight)
    return weight

# def loss_function2(pre_label, true_label):
#     batch_size, seq_len = pre_label.shape
#     time_weight = exponential_time_weight(1024)
#     time_weight = time_weight.to(pre_label.device).unsqueeze(0).expand(batch_size, -1)
#     # max_amp = torch.max(true_label.abs(), dim=1, keepdim=True)
#     # rel_amp = pre_label.abs() / (max_amp)
#     # amp_weight = torch.exp(-2 * rel_amp)
#     combined_weight = time_weight
#
#     weight_mes = combined_weight * (pre_label - true_label) ** 2
#     return torch.mean(weight_mes)

def loss_function3(pre, label, epoch, total_epochs,
                   early_start=1.0,late_start=0.5,
                   early_end=0.5, late_end=5.0, switch_epoch=20):
    batch_size, seq_len = label.shape
    position = torch.arange(seq_len, dtype=torch.float32, device=label.device).unsqueeze(0)
    split_point = int(seq_len * 0.25)
    if epoch < switch_epoch:
        early_weight = early_start
        late_weight = late_start
    else:
        progress = min
#

# data_set('train_data', 'train_label', 64000, len=1024)

# data_set('test_data', 'test_label', 12800, len=1024)
# print('done')
