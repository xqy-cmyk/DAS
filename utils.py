import os
import numpy as np
import torch

def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))


def r2_score(y_true, y_pred):
    mask = y_true == y_true
    a, b = y_true[mask], y_pred[mask]
    unexplained_error = np.nansum(np.square(a-b))
    total_error = np.nansum(np.square(a - np.nanmean(a)))
    return 1. - unexplained_error/total_error

def nanunbiased_rmse(y_true, y_pred):
    predmean = np.mean(y_pred)
    targetmean = np.mean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.mean((predanom-targetanom)**2))

def _rmse(y_true,y_pred):
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))

def _bias(y_true,y_pred):
    bias = np.nanmean(np.abs(y_pred-y_true))
    return bias

def _rv(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    rv = np.sqrt(np.mean((y_pred - mean_pred) ** 2) / np.mean((y_true - mean_true) ** 2))

    return rv


def _fhv(y_true, y_pred):
    combined = np.column_stack((y_true, y_pred))
    sorted_combined = combined[combined[:, 1].argsort()]
    percentile_2 = int(0.02 * len(y_true))
    fhv_values = sorted_combined[-percentile_2:]
    fhv = np.mean((fhv_values[:, 1] - fhv_values[:, 0]) / fhv_values[:, 0])

    return fhv


def _flv(y_true, y_pred):
    combined = np.column_stack((y_true, y_pred))
    sorted_combined = combined[combined[:, 1].argsort()]
    percentile_30 = int(0.3 * len(y_true))
    flv_values = sorted_combined[:percentile_30]
    flv = np.mean((flv_values[:, 1] - flv_values[:, 0]) / flv_values[:, 0])
    return flv

def _ACC(y_true,y_pred):
    y_true_anom = y_true-np.nanmean(y_true)
    y_pred_anom = y_pred-np.nanmean(y_pred)
    numerator = np.sum(y_true_anom*y_pred_anom)
    denominator = np.sqrt(np.sum(y_true_anom**2))*np.sqrt(np.sum(y_pred_anom**2))
    acc = numerator/denominator
    return acc


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt



def GetKGE(Qs, Qo):
    # Input variables
    # Qs: Simulated runoff
    # Qo: Observed runoff
    # Output variable
    # KGE: Kling-Gupta edddddfficiency coefficient
    if isinstance(Qs, torch.Tensor):
        Qs = Qs.cpu()
        Qo = Qo.cpu()
        Qs = Qs.numpy()  # Convert Qs to a NumPy array
        Qo = Qo.numpy()  # Convert Qo to a NumPy array

    if len(Qs) == len(Qo):
        mask = Qo != 0
        Qo = Qo[mask]
        Qs = Qs[mask]
        QsAve = np.mean(Qs)
        QoAve = np.mean(Qo)
        # COV = np.cov(np.array(Qs).flatten(), np.array(Qo).flatten())
        # COV = np.cov(Qs, Qo)
        # CC = COV[0, 1] / np.std(Qs) / np.std(Qo)
        CC = np.corrcoef(Qo, Qs)[0, 1]
        BR = QsAve / QoAve
        RV = (np.std(Qs) / QsAve) / (np.std(Qo) / QoAve)
        KGE = 1 - np.sqrt((CC - 1) ** 2 + (BR - 1) ** 2 + (RV - 1) ** 2)
        return KGE
from scipy.stats import pearsonr


def GetRMSE(y_pred,y_true):
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))

def GetMAE(y_pred,y_true):
    n = len(y_true)
    mae = sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n
    return mae

def GetPCC(Qs, Qo):
    if isinstance(Qs, torch.Tensor):
        Qs = Qs.cpu()
        Qo = Qo.cpu()
        Qs = Qs.numpy()  # Convert Qs to a NumPy array
        Qo = Qo.numpy()  # Convert Qo to a NumPy array
    if len(Qs) == len(Qo):
        mask = Qo != 0

        Qs[np.isnan(Qs)] = 0
        Qo[np.isnan(Qo)] = 0

        PCC, _ = pearsonr(Qs, Qo)
        return PCC

def GetNSE(simulated,observed):

    if isinstance(observed, torch.Tensor):
        observed = observed.cpu()
        simulated = simulated.cpu()
        observed = observed.numpy()
        simulated = simulated.numpy()
    mask = observed != 0
    observed = observed[mask]
    simulated = simulated[mask]
    mean_observed = np.mean(observed)
    numerator = np.sum((simulated - observed)**2)
    denominator = np.sum((observed - mean_observed)**2)
    nse = 1 - numerator / denominator
    return nse

def _plotloss(cfg,epoch_losses):
    imgPath = 'D:\jsj\yanyuguang\lg\lgg_new\Loss_img'
    plt.switch_backend('Agg') 

    plt.figure(figsize=(12.8, 7.2))
    for i in range(cfg['num_repeat']):
        plt.plot(epoch_losses[i], label=cfg['label'][i]) 
    # plt.plot(epoch_losses_sum, 'b', label='Sum of Losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()        
    plt.savefig(cfg['Loss_path']+'_loss.png')

def _plotbox(cfg, epoch_losses):
    fig, ax = plt.subplots(figsize=(12, 8)) 
    data = []
    for i in range(cfg['num_repeat']):
        data.append(np.nan_to_num(np.nanmean(epoch_losses[i], axis=1)))
    ax.boxplot(data)
    ax.set_title('MSE')
    ax.set_xticks(range(1, len(cfg['label']) + 1))
    ax.set_xticklabels(cfg['label'])  
    plt.savefig(cfg['Loss_path']+'_boxMSE.png')

def _boxkge(cfg, epoch_losses):
    fig, ax = plt.subplots(figsize=(12, 8))  
    data = []
    for i in range(cfg['num_repeat']):
        data.append(np.nan_to_num(np.nanmean(epoch_losses[i], axis=1)))
    ax.boxplot(data)
    ax.set_title('KGE')
    ax.set_xticks(range(1, len(cfg['label']) + 1))
    ax.set_xticklabels(cfg['label'])  
    plt.savefig(cfg['Loss_path']+'_boxKGE.png')

def _boxpcc(cfg, epoch_losses):
    fig, ax = plt.subplots(figsize=(12, 8))  
    data = []
    for i in range(cfg['num_repeat']):
        data.append(np.nan_to_num(np.nanmean(epoch_losses[i], axis=1)))
    ax.boxplot(data)
    ax.set_title('PCC')
    ax.set_xticks(range(1, len(cfg['label']) + 1))
    ax.set_xticklabels(cfg['label']) 
    plt.savefig(cfg['Loss_path']+'_boxpcc.png')


def _boxnse(cfg, epoch_losses):
    fig, ax = plt.subplots(figsize=(12, 8))  
    data = []
    for i in range(cfg['num_repeat']):
        data.append(np.nan_to_num(np.nanmean(epoch_losses[i], axis=1)))
    ax.boxplot(data)
    ax.set_title('NSE')
    ax.set_xticks(range(1, len(cfg['label']) + 1))
    ax.set_xticklabels(cfg['label'])  
    plt.savefig(cfg['Loss_path']+'_boxNSE.png')

def _boxbias(cfg, epoch_losses):
    fig, ax = plt.subplots(figsize=(12, 8))  
    data = []
    for i in range(cfg['num_repeat']):
        data.append(np.nan_to_num(np.nanmean(epoch_losses[i], axis=1)))
    ax.boxplot(data)
    ax.set_title('Bias')
    ax.set_xticks(range(1, len(cfg['label']) + 1))
    ax.set_xticklabels(cfg['label']) 
    plt.savefig(cfg['Loss_path']+'_boxBias.png')
