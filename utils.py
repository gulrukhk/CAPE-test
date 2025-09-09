import numpy as np
import matplotlib
import sys
#import viz
from collections import defaultdict
import matplotlib.pyplot as plt
import ecg_plot
import os
import time

import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model, Sequential

from scipy import signal
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import *

# 12-LEAD ECG LEAD NAMES
ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
ECG_8_LEADS = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# COLORMAP TEMPLATES FOR HORIZONTAL AND VERTICAL LEADS
LEAD_COLORS = {
    'I': plt.get_cmap('viridis', 12)(2),  # 0 degrees
    'II': plt.get_cmap('viridis', 12)(4),  # 60 degrees
    'III': plt.get_cmap('viridis', 12)(6),  # 120 degrees
    'aVR': plt.get_cmap('viridis', 12)(9),  # 150 degrees
    'aVL': plt.get_cmap('viridis', 12)(1),  # -30 degrees
    'aVF': plt.get_cmap('viridis', 12)(5),  # 90 degrees
    'V1': plt.get_cmap('plasma', 12)(2),
    'V2': plt.get_cmap('plasma', 12)(3),
    'V3': plt.get_cmap('plasma', 12)(4),
    'V4': plt.get_cmap('plasma', 12)(5),
    'V5': plt.get_cmap('plasma', 12)(6),
    'V6': plt.get_cmap('plasma', 12)(7)
}
#LEAD_COLORS = defaultdict(lambda: 'black', LEAD_COLORS)

def sample_pairs(X, n=4, repeat=0, extend=1):
    if repeat:
      return np.random.choice(ind, size = (n, 2))
    else:
        pair_indices = np.random.randint(X.shape[0]**2, size=int(extend*n))
        pair_indices = np.hstack(((pair_indices // X.shape[0]).reshape((-1,1)), (pair_indices % X.shape[0]).reshape((-1,1))))
        good_indices = pair_indices[:,0] != pair_indices[:,1]
        good_pairs = X[pair_indices[good_indices]]
        while good_pairs.shape[0] < n:
          pair_indices = np.random.randint(X.shape[0]**2, size=int(extend*n))
          pair_indices = np.hstack(((pair_indices // X.shape[0]).reshape((-1,1)), (pair_indices % X.shape[0]).reshape((-1,1))))
          good_indices = pair_indices[:,0] != pair_indices[:,1]
          good_pairs = np.concatenate((good_pairs, X[pair_indices[good_indices]]), axis=0)
        return good_pairs[:n]

def generate_indexes(data_ids, df_data, ind_dir, name='train', gen_func=sample_pairs, epochs=50, sl=10000, over_write=0):
    os.makedirs(ind_dir, exist_ok=True)
    filename = os.path.join(ind_dir, f'indexes_{name}_2.csv')
    indexes = np.zeros((len(data_ids), epochs, 2), dtype=np.uint)
    init= time.time()
    for n, id in enumerate(data_ids):
        samples = df_data.loc[df_data['id'] == id].index.values
        #samples = df_data.loc[df_data['id'] == id]['filename']
        #samples = u.get_indices_from_ids(df_data, samples, id='filename')

        if len(samples) >= 2:
            ind = gen_func(samples, n=epochs)
            indexes[n] = ind

        if (n + 1) % sl == 0:
            print(f'Processed {n + 1} IDs in {time.time()-init:.4f} sec')
    if over_write:
        with open(filename, 'w', newline='') as f:
            np.savetxt(f, indexes.reshape(indexes.shape[0], indexes.shape[1] * indexes.shape[2]), fmt='%s', delimiter=',')
        print(f'Indexes of shape = {indexes.shape} are saved to {filename}')
    return indexes

def load_indexes(ind_dir, name, num=0, epochs=50):
   filename = os.path.join(ind_dir, f'indexes_{name}.csv')
   if os.path.exists(filename):
        print(f'loading indexes from {filename}')
        indexes = np.loadtxt(filename, delimiter=',').astype(np.uint)
        indexes = indexes.reshape(indexes.shape[0], -1, 2)
        no_ids = indexes.shape[0]
        start_epoch = indexes.shape[1]
        print(f'indexes of shape {indexes.shape} are loaded for {start_epoch} epochs')
        indexes = indexes[:, :epochs, :]
        if num: indexes = indexes[:num, :, :]
        print(f'indexes of shape {indexes.shape} are returned')
        return indexes
   else: 
       print(f'The directory {ind_dir} is not found')

def get_indexes(data_ids, df_data, ind_dir=None, name ='train', gen_func = sample_pairs, epochs=50, sl=10000, overwrite=1):
  start_epoch = 0
  no_ids = data_ids.shape[0]
  saved_ids = 0
  indexes = np.full(3, None)
  if ind_dir:  
    filename = os.path.join(ind_dir, f'indexes_{name}.csv')
    if os.path.exists(filename) and overwrite==0:
        print(f'loading indexes from {filename}')
        indexes = np.loadtxt(filename, delimiter=',').astype(np.uint)
        indexes = indexes.reshape(indexes.shape[0], -1, 2)
        no_ids = indexes.shape[0]
        start_epoch = indexes.shape[1]
        print(f'indexes of shape {indexes.shape} are loaded for {start_epoch} epochs')
        if start_epoch < epochs:
          indexes = np.pad(indexes, ((0, 0), (0, epochs-start_epoch), (0, 0)), mode='constant')
          print(f'zero padded to shape ={indexes.shape}')         
        if no_ids < len(data_ids):
          indexes = np.pad(indexes, ((0, data_ids.shape[0]-no_ids), (0, 0), (0, 0)), mode='constant')
          print(f'zero padded to shape ={indexes.shape}')
        if (no_ids >= len(data_ids)) and (start_epoch >= epochs):
           indexes = indexes[:len(data_ids), :epochs, :] 
           print(f'indexes of shape {indexes.shape} are returned with {no_ids} ids and {epochs} epochs')
           return indexes
    else:
       safe_mkdir(ind_dir)
       print(f'creating {ind_dir} directory')
  if indexes.all()==None:
    indexes = np.zeros((len(data_ids), epochs, 2), dtype=np.uint)
  if start_epoch >= epochs:
    data_ids = data_ids[no_ids:]
  for n, id in enumerate(data_ids):   
    #print(n)
    samples = df_data.loc[df_data['id'] == id]['filename']
    ind = get_indices_from_ids(df_data, samples, id='filename')
    if n<no_ids and start_epoch < epochs:
      #ind = np.random.choice(ind, size = (epochs-start_epoch, 2))
      ind = gen_func(ind, n=epochs-start_epoch)
      indexes[n][start_epoch:] = ind
    else:
      #ind = np.random.choice(ind, size = (epochs, 2))
      ind = gen_func(ind, n=epochs)
      indexes[n] = ind
    if os.path.exists(ind_dir):
      if n%sl==0: 
        #print(n, id)
        arr = indexes[n-sl:n]
        wmode = 'w' if (n==0 and start_epoch!=epochs) else 'a'
        with open(filename, wmode, newline='') as f:          
          np.savetxt(f, indexes[n-sl:n].reshape(arr.shape[0], arr.shape[1]*arr.shape[2]), fmt='%s', delimiter=',')
        saved_ids = n
        #print(wmode)
        print(f'{indexes[n-sl:n].shape} : {indexes[:n].shape}) : {filename}')
      elif n==len(data_ids)-1:
        print(n, "last")
        arr = indexes[saved_ids:n+1]
        wmode = 'w' if (not os.path.isfile(filename)) else 'a'
        print(wmode)
        with open(filename, wmode, newline='') as f:          
          np.savetxt(f, indexes[saved_ids:n+1].reshape(arr.shape[0], arr.shape[1]*arr.shape[2]), fmt='%s', delimiter=',')
        print(f'indexes of shape = {indexes[saved_ids:n+1].shape} are saved to {filename}') 

  return indexes


def shuffle(arr, seed=None):
    if seed: np.random.seed(seed)
    np.random.shuffle(arr)
    return arr

def myprint(lines, line=[], to_print=1):
   '''
   lines: list to which the text should be appended
   line: text to display and add to list
   '''
   if to_print: print(line)
   lines.append(line+'\n')
   return lines
   
def to_file(text, txtfile=''):
   """
   This function writes text to a file.

   text: text to be written

   txtfile: file name
   """
   with open(txtfile, 'w') as file:
       file.writelines(text) 



# -----------------------------------------------ECG VISUALIZATION----------------------------------------------------#

def plot_ecg(ecg_signal, sampling_rate, lead_names=None, subplots=True, subplot_shape=None, ylim=None, share_ylim=True,
             title=None, std=None, percentiles=None, figsize=None, show_axes=True, **kwargs):
    """
    Plots ECG signal(s) in the time domain.

    Arguments:
        ecg_signal (ndarray): ECG signal(s) of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal(s).
        lead_names (list): List of lead names. If None, the leads will be named as Lead 1, Lead 2, etc.
        subplots (bool): If True, the ECG leads will be plotted in separate subplots.
        subplot_shape (tuple): Shape of the subplot grid. If None, the shape will be automatically determined.
        ylim (tuple): Y-axis limits of the plot.
        share_ylim (bool): If True, the y-axis limits of the subplots will be shared.
        title (str): Title of the plot.
        std (ndarray): Standard deviation of the ECG signal(s) of shape (num_samples, num_leads).
        percentiles (tuple): Percentiles of the ECG signal(s) of shape (2, num_samples, num_leads).
        figsize (tuple): Figure size.
        show_axes (bool): If True, the axes of the plot will be plotted.
        **kwargs: Additional arguments to be passed to the 
        
        .pyplot.plot function.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """
    # Check ECG signal shape
    if len(ecg_signal.shape) != 2:
        raise ValueError('ECG signal must have shape: (num_samples, num_leads)')

    # Get number of ECG leads and time_index vector
    time_index = np.arange(ecg_signal.shape[0]) / sampling_rate
    num_leads = ecg_signal.shape[1]

    # If share_ylim, find ECG max and min values
    ylim_ = None
    if ylim is not None:
        ylim_ = ylim
    if ylim is None and share_ylim is True:
        ylim_ = (np.min(ecg_signal), np.max(ecg_signal))

    # Check for Lead Names
    if lead_names is not None:
        # Check number of leads
        if len(lead_names) != num_leads:
            raise ValueError('Number of lead names must match the number of leads in the ECG data.')
        lead_colors = LEAD_COLORS
    else:
        lead_names = [f'Lead {i + 1}' for i in range(num_leads)]  # Lead x
        cmap = plt.get_cmap('cividis', num_leads)
        lead_colors = dict(zip(lead_names, cmap(np.linspace(0, 0.7, num_leads))))

    # Check subplot shape
    if subplots is True and num_leads > 1:
        if subplot_shape is not None:
            if subplot_shape[0] * subplot_shape[1] < num_leads:
                raise ValueError('Subplot shape is too small to fit all the leads.')
        else:
            subplot_shape = (num_leads//2, 2)

    if figsize is None:
        if subplots is True and num_leads > 1:
            figsize = (15, 6)
        else:
            figsize = (13, 4)

    # Plotting
    if subplots is True and num_leads > 1:
        fig, axes = plt.subplots(nrows=subplot_shape[0], ncols=subplot_shape[1], sharex='row', sharey='row',
                                 figsize=figsize)
        flat_axes = axes.T.flatten()

        for i in range(num_leads):
            flat_axes[i].plot(time_index, ecg_signal[:, i], c=lead_colors[lead_names[i]], **kwargs)
            flat_axes[i].set_ylim(ylim_)
            flat_axes[i].legend([lead_names[i]], loc='upper right')

            if std is not None:
                flat_axes[i].fill_between(time_index, ecg_signal[:, i] - std[:, i], ecg_signal[:, i] + std[:, i],
                                          alpha=0.2, color=lead_colors[lead_names[i]], label='_nolegend_')
            if percentiles is not None:
                flat_axes[i].fill_between(time_index, percentiles[0][:, i], percentiles[1][:, i], alpha=0.2,
                                          color=lead_colors[lead_names[i]], label='_nolegend_')
            if show_axes is False:
                _remove_ticks(flat_axes[i])

    else:
        fig = plt.figure(figsize=figsize)
        for i in range(num_leads):
            plt.plot(time_index, ecg_signal[:, i], c=lead_colors[lead_names[i]], **kwargs)

            if std is not None:
                plt.fill_between(time_index, ecg_signal[:, i] - std[:, i], ecg_signal[:, i] + std[:, i], alpha=0.2,
                                 color=lead_colors[lead_names[i]], label='_nolegend_')
            if percentiles is not None:
                plt.fill_between(time_index, percentiles[0][:, i], percentiles[1][:, i], alpha=0.2,
                                 color=lead_colors[lead_names[i]], label='_nolegend_')
        plt.ylim(ylim_)
        plt.legend(lead_names, loc='upper right')
        if show_axes is False:
            _remove_ticks(plt.gca())

    # Plot Labels
    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude (mV)")
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def my_ecg_plot(ecg, labels, file_name = None):
  if ecg.ndim > 2:
    print('the first ecg is plotted')
    ecg = ecg[0]
    print(f'shape = {ecg.shape}')
  samples = ecg.shape[0]
  leads = ecg.shape[1]
  if leads ==1:
    fig = plt.plot(ecg[:, 0], label=labels[0])
  else:
    fig, ax = plt.subplots(leads, 1, figsize=(10, 10))
    for l in np.arange(leads):
      ax[l].plot(ecg[:, l], label=labels[l])
      ax[l].legend()
  if file_name:
    plt.savefig(file_name)
  else:
    plt.show()
  
def hist_binary_labels(labels, names=None):
  plt.hist(labels, bins=[-.5,.5,1.5], ec="k")
  if names:
    plt.xticks([0, 1], names)
  else:
    plt.xticks([0, 1])
  plt.show()  
  
def get_vec(ecg, lead_names=['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
  if ecg.shape[-1] != 12:
    print(f'Vector can only be computed for 12 leads. There are {ecg.shape[1]} leads ')
    return
  else:
    leads ={}
    for l, index in enumerate(['V1', 'V2', 'V3', 'V4', 'V5', 'V6']):
      leads[index] = l + 6
    for l, index in enumerate(['DI', 'DII']):
      leads[index] = l
    if ecg.ndim==2: ecgs.expand_dims(ecgs, 0)
    X = -(-0.172 * ecg[:, :,leads['V1']] - 0.074 * ecg[:, :, leads['V2']] + 0.122 * ecg[:, :, leads['V3']] + 0.231 * ecg[:, :, leads['V4']] + 0.239 * ecg[:, :, leads['V5']] + 0.194 * ecg[:, :,  leads['V6']] + 0.156 * ecg[:, :, leads['DI']] - 0.010 * ecg[:, :, leads['DII']])
    Y = (0.057 * ecg[:, :, leads['V1']] - 0.019 * ecg[:, :, leads['V2']] - 0.106 * ecg[:, :, leads['V3']] - 0.022 * ecg[:, :, leads['V4']] + 0.041 * ecg[:, :, leads['V5']] + 0.048 * ecg[:, :,  leads['V6']] - 0.227 * ecg[:, :, leads['DI']] + 0.102 * ecg[:, :, leads['DII']])
    Z = -(-0.229 * ecg[:, :, leads['V1']] - 0.310 * ecg[:, :, leads['V2']] - 0.246  * ecg[:, :, leads['V3']] - 0.063 * ecg[:, :, leads['V4']] + 0.055 * ecg[:, :, leads['V5']] + 0.108 * ecg[:, :,  leads['V6']] + 0.022 * ecg[:, :, leads['DI']] + 0.102 * ecg[:, :, leads['DII']])
    
    return X, Y, Z
    
    
def from_xyz(xyz, axis=-1):
    x, y, z = np.moveaxis(xyz, axis, 0)

    lea = np.empty_like(xyz)

    pre_selector = ((slice(None),) * lea.ndim)[:axis]

    xy_sq = x ** 2 + y ** 2
    lea[(*pre_selector, 0)] = np.sqrt(xy_sq + z ** 2)
    lea[(*pre_selector, 1)] = np.arctan2(np.sqrt(xy_sq), z)
    lea[(*pre_selector, 2)] = np.arctan2(y, x)

    return lea


def to_xyz(lea, axis=-1):
    l, e, a = np.moveaxis(lea, axis, 0)

    xyz = np.empty_like(lea)

    pre_selector = ((slice(None),) * xyz.ndim)[:axis]

    xyz[(*pre_selector, 0)] = l * np.sin(e) * np.cos(a)
    xyz[(*pre_selector, 1)] = l * np.sin(e) * np.sin(a)
    xyz[(*pre_selector, 2)] = l * np.cos(e)

    return xyz
    
def scat_pred(y_test, y_pred, filename, mape=0):
     """
     This function makes a scatter plot for the actual and predicted outputs
     
     y_test: actual output

     y_pred: predicted output

     valpath : path to save the plot
     """
     y_test, y_pred= np.squeeze(y_test), np.squeeze(y_pred)
     fig = plt.figure( )
     #print('y_test', y_test[:10])
     #print('y_pred', y_pred[:10])
     error = np.absolute(y_test-y_pred)
     #print('error', error[:10])
     
     elabel = f'error = {np.mean(error):.4f}$\pm${np.std(error):.4f}'
     if mape: 
       mape = error/y_test
       elabel = elabel + f'  MAPE = {np.mean(mape):.4f}$\pm${np.std(mape):.4f}'
     plt.scatter(y_test, y_pred, label=elabel)
     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], label=f'truth', color='r', linestyle = 'dotted')     
     plt.legend()
     plt.xlabel("actual")
     plt.ylabel("predicted")
     plt.savefig(filename)
        
def hist2d(y_test, y_pred, filename, mape=0, bins=20):
     """
     This function makes a 2d hist plot for the actual and predicted outputs
     
     y_test: actual output

     y_pred: predicted output

     valpath : path to save the plot
     """
     y_test, y_pred= np.squeeze(y_test), np.squeeze(y_pred)
     fig = plt.figure( )
     error = np.absolute(y_test-y_pred)   
     elabel = f'error = {np.mean(error):.4f}$\pm${np.std(error):.4f}'
     if mape: 
       mape = error/y_test
       elabel = elabel + f'  MAPE = {np.mean(mape):.4f}$\pm${np.std(mape):.4f}'
     plt.hist2d(y_test, y_pred, label=elabel, bins=(bins, bins), cmap=plt.cm.jet)
     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], label=f'truth', color='r', linestyle = 'dotted')     
     plt.legend()
     plt.colorbar()
     plt.xlabel("actual")
     plt.ylabel("predicted")
     plt.savefig(filename)
    
def errorhist(y_test, y_pred, filename, name='error', weighted = 0, bins=20, density = False, norm=0):
     """
     This function makes a 1d hist plot for the error
     
     valpath : path to save the plot
     """
     y_test, y_pred= np.squeeze(y_test), np.squeeze(y_pred)
     fig = plt.figure( )
     error = y_test-y_pred  
     if name != 'error': 
        error = np.absolute(error)
     if name == 'mape':
        error = np.where(y_test > 0, error/y_test, 0)
     elabel = f'{name} = {np.mean(error):.4f}$\pm${np.std(error):.4f}'
     if norm:
        values, bins =  np.histogram(y_test, weights = error, bins=bins) 
        counts, _ =  np.histogram(y_test, bins=bins)
        n_values = np.where(counts>0 , values/counts, 0)
        elabel = f'Binned {name} = {np.mean(n_values):.4f}$\pm${np.std(n_values):.4f}'
        plt.bar(bins[:-1], n_values,width=bins[1]-bins[0], label=elabel)
     elif weighted:
        plt.hist(y_test, weights = error, label=elabel, bins=bins, density = density) 
     else:
        plt.hist(error, label=elabel, bins=bins)          
     plt.legend()
     plt.xlabel(f"{name}")
     plt.savefig(filename)
    
def regression_test(y_test, y_pred, res_dir, label='', plot=0):
    if plot:
        scat_pred(y_test, y_pred, os.path.join(res_dir, f'scatter_{label}.png'), mape=1)
        hist2d(y_test, y_pred, os.path.join(res_dir, f'hist2d_{label}.png'), mape=1)
        #errorhist(y_test, y_pred, os.path.join(res_dir, f'error_{label}.png'))
    r2score = r2_score(y_test, y_pred)
    return r2score

def compute_class_metrics(y_true, y_pred, class_names=None, one_hot_encoding=True):
    """
    Computes classification metrics for binary or mutli-class classification problems.
    Class Metrics: precision, recall, f1-score, support
    Average Metrics: accuracy, macro avg, weighted avg

    Arguments:
        y_true (ndarray): Target outputs
        y_pred (ndarray): Predicted outputs
        class_names (list): List of class names
        one_hot_encoding (bool): Whether the targets are one-hot encoded or not

    Returns:
        class_metrics (dict): Dictionary containing classification metrics
    """

    # Check if one-hot encoded
    if one_hot_encoding:
        # Compute number of classes
        num_classes = len(np.unique(y_true, axis=0))

        # Transform one-hot encoded outputs to multi-class labels
        lb = LabelBinarizer()
        lb.fit(np.arange(num_classes))
        y_true = lb.inverse_transform(y_true.copy())
        y_pred = lb.inverse_transform(y_pred.copy())

    class_metrics = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    return class_metrics


def compute_confusion_matrix(y_true, y_pred, one_hot_encoding=True):
    """
    Computes confusion matrix for binary or mutli-class classification problems

    Arguments:
        y_true (ndarray): Target outputs
        y_pred (ndarray): Predicted outputs
        one_hot_encoding (bool): Whether the targets are one-hot encoded or not

    Returns:
        conf_matrix (ndarray): Confusion matrix
    """

    # Check if one-hot encoded
    if one_hot_encoding:
        # Compute number of classes
        num_classes = len(np.unique(y_true, axis=0))

        # Transform one-hot encoded outputs to multi-class labels
        lb = LabelBinarizer()
        lb.fit(np.arange(num_classes))
        y_true = lb.inverse_transform(y_true.copy())
        y_pred = lb.inverse_transform(y_pred.copy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    return conf_matrix


def compute_roc_curve(y_true, y_pred, one_hot_encoding=True):
    """
    Computes ROC curve (False positive rates/True positive rates) and AUC for binary classification problems

    Arguments:
        y_true (ndarray): Target outputs
        y_pred (ndarray): Predicted outputs
        one_hot_encoding (bool): Whether the targets are one-hot encoded or not

    Returns:
        roc_curve_auc (tuple): Tuple containing fpr, tpr and roc_auc
    """

    # Check if one-hot encoded
    if one_hot_encoding:
        # Compute number of classes
        num_classes = len(np.unique(y_true, axis=0))

        # Transform one-hot encoded outputs to multi-class labels
        lb = LabelBinarizer()
        lb.fit(np.arange(num_classes))
        y_true = lb.inverse_transform(y_true.copy())
        y_pred = y_pred.copy()[:, 1]  # Probability of positive class

    # Check if binary classification
    if len(np.unique(y_true)) != 2:
        raise ValueError('ROC curve can only be computed for binary classification problems.')

    # Compute Roc Curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    roc_curve_auc = (fpr, tpr, roc_auc)

    return roc_curve_auc



def classification_test(y_test, y_pred, res_dir, c_metrics, names=['0', '1'], label='', plot=0):
    # Compute Classification Metrics, Confusion Matrix, ROC Curve
    class_metrics = compute_class_metrics(y_test, y_pred, class_names=names)
    if plot:
        conf_matrix = compute_confusion_matrix(y_test, y_pred)
        roc_curve_auc = compute_roc_curve(y_test, y_pred, False)

        fig = plot_class_metrics(class_metrics, class_names=names,
                           metrics=['precision', 'recall', 'f1-score'], avg_key='macro avg',
                           title='Classification Metrics')
        fig.savefig(os.path.join(res_dir, f'metrics_{label}.pdf'))
        fig = plot_confusion_matrix(conf_matrix, class_names=names, normalize=False,
                              title='Confusion Matrix')
        fig.savefig(os.path.join(res_dir, f'confusion_{label}.pdf'))
        fig = plot_roc_curve(roc_curve_auc, title='ROC Curve')               
        fig.savefig(os.path.join(res_dir, f'roc_{label}.pdf'))
    f1_score = class_metrics['macro avg']['f1-score']
    return f1_score
     
def plot_metric(history, valpath, metric = 'mae'):
     """
     This function plots the Train-Test mae

     valpath : path to save the plot
     """
     plt.figure()
     plt.plot(history[metric], label=f'Training {metric}')
     plt.plot(history[f'val_{metric}'], label=f'Testing {metric}')
     plt.xlabel('epochs')
     plt.ylabel(metric)
     plt.legend()
     plt.savefig(os.path.join(valpath, f'train_{metric}.png'))

def plot_loss(loss, val_loss, label, valpath):
     """
     This function plots the Train-Test losses

     valpath : path to save the plot
     """
     plt.figure()
     plt.plot(loss, label=f'Training {label}')
     plt.plot(val_loss, label=f'Testing {label}')
     plt.xlabel('epochs')
     plt.ylabel(label)
     #plt.ylim(max([0.5, max(loss), max(val_loss)]))
     plt.legend()
     plt.savefig(os.path.join(valpath, f'train_{label}.png'))
        
def compare_loss(losses, labels, name, valpath, ymin=0, ymax=10, add_fit =0, start_epoch=0, end_epoch=200):
     """
     This function compares the Train-Test losses

     valpath : path to save the plot
     """
     plt.figure()
     plt.title(f'{name} loss')
     
     for loss, label in zip(losses, labels):
       last_epoch = min(end_epoch, len(loss))
       epochs = np.arange(start_epoch, last_epoch)
       plt.plot(epochs, loss[start_epoch: last_epoch], label=f'{label} (min={min(loss):0.6f})')
       if add_fit:
          z = np.poly1d(np.polyfit(epochs, loss[start_epoch: last_epoch], 1))
          plt.plot(epochs, z(epochs), color=plt.gca().lines[-1].get_color(), linestyle='--')
     #plt.plot(val_loss, label=f'Testing {label} ')
     plt.xlabel('epochs')
     plt.ylabel(name)
     #plt.ylim(max([0.5, max(loss), max(val_loss)]))
     plt.ylim(ymin, ymax)
     plt.legend()
     plt.savefig(os.path.join(valpath, f'loss_comparison_{name}.png'))
    
def get_indices_from_ids(ecg_labels, ids, id='filename'):
    """
    Returns the indices of the ecg_labels DataFrame for the given ids.

    Arguments:
        ecg_labels (DataFrame): ECG dataset labels.
        ids (DataFrame): List of ids.
        id (str): ECG ID column name.

    Returns:
        indices (ndarray): List of indices.
    """
    # Check if IDs exist in ecg_labels
    if not ids.isin(ecg_labels[id]).all():
        raise ValueError('Invalid Split Indices File')

    # Check if IDs are unique
    if not ids.is_unique:
        raise ValueError('IDs must be unique')

    # Get Indices from IDs (New Implementation)
    id_to_index = pd.Series(ecg_labels.index, index=ecg_labels[id])
    indices = id_to_index.loc[ids].tolist()

    return np.array(indices)
    
def build_MLP(input_shape, out_size, layers =2, units = 64, activation='relu', dropout_rate=0.2, out_activation='sigmoid'):
     """
     This function builds a MLP

     returns a model
     """
     mlp_model = Sequential() 
     if isinstance(units, list):
        if len(units) < layers:
            units = units + ([units[-1]] * (layers - len(units)))  
     else:
         units = [units] * layers
     layers = len(units)
     mlp_model.add(Dense(units[0], activation=activation, input_shape=input_shape))
     if dropout_rate:
         mlp_model.add(Dropout(dropout_rate))       
     for layer in np.arange(layers-1):
       mlp_model.add(Dense(units[layer+1], activation=activation))
       if dropout_rate:
         mlp_model.add(Dropout(dropout_rate))
     # output layer 
     mlp_model.add(Dense(out_size, activation=out_activation)) 
     return mlp_model

def build_linear_probe(input_shape, out_size, out_activation='sigmoid'):
     """
     This function builds a MLP

     returns a model
     """
     mlp_model = Sequential() 
     
     mlp_model.add(Dense(out_size, activation=out_activation, input_shape=input_shape))
     return mlp_model


def print_dict(dictionary, ident = '', braces=1, line=' '):
    """ 
    This function recursively prints nested dictionaries.

    ident, braces and line are required for recursive application 

    returns text
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            line = line + f'{ident} [{key}]\n'
            line = print_dict(value, ident+'  ', braces+1, line)
        else:
            line = line + ident+f'{key}\n'
    return line

def get_class_config(myclass, config=[]):
   """
   This function obtains the parameters values from a class as text and can also append it to provided text.

   myclass: the class whose configuration is required

   config: previous text to append the output with

   reurns the text with class configuration
   """
   import inspect
   # members of an object 
   for i in inspect.getmembers(myclass):      
     # to remove private and protected
     # functions
     if not (i[0].startswith('_') or i[0]=='stream'):          
        # To remove other methods that
        # doesnot start with a underscore
        if not (inspect.ismethod(i[1]) or hasattr(i[1], '__call__')): 
            line = f'{i[0]}: '
            if hasattr(i[1], 'name'):
              line = line + i[1].name + '\n'
            elif hasattr(i[1], 'shape'):
                line = line +  f' array of shape: {i[1].shape}\n' 
            else:                         
              line = line + f'{i[1]}  \n'              
            config.append(line)
   config.append('\n')
   return config

def get_sample_weights_from_targets(y, task = 'regression', one_hot_encoding=True, bins=20, apply_sqrt=False):
    """
    Extracts and returns the ECG sample weights based on the number of class instances

    Arguments:
        y (ndarray): ECG targets of shape (num_instances, 1).
        one_hot_encoding (bool): Whether the targets are one-hot encoded or not.

    Returns:
        Xtr_w (ndarray): Sample weights of shape (num_instances,).
    """
    #from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
    #from sklearn.preprocessing import LabelBinarizer
    if task == 'classification':
      # Check if one-hot encoded
      if one_hot_encoding:
        # Compute number of classes
        num_classes = len(np.unique(y, axis=0))

        # Transform one-hot encoded outputs to multi-class labels
        lb = LabelBinarizer()
        lb.fit(np.arange(num_classes))
        y = lb.inverse_transform(y)

      class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=np.squeeze(y))
      class_weights = dict(zip([c for c in range(len(np.unique(y)))], class_weights))
      sample_weights = compute_sample_weight(class_weights, y)
    else:
      label_weights = []
      # For each continuous weight label, compute the weight of each sample based on its histogram frequency
      freqs, bins = np.histogram(y, bins=bins)
      freqs = np.insert(freqs, 0, 1)  # Add freq of 1 at index 0 for samples in (-inf, bins[0])
      freqs = np.append(freqs, 1)  # Add freq of 1 at index -1 for samples in (bins[-1], inf)
      freqs[freqs == 0] = 1  # Replace freq of 0 with 1 to avoid division by 0
      sample_weights = 1 / freqs[np.digitize(y, bins)]
      # Apply Square Root to decrease the effect of large weights from label outliers
      if apply_sqrt:
        sample_weights = np.sqrt(sample_weights)

      # Normalize sample weights
      sample_weights = sample_weights / np.sum(sample_weights)
    return sample_weights


def safe_mkdir(path):
   '''
   This function creates a directory
   Safe mkdir (i.e., don't create if already exists,and no violation of race conditions)

   path: folder to create
   '''
   from os import makedirs
   from errno import EEXIST
   try:
       makedirs(path)
   except OSError as exception:
       if exception.errno != EEXIST:
           raise exception

LEADS = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]

BIDMC_leads = [ 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def process_ecg_8lead(ecg: np.ndarray, ecg_samples: int = 4096) -> np.ndarray:
    """
    Prepares an ECG for use in a tensorflow model
    :param ecg: A dictionary mapping lead name to lead values.
                The lead values should be measured in milli-volts.
                Each lead should represent 10s of samples.
    :param ecg_samples: Length of each lead for input into the model.
    :return: a numpy array of the ECG shaped (ecg_samples, 12)
    """
    #remove zero padding
    if ecg.ndim < 3:
      ecg = np.expand_dims(ecg, 0)
    ecg = ecg[:, 48:4048, :]
    #initialize processed array
    out = np.zeros((ecg.shape[0], ecg_samples, 12))
    lead_idx = [0, 1, 6, 7, 8, 9, 10, 11] # index of 8 lead format
    #exit()
    for i in np.arange(ecg.shape[0]):
      for j in np.arange(12):
        if j in lead_idx:
          interpolated_lead = np.interp(
             np.linspace(0, 1, ecg_samples),
             np.linspace(0, 1, ecg.shape[1]),
             ecg[i, :, lead_idx.index(j)],)
          out[i, :, j] = interpolated_lead
        elif j == LEADS.index('III'):
            out[i, :, j] = out[i, :, LEADS.index('II')] - out[i, :, LEADS.index('I')]
        elif j == LEADS.index('aVR'):
            out[i, :, j] = - ((out[i, :, LEADS.index('I')] + out[i, :, LEADS.index('II')]) / 2)
        elif j == LEADS.index('aVL'):
            out[i, :, j] = out[i, :, LEADS.index('I')] - (out[i, :, LEADS.index('II')] / 2)
        elif j == LEADS.index('aVF'):
            out[i, :, j] = out[i, :, LEADS.index('II')] - (out[i, :, LEADS.index('I')] / 2)
    return out

def bandpass_filtering(ecg_signal, sampling_rate, lowcut=None, highcut=None, order=3):
    """
    Applies a bandpass IIR butterworth filter to the ECG signal.
    The Gustafsson's method is used to determine the initial staes of forward-backward filtering (Fustafsson, 1996).

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        lowcut (float or None): Low cutoff frequency in Hz. If None, a low-pass filter is applied.
        highcut (float or None): High cutoff frequency in Hz. If None, a high-pass filter is applied.
        order (int): Order of the bandpass filter.

    Returns:
        ecg_signal_filtered (ndarray): Filtered ECG signal.
    """
    if lowcut is None and highcut is None:
        raise ValueError('At least one of lowcut or highcut must be specified')

    # Lowpass filter
    if lowcut is None:
        wn = highcut / (sampling_rate / 2)
        ba = signal.iirfilter(order, wn, btype='lowpass', analog=False, ftype='butter', output='ba')
    # Highpass filter
    elif highcut is None:
        wn = lowcut / (sampling_rate / 2)
        ba = signal.iirfilter(order, wn, btype='highpass', analog=False, ftype='butter', output='ba')
    # Bandpass filter
    else:
        wn = [lowcut / (sampling_rate / 2), highcut / (sampling_rate / 2)]
        ba = signal.iirfilter(order, wn, btype='bandpass', analog=False, ftype='butter', output='ba')

    ecg_signal_filtered = signal.filtfilt(ba[0], ba[1], ecg_signal, axis=0, method='gust')

    return ecg_signal_filtered


def notch_filtering(ecg_signal, sampling_rate, powerline_freq=50, harmonics=True):
    """
    Applies a notch filter to the ECG signal to remove the powerline interference.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        powerline_freq (int or None): Powerline frequency in Hz.
        harmonics (bool): If True, also remove the harmonics of the powerline frequency.

    Returns:
        ecg_signal_nopowerline (ndarray): ECG signal with no powerline interference.
    """
    q = 30.0  # Quality factor
    b, a = signal.iirnotch(powerline_freq, q, fs=sampling_rate)
    ecg_signal_nopowerline = signal.filtfilt(b, a, ecg_signal, axis=0, padtype='constant')

    # Remove the harmonics of the powerline frequency
    if harmonics:
        highest_harmonic_order = int(np.floor((sampling_rate / 2) / powerline_freq))
        for harmonic in range(2, highest_harmonic_order + 1):
            b, a = signal.iirnotch(harmonic * powerline_freq, q, fs=sampling_rate)
            ecg_signal_nopowerline = signal.filtfilt(b, a, ecg_signal_nopowerline, axis=0, padtype='constant')

    return ecg_signal_nopowerline


def resampling(ecg_signal, sampling_rate, new_sampling_rate, method='polyphase'):
    """
    Resamples the ECG signal to the new sampling rate. 'fourier' method uses the scipy.signal.resample function,
    while 'polyphase' uses the scipy.signal.resample_poly function.
    'polyphase' is preferable for small downsampling ratios, as well as to prevent aliasing effects.

    Arguments:
        ecg_signal (ndarray): ECG signal of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal.
        new_sampling_rate (int): New sampling rate of the ECG signal.
        method (str): Resampling method. Can be 'fourier' or 'polyphase'.

    Returns:
        ecg_signal_resampled (ndarray): Resampled ECG signal.
    """

    # Check if method is valid
    if method not in ['fourier', 'polyphase']:
        raise ValueError(f'Invalid resampling method {method}')

    ecg_signal_resampled = None
    # Calculate new number of samples
    new_num_samples = int(ecg_signal.shape[0] * (new_sampling_rate / sampling_rate))

    # Resample ecg_signal
    if method == 'fourier':
        ecg_signal_resampled = signal.resample(ecg_signal, new_num_samples, axis=0)
    elif method == 'polyphase':
        ecg_signal_resampled = signal.resample_poly(ecg_signal, new_num_samples, ecg_signal.shape[0], axis=0)

    return ecg_signal_resampled


def load_raw_data(df, sampling_rate, path):
    """
    Loads ecg in wfdb format

    Arguments:
        df (pandas dataframe): containing filenames as 'filename_lr'
        sampling_rate (int): Sampling rate of the ECG signal.
        path: the location of wfdb data
    Returns:
        ecg_signal_nopowerline (ndarray): ECG signal with no powerline interference.
    """
    import wfdb
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def create_dataset(ecgs, out_path, data_key = 'ECGs'):
  f_ecg = h5py.File(out_path, 'w')
  f_ecg.create_dataset(data_key, data=ecgs)
  f_ecg.close()
  print(f'ecgs saved in {ecg_file}')

def ecg_process(ecg, lead_in, lead_out, ecg_sampling= 500, filt=0, sampling_rate=400, num_samples=4096, pad=0):
    """
    Convert 8 to 12 lead ecg and interpolate to required number of samples
    :param ecg: nparray of shape (instances, samples, leads)
    :param ecg_sampling: original sampling rate
    :param num_leads: required number of leads 
    :param filt: 0: no filtering; 1: applying bandpass; 2: apply_notch
    :return: a numpy array of the ECG shaped (instances, samples, num_leads)
    """
    # if single ecg
    if ecg.ndim < 3:
      ecg = np.expand_dims(ecg, 0)
    
    # number of samples for new sampling rate
    out_samples = int((ecg.shape[1]/ecg_sampling) * sampling_rate)
    #initialize processed array
    out = np.zeros((ecg.shape[0], num_samples, len(lead_out)))
    for i in np.arange(out.shape[0]):
      for j, lead in enumerate(lead_out):
        #print(lead)
        # select lead from input
        if lead in lead_in:
           new_lead = ecg[i, :, lead_in.index(lead)]
        elif lead == 'III':
            new_lead = ecg[i, :, lead_in.index('II')] - ecg[i, :, lead_in.index('I')]
        elif lead == 'aVR':
            new_lead = - ((ecg[i, :, lead_in.index('I')] + ecg[i, :, lead_in.index('II')]) / 2)
        elif lead == 'aVL':
            new_lead = ecg[i, :, lead_in.index('I')] - (ecg[i, :, lead_in.index('II')] / 2)
        elif lead == 'aVF':
            new_lead = ecg[i, :, lead_in.index('II')] - (ecg[i, :, lead_in.index('I')] / 2)
            
        # interpolate if sampling rates are not equal or to interpolate to larger size  
        if ecg_sampling != sampling_rate or (out_samples < num_samples and pad==0):
           num_interp = num_samples if out_samples < num_samples else out_samples
           new_lead = np.interp(
               np.linspace(0, 1, num_interp),
               np.linspace(0, 1, len(new_lead)),
               new_lead,)
           #print(f'interpolated to {new_lead.shape}')
        if num_samples > len(new_lead):
           diff = int((num_samples - len(new_lead))/2)
           new_lead = np.pad(new_lead, ((0, 0), (diff, num_samples-len(new_lead)-diff), (0, 0)), mode='constant')
           #print(f'padded to {new_lead.shape}')
        elif num_samples < len(new_lead):
           diff = int((len(new_lead)-num_samples)/2)
           #print(f'{diff} {new_lead.shape}')
           new_lead = new_lead[diff:diff+num_samples]
           #print(f'truncated to {new_lead.shape}')
        if filt>0:
           new_lead = bandpass_filtering(new_lead, sampling_rate, lowcut=0.5, highcut=100)
        if filt==2:
           new_lead = notch_filtering(new_lead, sampling_rate, powerline_freq=60)           
        out[i, :, j] = new_lead
    return out


def get_data_info(data_type='rhythm'):
    data_info = {}
    data_info['path'] = {}
    data_info['path']['code'] = 'code_braveheart_rhythm_8l_400_bf_nf'
    data_info['path']['bidmc'] = 'bidmc_8l_400_bf_nf'
    data_info['path']['shanghai'] = 'shanghai_8l_400_bf_nf'
    data_info['path']['ukb'] = 'ukb_braveheart_rhythm_8l_400_bf_nf'
    data_info['path']['ptb'] = 'ptb_xl_rhythm_8l_400_bf'
    data_info['path']['vanderbilt'] = 'vanderbilt_braveheart_rhythm_8l_400_bf_nf'
    data_info['path']['mimic'] = 'mimic_braveheart_rhythm_8l_400_bf'
    
    data_info['id'] = {}
    data_info['id']['code'] = 'id_patient'
    data_info['id']['bidmc'] = 'id'
    data_info['id']['shanghai'] = 'ID'
    data_info['id']['ukb'] = 'id'
    data_info['id']['ptb'] = 'patient_id'
    data_info['id']['vanderbilt'] = 'id'
    data_info['id']['mimic'] = 'subject_id'

    data_info['pad'] = {}
    data_info['pad']['code'] = 600
    data_info['pad']['bidmc'] = 100
    data_info['pad']['shanghai'] = 100
    data_info['pad']['ukb'] = 10
    data_info['pad']['ptb'] = 10
    data_info['pad']['vanderbilt'] = 100
    data_info['pad']['mimic'] = 100
    
    data_info['sampling'] = {}
    data_info['sampling']['code'] = 400
    data_info['sampling']['bidmc'] = 400
    data_info['sampling']['shanghai'] = 400
    data_info['sampling']['ukb'] = 400
    data_info['sampling']['ptb'] = 500
    data_info['sampling']['vanderbilt'] = 400
    data_info['sampling']['mimic'] = 400

    data_info['date'] = {}
    data_info['date']['code'] = 'date_exam'
    data_info['date']['bidmc'] = 'ecg_date'
    data_info['date']['shanghai'] = 'ECGCheckTime'
    data_info['date']['ukb'] = 'ecg_date'
    data_info['date']['ptb'] = 'recording_date'
    data_info['date']['vanderbilt'] = ''
    data_info['date']['mimic'] = 'ecg_time'
    
    data_info['filename'] = {}
    data_info['filename']['code'] = 'id_exam'
    data_info['filename']['bidmc'] = 'filename'
    data_info['filename']['shanghai'] = 'filename'
    data_info['filename']['ukb'] = 'filename'
    data_info['filename']['ptb'] = 'filename_hr'
    data_info['filename']['vanderbilt'] = 'filename'
    data_info['filename']['mimic'] = 'filename'
    
    data_info['age'] = {}
    data_info['age']['code'] = 'age'
    data_info['age']['bidmc'] = 'age_ecg'
    data_info['age']['shanghai'] = 'age'
    data_info['age']['ukb'] = 'age_ecg'
    data_info['age']['ptb'] = 'age'

    data_info['sex'] = {}
    data_info['sex']['code'] = 'sex'
    data_info['sex']['bidmc'] = 'female'
    data_info['sex']['shanghai'] = 'sex'
    data_info['sex']['ukb'] = 'gender'
    data_info['sex']['ptb'] = 'sex'
    
    data_info['split'] = {}
    data_info['split']['code']={}
    data_info['split']['bidmc']={}
    data_info['split']['shanghai']={}
    data_info['split']['ukb'] ={}
    data_info['split']['ptb'] = {}
    for split in ['train', 'val', 'test']: 
      data_info['split']['code'][split] = f'code_{data_type}_{split}_ids.csv'
      data_info['split']['bidmc'][split] = f'arun_bidmc_{data_type}_{split}_ids.csv'
      data_info['split']['shanghai'][split] = f'shanghai_{data_type}_{split}_ids.csv'
      data_info['split']['ukb'][split] = f'ukb_{data_type}_{split}_ids.csv'
      data_info['split']['ptb'][split] = f'ptb_{split}_split.csv'
    return data_info

# Visualization methods for ECG signals and ML Model Metrics

# COLORMAP TEMPLATES FOR HORIZONTAL AND VERTICAL LEADS
LEAD_COLORS = {
    'I': plt.get_cmap('viridis', 12)(2),  # 0 degrees
    'II': plt.get_cmap('viridis', 12)(4),  # 60 degrees
    'III': plt.get_cmap('viridis', 12)(6),  # 120 degrees
    'aVR': plt.get_cmap('viridis', 12)(9),  # 150 degrees
    'aVL': plt.get_cmap('viridis', 12)(1),  # -30 degrees
    'aVF': plt.get_cmap('viridis', 12)(5),  # 90 degrees
    'V1': plt.get_cmap('plasma', 12)(2),
    'V2': plt.get_cmap('plasma', 12)(3),
    'V3': plt.get_cmap('plasma', 12)(4),
    'V4': plt.get_cmap('plasma', 12)(5),
    'V5': plt.get_cmap('plasma', 12)(6),
    'V6': plt.get_cmap('plasma', 12)(7)
}
LEAD_COLORS = defaultdict(lambda: 'black', LEAD_COLORS)


# -----------------------------------------------ECG VISUALIZATION----------------------------------------------------#

def plot_ecg(ecg_signal, sampling_rate, lead_names=None, subplots=True, subplot_shape=None, ylim=None, share_ylim=True,
             title=None, std=None, percentiles=None, figsize=None, show_axes=True, **kwargs):
    """
    Plots ECG signal(s) in the time domain.

    Arguments:
        ecg_signal (ndarray): ECG signal(s) of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal(s).
        lead_names (list): List of lead names. If None, the leads will be named as Lead 1, Lead 2, etc.
        subplots (bool): If True, the ECG leads will be plotted in separate subplots.
        subplot_shape (tuple): Shape of the subplot grid. If None, the shape will be automatically determined.
        ylim (tuple): Y-axis limits of the plot.
        share_ylim (bool): If True, the y-axis limits of the subplots will be shared.
        title (str): Title of the plot.
        std (ndarray): Standard deviation of the ECG signal(s) of shape (num_samples, num_leads).
        percentiles (tuple): Percentiles of the ECG signal(s) of shape (2, num_samples, num_leads).
        figsize (tuple): Figure size.
        show_axes (bool): If True, the axes of the plot will be plotted.
        **kwargs: Additional arguments to be passed to the matplotlib.pyplot.plot function.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """
    # Check ECG signal shape
    if len(ecg_signal.shape) != 2:
        raise ValueError('ECG signal must have shape: (num_samples, num_leads)')

    # Get number of ECG leads and time_index vector
    time_index = np.arange(ecg_signal.shape[0]) / sampling_rate
    num_leads = ecg_signal.shape[1]

    # If share_ylim, find ECG max and min values
    ylim_ = None
    if ylim is not None:
        ylim_ = ylim
    if ylim is None and share_ylim is True:
        ylim_ = (np.min(ecg_signal), np.max(ecg_signal))

    # Check for Lead Names
    if lead_names is not None:
        # Check number of leads
        if len(lead_names) != num_leads:
            raise ValueError('Number of lead names must match the number of leads in the ECG data.')
        lead_colors = LEAD_COLORS
    else:
        lead_names = [f'Lead {i + 1}' for i in range(num_leads)]  # Lead x
        cmap = plt.get_cmap('cividis', num_leads)
        lead_colors = dict(zip(lead_names, cmap(np.linspace(0, 0.7, num_leads))))

    # Check subplot shape
    if subplots is True and num_leads > 1:
        if subplot_shape is not None:
            if subplot_shape[0] * subplot_shape[1] < num_leads:
                raise ValueError('Subplot shape is too small to fit all the leads.')
        else:
            subplot_shape = (num_leads, 1)

    if figsize is None:
        if subplots is True and num_leads > 1:
            figsize = (15, 6)
        else:
            figsize = (13, 4)

    # Plotting
    if subplots is True and num_leads > 1:
        fig, axes = plt.subplots(nrows=subplot_shape[0], ncols=subplot_shape[1], sharex='row', sharey='row',
                                 figsize=figsize)
        flat_axes = axes.T.flatten()

        for i in range(num_leads):
            flat_axes[i].plot(time_index, ecg_signal[:, i], c=lead_colors[lead_names[i]], **kwargs)
            flat_axes[i].set_ylim(ylim_)
            flat_axes[i].legend([lead_names[i]], loc='upper right')

            if std is not None:
                flat_axes[i].fill_between(time_index, ecg_signal[:, i] - std[:, i], ecg_signal[:, i] + std[:, i],
                                          alpha=0.2, color=lead_colors[lead_names[i]], label='_nolegend_')
            if percentiles is not None:
                flat_axes[i].fill_between(time_index, percentiles[0][:, i], percentiles[1][:, i], alpha=0.2,
                                          color=lead_colors[lead_names[i]], label='_nolegend_')
            if show_axes is False:
                _remove_ticks(flat_axes[i])

    else:
        fig = plt.figure(figsize=figsize)
        for i in range(num_leads):
            plt.plot(time_index, ecg_signal[:, i], c=lead_colors[lead_names[i]], **kwargs)

            if std is not None:
                plt.fill_between(time_index, ecg_signal[:, i] - std[:, i], ecg_signal[:, i] + std[:, i], alpha=0.2,
                                 color=lead_colors[lead_names[i]], label='_nolegend_')
            if percentiles is not None:
                plt.fill_between(time_index, percentiles[0][:, i], percentiles[1][:, i], alpha=0.2,
                                 color=lead_colors[lead_names[i]], label='_nolegend_')
        plt.ylim(ylim_)
        plt.legend(lead_names, loc='upper right')
        if show_axes is False:
            _remove_ticks(plt.gca())

    # Plot Labels
    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude (mV)")
    fig.suptitle(title)
    fig.tight_layout()

    return fig


def _remove_ticks(ax):
    # Remove ticks and tick labels, but keep the y-label
    ax.set_yticks([])  # Remove y-ticks
    ax.set_xticks([])  # Remove x-ticks
    ax.set_xticklabels([])  # Remove x-tick labels
    ax.set_yticklabels([])  # Remove y-tick labels
    ax.spines['top'].set_visible(False)  # Remove top frame line
    ax.spines['right'].set_visible(False)  # Remove right frame line
    ax.spines['bottom'].set_visible(False)  # Remove bottom frame line
    ax.spines['left'].set_visible(False)  # Remove left frame line


def plot_ecg_paper(ecg_signal, sampling_rate, lead_names=None, title='', columns=4, row_height=6, style=None, **kwargs):
    """
    Plots ECG signal(s) in the time domain, using the ecg_plot library.

    Arguments:
        ecg_signal (ndarray): ECG signal(s) of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal(s).
        lead_names (list): List of lead names. If None, the standard 12-lead ECG leads will be used.
        title (str): Title of the plot.
        columns (int): Number of columns of the plot.
        row_height (int): Height of each row of the plot.
        style (str): Style of the plot. If None, the default style will be used.
        **kwargs: Additional arguments to be passed to the ecg_plot.plot function.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """
    # Check for Lead Names
    if lead_names is None:
        lead_names = ECG_12_LEADS

    ecg_plot.plot(ecg_signal.T, sample_rate=sampling_rate, lead_index=lead_names, columns=columns,
                  row_height=row_height, title=title, style=style, **kwargs)
    fig = ecg_plot.ecg_plot.plt.figure(1)

    return fig


def plot_ecg_comp(ecg_signal_1, ecg_signal_2, sampling_rate, lead_names=None, share_ylim=True, window=None,
                  title=None):
    """
   Plots ECG signal(s) in the time domain.

   Arguments:
       ecg_signal_1 (ndarray): ECG signal(s) of shape (num_samples, num_leads).
       ecg_signal_2 (ndarray): ECG signal(s) of shape (num_samples, num_leads).
       sampling_rate (int): Sampling rate of the ECG signal(s).
       lead_names (list): List of lead names. If None, the leads will be named as Lead 1, Lead 2, etc.
       subplots (bool): If True, the ECG leads will be plotted in separate subplots.
       share_ylim (bool): If True, the y-axis limits of the subplots will be shared.
       window (tuple): Tuple of start and end values for windowing (sec)
       title (str): Title of the plot.

   Returns:
       fig (matplotlib.figure.Figure): Figure object.
   """
    # Check ECG signal shape
    if len(ecg_signal_1.shape) != 2 or len(ecg_signal_2.shape) != 2:
        raise ValueError('ECG signals must have shape: (num_samples, num_leads)')

    # Check for window_size
    if window is not None:
        start = int(window[0] * sampling_rate)
        end = int(window[1] * sampling_rate)
        # Resize ECG signals
        ecg_signal_1 = ecg_signal_1[start:end]
        ecg_signal_2 = ecg_signal_2[start:end]

    # Get number of ECG leads and time_index vector
    time_index = np.arange(ecg_signal_1.shape[0]) / sampling_rate
    num_leads = ecg_signal_1.shape[1]

    # If share_ylim, find ECG max and min values
    ylim = None
    if share_ylim is True:
        ylim = (np.min([ecg_signal_1, ecg_signal_2]), np.max([ecg_signal_1, ecg_signal_2]))

    # Check for Lead Names
    if lead_names is not None:
        # Check number of leads
        if len(lead_names) != num_leads:
            raise ValueError('Number of lead names must match the number of leads in the ECG data.')
        # Check lead names
        if not set(lead_names).issubset(ECG_12_LEADS):
            raise ValueError('Unknown Lead names')
    else:
        lead_names = [f'Lead {i + 1}' for i in range(num_leads)]  # Lead x

    # Create figure
    fig, axes = plt.subplots(nrows=1, ncols=num_leads, sharex='row', sharey='row', figsize=(20, 6))
    if num_leads == 1:
        axes = [axes]

    for i in range(num_leads):
        axes[i].plot(time_index, ecg_signal_1[:, i], c='black', alpha=0.75, linewidth=1)
        axes[i].plot(time_index, ecg_signal_2[:, i], c='red', alpha=0.75, linewidth=1)
        axes[i].set_ylim(ylim)
        axes[i].legend([lead_names[i]], loc='upper right')

    # Plot Labels
    fig.supxlabel("Time (seconds)")
    fig.supylabel("Amplitude (mV)")
    fig.suptitle(title)
    fig.tight_layout()

    return fig


# -----------------------------------------MODEL RESULTS VISUALIZATION------------------------------------------------#

def plot_histogram(data, label=None, title=None, bins=100, clip_p=None, metric_precision=2):
    """
    Method to plot histogram of data.

    Arguments:
        data (np.array): Data to plot.
        label (str): Label for data.
        title (str): Title for plot.
        bins (int): Number of bins for histogram.
        clip_p (int or None): Percentile for left/right clipping of the data.

    Returns:
        fig (matplotlib.pyplot.figure): Figure object.
    """

    clean_data = data.copy()
    # Remove outliers for visualization based on percentiles
    if clip_p is not None:
        low_bound = np.percentile(data, clip_p)
        high_bound = np.percentile(data, 100 - clip_p)
        clean_data = np.array([value for value in data if low_bound <= value <= high_bound])

    # Compute statistics
    median = np.median(data)
    mean = np.mean(clean_data)
    iqr = np.percentile(data, 75) - np.percentile(data, 25)

    # Create figure
    fig = plt.figure(figsize=(13, 5))
    # Plot histogram
    plt.hist(clean_data, bins=bins, alpha=0.75)
    plt.axvline(median, color='green', linestyle='dashed', linewidth=1.5)
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=1.5)
    # Add labels
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Count')
    plt.legend([f'Median: {median:.{metric_precision}f} (IQR: {iqr:.{metric_precision}f})',
                f'Mean: {mean:.{metric_precision}f}'])

    return fig


def plot_history(history, loss='Loss', metric='Metric', title=None):
    """
    Plots the training and validation history (loss and metrics values per training epoch) of an ML model. In case
    multiple histories are given as a list, computes the mean and std of the corresponding values.

    Arguments:
        history (dict/list): dictionary/dictionaries containing training and validation metrics
        (e.g., 'loss', 'val_loss', etc.)
        loss (str): Loss function used during training.
        metric (str): Metric used to assess model performance.
        title (str): Title of the plot.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """

    # Check for multiple history instances
    multiple_instances = False
    if isinstance(history, list):
        multiple_instances = True
        # Check if all instances are dictionaries
        if not all(isinstance(item, dict) for item in history):
            raise TypeError('History objects must be dictionaries')
        # Check if all instances have the same keys
        if not all(history[0].keys() == item.keys() for item in history):
            raise ValueError('History objects must have the same keys')
    else:  # Check if history is a dictionary
        if not isinstance(history, dict):
            raise TypeError('History object must be a dictionary.')

    # If Multiple Histories, Compute Average and Std of Loss and Metrics
    history_avg = {}
    history_std = {}
    if multiple_instances:
        loss_keys = [key for key in history[0].keys() if 'loss' in key]
        metrics_keys = [key for key in history[0].keys() if 'loss' not in key and 'lr' not in key]

        for loss_key in loss_keys:
            history_avg[loss_key] = np.mean([instance[loss_key] for instance in history], axis=0)
            history_std[loss_key] = np.std([instance[loss_key] for instance in history], axis=0)

        for metric_key in metrics_keys:
            history_avg[metric_key] = np.mean([instance[metric_key] for instance in history], axis=0)
            history_std[metric_key] = np.std([instance[metric_key] for instance in history], axis=0)

        history = history_avg

    # Plotting
    colors = defaultdict(lambda: 'black')
    colors['loss'] = 'violet'
    colors['val_loss'] = 'cyan'
    colors['accuracy'] = 'purple'
    colors['val_accuracy'] = 'dodgerblue'
    colors['auc'] = 'purple'
    colors['val_auc'] = 'dodgerblue'
    colors['mae'] = 'purple'
    colors['val_mae'] = 'dodgerblue'

    # Subplot
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    # Epochs, Loss Keys, Metrics Keys
    epochs = np.arange(1, len(history['loss']) + 1)
    loss_keys = [key for key in history.keys() if 'loss' in key]
    metrics_keys = [key for key in history.keys() if 'loss' not in key and 'lr' not in key]

    # Plot Loss Keys
    for loss_key in loss_keys:
        axes[0].plot(epochs, history[loss_key], color=colors[loss_key], linewidth=2)
        if multiple_instances:
            axes[0].fill_between(epochs, history[loss_key] - history_std[loss_key],
                                 history[loss_key] + history_std[loss_key], alpha=0.2, color=colors[loss_key],
                                 label='_nolegend_')

    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel(loss)
    axes[0].legend(loss_keys, loc='upper right')
    axes[0].set_title('Loss History')

    # Plot Metrics Keys
    for metrics_key in metrics_keys:
        axes[1].plot(epochs, history[metrics_key], color=colors[metrics_key], linewidth=2)
        if multiple_instances:
            axes[1].fill_between(epochs, history[metrics_key] - history_std[metrics_key],
                                 history[metrics_key] + history_std[metrics_key], alpha=0.2, color=colors[metrics_key],
                                 label='_nolegend_')
    axes[1].set_xlabel('Training Epoch')
    axes[1].set_ylabel(metric)
    axes[1].legend(metrics_keys, loc='lower right')
    axes[1].set_title('Metrics History')

    fig.suptitle(title)
    fig.tight_layout()
    # plt.show()

    return fig


def plot_class_metrics(class_metrics, class_names=None, metrics=None, avg_key='macro avg',
                       title='Classification Metrics'):
    """
    Plots the classification metrics computed by utils.compute_class_metrics(). If multiple class_metrics are given
     as input, computes the mean and std of the corresponding values.

    Arguments:
        class_metrics (dict/list): Dictionary/Dictionaries containing classification metrics.
        class_names (list): List of class names.
        metrics (list): List of metrics to plot.
        avg_key (str): Type of averaging performed on the data 'macro avg' or 'weighted avg'
        title (str): Title of the plot.

    Returns:
        fig (matplotlib.figure.Figure): Figure object.
    """

    # Check for class_names
    if class_names is None:
        class_names = [str(i) for i in range(len(class_metrics))]
    # Check for metrics
    if metrics is None:
        metrics = ['precision', 'recall', 'f1-score']  # Default metrics
    # Check average type
    if avg_key not in ['macro avg', 'weighted avg']:
        raise ValueError('Average key must be "macro avg" or "weighted avg".')

    # Check for multiple class_metrics instances
    multiple_instances = False
    if isinstance(class_metrics, list):
        multiple_instances = True
        # Check if all instances are dictionaries
        if not all(isinstance(item, dict) for item in class_metrics):
            raise TypeError('Class Metrics objects must be dictionaries')
        # Check if all instances have the same keys
        if not all(class_metrics[0].keys() == item.keys() for item in class_metrics):
            raise ValueError('Class Metrics objects must have the same keys')
        # Check if all instances have class_names in class_metrics
        if not all(set(class_names).issubset(item.keys()) for item in class_metrics):
            raise ValueError('Class Metrics objects must contain all class names')
        # Check if all instances have metrics in class_metrics
        if not all(set(metrics).issubset(item[class_names[0]].keys()) for item in class_metrics):
            raise ValueError('Class Metrics objects must contain all given metrics')
        # Check if all instances have average key in class_metrics
        if not all(avg_key in item.keys() for item in class_metrics):
            raise ValueError('Class Metrics objects must contain average values')
    # Single Instance Case
    else:  # Check if class_metrics is a dictionary
        if not isinstance(class_metrics, dict):
            raise TypeError('Class Metrics object must be a dictionary.')
        # Check if class_names in class_metrics
        if not set(class_names).issubset(class_metrics.keys()):
            raise ValueError('Class Metrics must contain all class names.')
        # Check if metrics in class_metrics
        if not set(metrics).issubset(class_metrics[class_names[0]].keys()):
            raise ValueError('Class metrics must contain all given metrics.')
        # Check if average key in class_metrics
        if avg_key not in class_metrics.keys():
            raise ValueError('Class Metrics must contain average values.')

    # If Multiple class_metrics, Compute Average and Std
    class_metrics_avg = {}
    class_metrics_std = {}
    if multiple_instances:
        class_metrics_keys = class_metrics[0].keys()

        for class_metrics_key in class_metrics_keys:  # For each class key
            if isinstance(class_metrics[0][class_metrics_key], dict):  # If dict, compute avg and std for each metric
                class_metrics_avg[class_metrics_key] = dict(
                    zip([key for key in class_metrics[0][class_metrics_key].keys()],
                        [np.mean([instance[class_metrics_key][key] for instance in class_metrics], axis=0) for key in
                         class_metrics[0][class_metrics_key].keys()]))
                class_metrics_std[class_metrics_key] = dict(
                    zip([key for key in class_metrics[0][class_metrics_key].keys()],
                        [np.std([instance[class_metrics_key][key] for instance in class_metrics], axis=0) for key in
                         class_metrics[0][class_metrics_key].keys()]))
            else:  # Float case
                class_metrics_avg[class_metrics_key] = np.mean(
                    [instance[class_metrics_key] for instance in class_metrics], axis=0)
                class_metrics_std[class_metrics_key] = np.std(
                    [instance[class_metrics_key] for instance in class_metrics], axis=0)

        class_metrics = class_metrics_avg

    # Plotting
    colors = defaultdict(lambda: 'black')
    colors['accuracy'] = 'dodgerblue'
    colors['precision'] = 'mediumaquamarine'
    colors['recall'] = 'lightseagreen'
    colors['f1-score'] = 'turquoise'

    fig = plt.figure(figsize=(10, 7))
    bar_width = 0.25

    # Plot Class Metrics
    for i in range(len(class_names)):  # For each class
        for j in range(len(metrics)):  # For each metric
            plt.bar(i + j * bar_width, class_metrics[class_names[i]][metrics[j]], width=bar_width,
                    color=colors[metrics[j]])
            plt.text(i + j * bar_width, class_metrics[class_names[i]][metrics[j]],
                     '{:.2f}'.format(class_metrics[class_names[i]][metrics[j]]), ha='center', va='top')
            if multiple_instances:
                plt.errorbar(i + j * bar_width, class_metrics[class_names[i]][metrics[j]],
                             yerr=class_metrics_std[class_names[i]][metrics[j]], ecolor='black', alpha=0.5)

    # Plot Average Metrics
    for j in range(len(metrics)):  # For each metric
        plt.bar(len(class_names) + j * bar_width, class_metrics[avg_key][metrics[j]], width=bar_width,
                color=colors[metrics[j]], label=metrics[j])
        plt.text(len(class_names) + j * bar_width, class_metrics[avg_key][metrics[j]],
                 '{:.2f}'.format(class_metrics[avg_key][metrics[j]]), ha='center', va='top')
        if multiple_instances:
            plt.errorbar(len(class_names) + j * bar_width, class_metrics[avg_key][metrics[j]],
                         yerr=class_metrics_std[avg_key][metrics[j]], ecolor='black', alpha=0.5)

    # Plot Accuracy
    plt.bar(len(class_names) + 1 + bar_width, class_metrics['accuracy'], width=bar_width, color=colors['accuracy'],
            label='accuracy')
    plt.text(len(class_names) + 1 + bar_width, class_metrics['accuracy'],
             '{:.2f}'.format(class_metrics['accuracy']), ha='center', va='bottom')
    if multiple_instances:
        plt.errorbar(len(class_names) + 1 + bar_width, class_metrics['accuracy'], yerr=class_metrics_std['accuracy'],
                     ecolor='black', alpha=0.5)

    plt.xlabel('Class Group')
    plt.xticks(np.arange(len(class_names) + 2) + (len(metrics) - 1) / 2.0 * bar_width,
               class_names + [avg_key, 'accuracy'])
    plt.ylabel('Metric Value')
    plt.legend(loc='lower right')
    plt.title(title)

    return fig


def plot_confusion_matrix(conf_matrix, class_names=None, normalize=False, title='Confusion Matrix'):
    """
    Plots a confusion matrix, given the matrix and the class names. If multiple confusion matrices are given
     as input, it computes the average confusion matrix.

    Arguments:
        conf_matrix (ndarray/list): The confusion matrix array (or a list of arrays)
        class_names (list): List of class names
        normalize (bool): If True, the confusion matrix is normalized
        title (str): Title of the plot

    Returns:
        figure (matplotlib.figure.Figure): Figure object.
    """

    # Check for class_names
    if class_names is None:
        class_names = [str(i) for i in range(len(conf_matrix))]

    # Check for multiple instances
    if isinstance(conf_matrix, list):
        conf_matrix = np.mean(conf_matrix, axis=0, dtype='int64')

    # Normalization
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting
    colormap = plt.get_cmap('Blues')
    fig = plt.figure(figsize=(8, 8))
    plt.matshow(conf_matrix, cmap=colormap, fignum=fig.number)

    # Add Text Values
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix)):
            plt.text(j, i, format(conf_matrix[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment="center", verticalalignment="center",
                     color="white" if conf_matrix[i, j] > np.max(conf_matrix) / 2. else "black")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)

    return fig


def plot_roc_curve(roc_curve_auc, title='ROC Curve'):
    """
    Plots a ROC curve, given the 'false positive rate', 'true positive rate' and the AUC. If multiple roc curves
     are given as input, it computes the average the roc curves.

    Arguments:
        roc_curve_auc (tuple/list): ROC curve (fpr, tpr, roc_auc) or list of tuples
        title (str): Title of the plot

    Returns:
        figure (matplotlib.figure.Figure): Figure object.
    """

    # False Positive Rate, True Positive Rate, Thresholds

    # Check for multiple instances
    if isinstance(roc_curve_auc, list):
        # Compute Average fpr, tpr, roc_auc
        fpr = np.mean([instance[0] for instance in roc_curve_auc], axis=0)
        tpr = np.mean([instance[1] for instance in roc_curve_auc], axis=0)
        roc_auc = np.mean([instance[2] for instance in roc_curve_auc], axis=0)
    else:
        fpr, tpr, roc_auc = roc_curve_auc

    # Plotting
    color = 'steelblue'
    fig = plt.figure(figsize=(8, 8))

    plt.plot(fpr, tpr, color=color, linewidth=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', linewidth=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    # plt.show()

    return fig
                         
def get_regression_metrics(y_test, y_pred, metrics):
    result = {}
    for m in metrics:
        if m in ['MAE', 'mae']:
            res = mean_absolute_error(y_test, y_pred)
        elif m in ['mape',  'MAPE'] :
            res = mean_absolute_percentage_error(y_test, y_pred)            
        elif m in ['r2', 'r2_score']:
            res = r2_score(y_test, y_pred)            
        result[m] = res
    return result
    
def get_class_metrics(y_test, y_pred, metrics):
    result = {}
    for m in metrics:
        if m in ['AUC',  'auc']:
            res = roc_auc_score(y_test, y_pred)
        if m in ['accuracy',  'acc'] :
            res = accuracy_score(y_test, np.where(y_pred>0.5, 1, 0))
        if m in ['f1',  'f1_score']:
            res = f1_score(y_test, np.where(y_pred>0.5, 1, 0))
        result[m] = res
    return result
                         
def get_multiclass_metrics(y_test, y_pred, metrics, num_classes):
    result = {}
    for m in metrics:
        if m == 'micro_auc':
            res = roc_auc_score(y_test, y_pred, average='micro')
        elif m == 'macro_auc':
            res = roc_auc_score(y_test, y_pred, average='macro')
        elif m in ['auc', 'AUC'] :
            res = roc_auc_score(y_test, y_pred)
        elif m in ['macro_f1score', 'f1_score']:           
            res = f1_score(y_test, np.where(y_pred > 0.5, 1, 0), average='macro')
        elif 'acc' in m:
            if num_classes>1:
                res = 0
                for i in range(num_classes):
                    y_score = np.where(y_pred[:, i] > 0.5, 1, 0)
                    acc_score = balanced_accuracy_score if m == 'wt_acc' else accuracy_score
                    res = res + acc_score(y_test[:, i], y_score)
                    res = res/num_classes
            else:
                acc_score = balanced_accuracy_score if m == 'wt_acc' else accuracy_score
                res = acc_score(y_test, np.where(y_pred > 0.5, 1, 0))
        result[m] = res
    return result