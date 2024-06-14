#!/usr/bin/env python
# coding: utf-8
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import pickle
import lzma
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from utils.helpers_torch import SurrGradSpike,read_word_files,build_connectivity_matrix_tde,filter_basedon_matlabcorr, Tau_init, generate_keywords_dataset_singleKW
from sklearn.metrics import confusion_matrix

### Improvements respect TDE code:
##  - separation of the timeset in the simulation and the bin_size of the input --> it should always me multiple
##  - Added code for inference and recording of number of spikes in each layer
##  - Added dropout. Pdrop: Probability that in the forward pass the input is 0 in the network. Helps with overfitting
##  - Tau_alpha only positive. Added torch.abs(tau_a) so we cannot train time constants towards negative values.
##  - Posibility of weights in the output layer only positive. Enforcing a local code where the neurons only add
##  - Parameters parse as a dictionary for easy trace in the folder of the simulation

Category = 'seven'
params = {
# Simulation variables ##################################################################
# Best params: wdec 1e-4 L1L2null lr 0.0015 bin_size=15 epochs 3000
'Dataout_dirname': 'job_scripts/SNN_gro/keyspot_torch/tde_unbi_ncell_5p_1_Cat'+Category,
#'Dataout_dirname' : "/Users/apequeno/Project/GRON/software/keyword_spotting/outputs/comp_corrlevels/tde_unbi_ncell_1p_1_Cat"+Category,
'Epochs' : 3000,           # 300 set the number of epochs you want to train the network here 500
'Batch_size' : 500,        # 500 timestep 15 -- 400 for timestep5 -- 200 timestep3 -- 150 timestep1 but! now del dummy
'NWORKERS' : 0,
'Bin_size' : 15,    # All comparatives at 15
'Step_size' : 15,   # Time in cell dynamics Every cell dynamics are calculated based on this number ( Therefore the output )
'Learning_rate' : 0.0015,  # Original 0.0015  x3 0.0045
'L1_reg' : 0,       # Weight decay already applies Ori 0.0015
'L2_reg' : 0,       # Weight decay already applies Ori 0.000001
'WeightDecay' : 0.0001,
'Use_dropout' : True,
'PDrop' : 0.1,
'Enforce_Wpos' : False,
# Load previous training
'Datain_dirname' : "job_scripts/SNN_gro/keyspot_torch/gpu_tde_3k_bin1",
#'Datain_dirname' : "/Users/apequeno/Project/GRON/software/keyword_spotting/outputs/comp_corrlevels/tde_corr_unbi_011_1",
'Fileprefix_saved' : "snn_tau_train_best",
'Load_saved_model' : False,
# Filter the channels whose correlation in the spike frequencies are smaller than threshold 0c5, 1 or 2 check Input_correlation_analysis.m
'FilterCorr' : True,
'FilterCorr_file' : "job_scripts/SNN_gro/keyspot_torch/data/spikes_tidigits_noise0/filter_channels_cells_5p_"+Category, #data/spikes_tidigits_noise0/filter_channels_cells_10p_  job_scripts/SNN_gro/keyspot_torch/data/spikes_tidigits_noise0/filter_channels_corr2
#'FilterCorr_file' :'data/spikes_tidigits_noise0/filter_channels_cells_1p_'+Category,
'FilterTime': True,    # FIXED NO EFFECT always to 1500ms filter
'TauInit_corr': False,   # If true there will be the taus in the analysis for all categories and not specific to the category in question...
'Train_percent': None   # Train percent None if it does not apply otherwise 0.25, 0.50, 0.75, 0.90
}
#####################################################################

# Device configuration for CUDA
if torch.cuda.device_count() > 1:
    gpu_sel = 1
    gpu_av = [torch.cuda.is_available() for ii in range(torch.cuda.device_count())]
    print("Detected {} GPUs. The load will be shared.".format(torch.cuda.device_count()))
    if True in gpu_av:
        if gpu_av[gpu_sel]:
            device = torch.device("cuda:" + str(gpu_sel))
        else:
            device = torch.device("cuda:" + str(gpu_av.index(True)))
        # torch.cuda.set_per_process_memory_fraction(0.25, device=device)
    else:
        device = torch.device("cpu")
else:
    if torch.cuda.is_available():
        print("Single GPU detected. Setting up the simulation there.")
        device = torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(0.3,
                                                   device=device)  # decrese or drop memory fraction if more is available (the smaller the better)
    else:
        device = torch.device("cpu")
        print("No GPU detected. Running on CPU.")
print(device) #print(torch.cuda.memory_summary())
# Seed use
use_seed = False  # set seed to achieve reprodicable results
if use_seed:
    seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Seed set to {}".format(seed))
else:
    print("Shuffle data randomly")

global letters
letters = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'oh']
dtype = torch.float

def load_layers_alpha(file_prefix, map_location, requires_grad=True):
    """ suffix = best | last """
    global nb_hidden
    # Layers and taus file
    lays = torch.load("{}_layers.pt".format(file_prefix), map_location=map_location)
    taus = torch.load("{}_tau_alpha.pt".format(file_prefix), map_location=map_location)
    lays[0].requires_grad = False # TDE weight layer doesnt required grad
    lays[1].requires_grad = requires_grad
    taus.requires_grad = requires_grad
    nb_hidden = lays[1].shape[0]
    return (lays, taus)

def define_globals(time_size=1, load_model=False):
    ''' This will be included in build and train '''
    global nb_inputs
    nb_inputs = 32
    global nb_hidden
    global reg_spikes
    reg_spikes = params['L1_reg']
    global reg_neurons
    reg_neurons = params['L2_reg']
    global nb_outputs
    nb_outputs = 2

    # Load previous pretrained model or generate a random distribution of output weights w2 and tau_alpha of facilitatory
    if load_model:
        layers, tau_alpha = load_layers_alpha("{}/{}".format(params['Datain_dirname'], params['Fileprefix_saved']), device)
        w1 = layers[0]
        w2 = layers[1]
    else:
        # Network layers
        layers = []
        # TDEs parameters
        w_fac = 1  # 50000
        tau_fac = 1  # 0.008
        w_trig = 1  # 50000
        tau_trig = 1  # 0.002
        max_dist = 32
        # Function coming from nengo TDE conversion--> better postprocess the signal
        w_fac_mat, w_trig_mat, tde_on = build_connectivity_matrix_tde(nb_inputs, w_fac=w_fac, w_trig=w_trig,
                                                                      tau_fac=tau_fac, tau_trig=tau_trig,
                                                                      max_dist=max_dist)
        nb_hidden = len(tde_on)
        w_fac_f = w_fac_mat[tde_on, :]
        w_trig_f = w_trig_mat[tde_on, :]
        tde_off = []
        if params['FilterCorr']:
            tde_off = filter_basedon_matlabcorr(params['FilterCorr_file'])
            w_fac_f = np.delete(w_fac_f, tde_off, 0)
            w_trig_f = np.delete(w_trig_f, tde_off, 0)
            nb_hidden = w_trig_f.shape[0]
        # Layer input to TDE --> It will be connected from the input in pair somehow
            #w1 = torch.empty((nb_inputs, nb_hidden), device=device, dtype=dtype, requires_grad=True)
            #torch.nn.init.constant_(w1, 1) # Initialize weights at 1 --> Only train constants
            #torch.nn.init.normal_(w1, mean=0.0, std=fwd_weight_scale / np.sqrt(nb_inputs))
        w1 = torch.stack([torch.tensor(w_fac_f, device=device, dtype=torch.float32), torch.tensor(w_trig_f, device=device, dtype=torch.float32)], dim=2)
        layers.append(w1)
        # Layer from TDE fully connected to Output
        w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
        torch.nn.init.normal_(w2, mean=0.0, std=1)
        layers.append(w2)
        # Network parameters
        # mid layer TDE
        if params['TauInit_corr']:
            tau_np = 0.001 * np.array(Tau_init)
            tau_np = np.delete(tau_np, tde_off, 0)
            tau_alpha = torch.tensor(tau_np, device=device, dtype=dtype, requires_grad=True)
        else:
            tau_alpha = torch.empty(nb_hidden, device=device, dtype=dtype, requires_grad=True)
            #torch.nn.init.normal_(tau_alpha, mean=30.0, std=10) # ms space 10-100 but lif we choose is 2
            torch.nn.init.uniform_(tau_alpha, 0.150, 0.250)
    # Copy of init tau for comparison
    i_name = ('{}/snn_tau_train_taus_init_random.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(i_name, 'wb') as handle:
        pickle.dump(np.array(tau_alpha.detach().cpu().numpy()), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # output layer LIF
    # From 1_spikesToTDE #tau_fac = 0.008#tau_trig = 0.002
    tau_syn = 0.008  # params['tau_syn']  # ms It can be 60             Bug: 8
    tau_mem = 0.002  # params['tau_mem']  # ms           it can be 80   Bug: 2
    global time_step
    time_step = time_size * 0.001
    global alpha_lif
    alpha_lif = float(np.exp(-time_step / tau_syn))
    global beta_lif
    beta_lif = float(np.exp(-time_step / tau_mem))

    # We want to train the alpha constant only of the TDE and the weights of the output layer
    opt_parameters = [tau_alpha, w2]

    with open('{}/sim_variables.txt'.format(params['Dataout_dirname']), 'w') as f:
        json.dump(params, f, indent=4, skipkeys=True)
    return layers, tau_alpha, opt_parameters

def run_snn_w_tde(input_nn, layers, tau_a, active_dropout=params['Use_dropout']):
    # Dropout is only for training Therefore active_dropout in inference should always be False CHECK
    bs, nb_input_steps, _ = input_nn.shape   # Batch size --> Number of samples
    step_ratio = int(params['Bin_size'] / params['Step_size'])
    nb_steps = nb_input_steps * (step_ratio)
    # h1_from_input = torch.einsum("abc,cd->abd", (inputs.tile((nb_input_copies,)), layers[0]))
    dropout_parser = nn.Dropout(p=params['PDrop'])
    h1_input = torch.einsum("abc,dce->abde", (input_nn, layers[0]))
    h1_input = dropout_parser(h1_input) if active_dropout else h1_input

    dummy_zeroes = torch.zeros((bs, nb_hidden), device=device, dtype=dtype, requires_grad=False)
    syn_f = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)
    current = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)
    mem = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)
    out = torch.zeros((bs, nb_hidden), device=device, dtype=dtype)

    min_cap = 0.0001 * torch.ones_like(tau_a)
    tau_a = torch.maximum(torch.abs(tau_a), min_cap)  # tau positive, 4 decimals and minimum 0.1 ms
    alpha = torch.exp(-time_step / tau_a)
    alpha = alpha.unsqueeze(0).repeat(bs, 1)
    # Here we define two lists which we use to record the membrane potentials and output spikes
    mem_rec = []
    spk_rec = []
    h1_f = h1_input[:,:,:,0] # facilitatory input
    h1_t = h1_input[:,:,:,1] # Thriggering input

    # Time modulation constants alpha_lif, beta_lif ... has been calculated per time step so they are a constant!
    # Compute TDE layer activity
    for t in range(nb_steps):
        h1 = h1_f[:, int(t/step_ratio)] if t % step_ratio == 0 else dummy_zeroes # h1_f++
        h2 = h1_t[:, int(t/step_ratio)] if t % step_ratio == 0 else dummy_zeroes # h1_f++

        # Leak and integrate with Trigger permission
        new_syn_f = alpha * syn_f + h1   # Gain
        # Epsc --> Current that has the time constant
        new_current = alpha_lif * current + new_syn_f * h2  # --> You can add a new parameter trained here that is the gain of the current
        # Integrate current in the membrane
        new_mem = beta_lif * mem + new_current # H2 is just 0 or 1 because of spike. 1: copy the trace of syn_t ; 0: nothing

        # Fire
        mthr = new_mem - 1.0
        out = spike_fn(mthr)
        rst = out.detach()  # We do not want to backprop through the reset

        mem = new_mem * (1.0 - rst)
        current = new_current
        syn_f = new_syn_f

        mem_rec.append(mem)
        spk_rec.append(out)
    # Now we merge the recorded membrane potentials into a single tensor
    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)
    del input_nn
    del dummy_zeroes

    # Compute Readout layer activity
    _, nb_steps, _ = spk_rec.shape
    layers[1] = torch.abs(layers[1]) if params['Enforce_Wpos'] else layers[1]
    h2 = torch.einsum("abc,cd->abd", (spk_rec, layers[1]))
    h2 = dropout_parser(h2) if active_dropout else h2
    flt = torch.zeros((bs, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((bs, nb_outputs), device=device, dtype=dtype)
    s_out_rec = []  # out is initialized as zeros, so it is fine to start with this
    out_rec = []
    for t in range(nb_steps):
        # Leaky integrate
        new_flt = alpha_lif * flt + h2[:, t]
        new_out = beta_lif * out + flt
        # Fire
        mthr_out = new_out - 1.0
        s_out = spike_fn(mthr_out)
        rst_out = s_out.detach()

        out = new_out * (1.0 - rst_out)
        flt = new_flt

        out_rec.append(out)
        s_out_rec.append(s_out)
    # Stack variables
    out_rec = torch.stack(out_rec, dim=1)
    s_out_rec = torch.stack(s_out_rec, dim=1)

    other_recs = [mem_rec, spk_rec, s_out_rec]
    layers_update = layers
    tau_update = tau_a

    return out_rec, other_recs, layers_update, tau_update

def compute_classification_accuracy(dataset, layers=None, tau=None, early=False, conmatrix=False, consufix='No_Specified'):
    """ Computes classification accuracy on supplied data in batches. """

    generator = DataLoader(dataset, batch_size=params['Batch_size'],
                           shuffle=False, num_workers=params['NWORKERS'])
    accs = []
    nspks_midlayer = np.array([])
    nspks_outlayer = np.array([])
    trues = []
    preds = []
    for x_local, y_local in generator:
        x_local, y_local = x_local.to(device), y_local.to(device)
        output, others, _, _ = run_snn_w_tde(x_local, layers, tau, active_dropout=False)
        del x_local
        # with output spikes
        spk_l1 = others[1].detach().cpu().numpy()
        spk_l2 = others[2].detach().cpu().numpy()
        nspks_midlayer = np.concatenate([nspks_midlayer, spk_l1], axis=0) if nspks_midlayer.size else spk_l1
        nspks_outlayer = np.concatenate([nspks_outlayer, spk_l2], axis=0) if nspks_outlayer.size else spk_l2

        m = (torch.sum(others[-1], 1))  # sum over time
        _, am = (torch.max(m, 1))  # argmax over output units
        # compare to labels
        tmp = np.mean((y_local == am).detach().cpu().numpy())
        accs.append(float(tmp))
        if conmatrix:
            trues.extend(y_local.detach().cpu().numpy())
            preds.extend(am.detach().cpu().numpy())
        del y_local

    if conmatrix:
        cm = confusion_matrix(trues, preds, normalize='true')
        cm_df = pd.DataFrame(cm, index=[ii for ii in letters], columns=[jj for jj in letters])
        cm_name = ('{}/confussion_PD_{}.pkl'.format(params['Dataout_dirname'], consufix))
        cm_df.to_pickle(cm_name)

        ''' No seaborn in FRANKLIN container
        plt.figure(figsize=(12, 9))
        sn.heatmap(cm_df,
                   annot=True,
                   fmt='.1g',
                   # linewidths=0.005,
                   # linecolor='black',
                   cbar=False,
                   square=False,
                   cmap="YlGnBu")
        plt.xlabel('\nPredicted')
        plt.ylabel('True\n')
        plt.xticks(rotation=0)
        plt.savefig('{}/confussion_{}.png'.format(Dataout_dirname, consufix), dpi=300)
        plt.close()
        '''

    return np.mean(accs), [nspks_midlayer, nspks_outlayer]

def check_accuracies(ds_train, ds_test, best_layers, best_taus):
    # Train spikes
    train_acc, _ = compute_classification_accuracy(ds_train, layers=best_layers, tau=best_taus, early=True,
                                                   conmatrix=False, consufix='TrainSplit')
    # Test spikes
    test_acc, _ = compute_classification_accuracy(ds_test, layers=best_layers, tau=best_taus, early=True,
                                                  conmatrix=False, consufix='TestSplit')
    print("Train accuracy: {}%".format(np.round(train_acc * 100, 2)))
    print("Test accuracy: {}%".format(np.round(test_acc * 100, 2)))
    print("Test accuracy as it comes, without rounding: {}".format(test_acc))

def plot_figures_epochs(loss_hist, acc_hist, dirname):
    # Figure Lost function over time
    plt.figure()
    plt.plot(range(1, len(loss_hist) + 1), loss_hist, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss ")
    plt.savefig('{}/Loss_per_epoch.png'.format(dirname), dpi=300)
    plt.close()

    # Figure Lost function over time
    plt.figure()
    plt.plot(range(1, len(acc_hist[0]) + 1), 100 * np.array(acc_hist[0]), color='blue')
    plt.plot(range(1, len(acc_hist[1]) + 1), 100 * np.array(acc_hist[1]), color='orange')
    plt.axhline(y=(100 * np.max(np.array(acc_hist[1]))), color='red') #, xmin=0, xmax=len(acc_hist[1]), color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(["Training", "Test"], loc='lower right')
    plt.savefig('{}/accuracy_per_epoch.png'.format(dirname), dpi=300)
    plt.close()
    return

def save_network_params(best_layers, best_taus, loss_hist, acc_hist, last_layers, last_taus):
    # Save layers and taus in pytorch format for posterior loading
    lay_name = ('{}/snn_tau_train_best_layers.pt'.format(params['Dataout_dirname']))
    torch.save(best_layers, lay_name)
    tau_name = ('{}/snn_tau_train_best_tau_alpha.pt'.format(params['Dataout_dirname']))
    torch.save(best_taus, tau_name)
    lay_name = ('{}/snn_tau_train_last_layers.pt'.format(params['Dataout_dirname']))
    torch.save(last_layers, lay_name)
    tau_name = ('{}/snn_tau_train_last_tau_alpha.pt'.format(params['Dataout_dirname']))
    torch.save(last_taus, tau_name)
    # Save variables: return loss_hist, accs_hist, best_acc_layers, best_tau_layers, ttc_hist for visualization
    f_name = ('{}/snn_tau_train_loss_epoch.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(f_name, 'wb') as handle:
        pickle.dump(np.array(loss_hist), handle, protocol=4)
    g_name = ('{}/snn_tau_train_accs_tv_epoch.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(g_name, 'wb') as handle:
        pickle.dump(np.array(acc_hist), handle, protocol=4)
    h_name = ('{}/snn_tau_train_best_layers_tde.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(h_name, 'wb') as handle:
        tdelayer = best_layers[0].detach().cpu().numpy()
        pickle.dump(tdelayer, handle, protocol=4)
    hh_name = ('{}/snn_tau_train_best_layers_out.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(hh_name, 'wb') as handle:
        outlayer = best_layers[1].detach().cpu().numpy()
        pickle.dump(outlayer, handle, protocol=4)
    i_name = ('{}/snn_tau_train_best_taus.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(i_name, 'wb') as handle:
        tau_tdes = best_taus.detach().cpu().numpy()
        pickle.dump(tau_tdes, handle, protocol=4)
    h_name = ('{}/snn_tau_train_last_layers_tde.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(h_name, 'wb') as handle:
        tdelayer = last_layers[0].detach().cpu().numpy()
        pickle.dump(tdelayer, handle, protocol=4)
    hh_name = ('{}/snn_tau_train_last_layers_out.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(hh_name, 'wb') as handle:
        outlayer = last_layers[1].detach().cpu().numpy()
        pickle.dump(outlayer, handle, protocol=4)
    i_name = ('{}/snn_tau_train_last_taus.pkl.lzma'.format(params['Dataout_dirname']))
    with lzma.open(i_name, 'wb') as handle:
        tau_tdes = last_taus.detach().cpu().numpy()
        pickle.dump(tau_tdes, handle, protocol=4)
    # Save model layers for classification and confusion matrix later loading values
    return

def train(ds_train, ds_test, opt_params, layers, tau, nb_epochs):
    optimizer = torch.optim.Adamax(opt_params, lr=params['Learning_rate'], betas=(0.9, 0.995), weight_decay=params['WeightDecay'])  # params['lr']
    log_softmax_fn = nn.LogSoftmax(dim=1)  # The log softmax function across output units
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss function
    generator = DataLoader(ds_train, batch_size=params['Batch_size'], shuffle=True, num_workers=params['NWORKERS']) #num_workers = 2 but error parallel so 0

    # The optimization loop
    loss_hist = []
    accs_hist = [[], []]
    for e in range(nb_epochs):
        local_loss = []
        accs = []       # accs: mean training accuracies for each batch
        for x_local, y_local in generator:
            x_local, y_local = x_local.to(device), y_local.to(device)
            #print("Loop: ")
            #print("Memory allocated: {}".format((torch.cuda.memory_allocated() / 1024 ** 2)))
            output, recs, layers_update, tau_update = run_snn_w_tde(x_local, layers, tau)
            #print("Memory allocated: {}".format((torch.cuda.memory_allocated() / 1024 ** 2)))
            del x_local # Memory optimization
            #print("After del- Memory allocated: {}".format((torch.cuda.memory_allocated() / 1024 ** 2)))

            _, spks, outmem = recs
            m = torch.sum(outmem, 1)  # sum spikes of the output over time
            log_p_y = log_softmax_fn(m) # Calculate softmax --> Normalize the output [0,1]
            # Does tau changes? torch.sum(tau-tau_update)

            # Loss function from output and label
            loss_val = loss_fn(log_p_y, y_local)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # memory optimization
            # local_loss is accumulating history across your training loop, since loss_val is a differentiable variable with autograd history. Avoid with float(loss_val) or loss_val.item
            local_loss.append(loss_val.item())

            # compare to labels
            _, am = torch.max(m, 1)  # argmax over output units
            tmp = np.mean((y_local == am).detach().cpu().numpy())
            del y_local # memory optimization
            accs.append(float(tmp))
            #print("End of loop generator Memory allocated: {}".format((torch.cuda.memory_allocated() / 1024 ** 2)))

        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)

        # Train accuracy per epoch. With Dropout we need to recalculate
        mean_accs = np.mean(accs)
        if params['Use_dropout']:
            train_acc, _ = compute_classification_accuracy(ds_train, layers_update, tau_update,early=True)
            accs_hist[0].append(train_acc)
        else:
            accs_hist[0].append(mean_accs)

        # Test accuracy per epoch
        torch.cuda.empty_cache()
        test_acc, _ = compute_classification_accuracy(ds_test, layers_update, tau_update,early=True)
        accs_hist[1].append(test_acc)  # only safe best test

        # Save best test-->[1]    train-->[0]
        if np.max(test_acc) >= np.max(accs_hist[1]):
        #if np.max(mean_accs) >= np.max(accs_hist[0]):
            best_acc_layers = []
            best_tau_layers = tau_update.detach().clone()
            for ii in layers_update:
                best_acc_layers.append(ii.detach().clone())

        # Print each 100? only when e%100 == 0
        print("Epoch {}/{} done. Train accuracy: {:.2f}%, Test accuracy: {:.2f}%, OUT:{}.".format(e + 1, nb_epochs,
                                                                                          accs_hist[0][-1] * 100,
                                                                                          accs_hist[1][-1] * 100,
                                                                                          params['Dataout_dirname']))
    del layers, tau
    last_acc_layers = []
    last_tau_layers = tau_update.detach().clone()
    for ii in layers_update:
        last_acc_layers.append(ii.detach().clone())
    return loss_hist, accs_hist, best_acc_layers, best_tau_layers, last_acc_layers, last_tau_layers

def main_training():
    dataset_dir = 'spikes_tidigits_noise0'
    global params
    global spike_fn # scale of the backward surrogate gradient is defined inside the SurrGradSpike function
    spike_fn = SurrGradSpike().apply

    # Read Data input from files test=True reduced the samples 1/4 to 1000 for testing purposes
    [ds_train, ds_test], labels = generate_keywords_dataset_singleKW(dataset_dir, device, letter_written=letters, bin_size=params['Bin_size'], inference=False, train_percent=params['Train_percent'], category=Category)
    # Defined the network architecture layers and the parameters to optimize.
    # Time of the decay is based on the step_size. Every step in the cell dynamic is not equivalent to one input anymore
    layers, tau, opt_params = define_globals(time_size=params['Step_size'], load_model=params['Load_saved_model'])
    if params['Load_saved_model']:
        check_accuracies(ds_train, ds_test, layers, tau)
    # Train: Minimize the loss function according to the parameters. Update every epoch and iterate
    #        Involves already the testing accuracy as a parameter to optimize
    loss_hist, acc_hist, best_layers, best_taus, last_acc_layers, last_tau_layers = train(ds_train, ds_test, opt_params, layers, tau, params['Epochs'])
    #          acc_hist[0] training      acc_hist[1] testing
    # Plot figures
    plot_figures_epochs(loss_hist, acc_hist, params['Dataout_dirname'])
    # Save params for postprocessing and load model
    save_network_params(best_layers, best_taus, loss_hist, acc_hist, last_acc_layers, last_tau_layers)
    # Print the latest accuracy values Train/Test for the trained
    check_accuracies(ds_train, ds_test, best_layers, best_taus)
    return

def main_inference():
    ''' Use always with Load_model True and the directory correctly pointing to the model on top '''
    dataset_dir = 'spikes_tidigits_noise0'
    global params
    global spike_fn # scale of the backward surrogate gradient is defined inside the SurrGradSpike function
    spike_fn = SurrGradSpike().apply

    # Read Data input from files test=True reduced the samples 1/4 to 1000 for testing purposes
    [ds_full], labels = generate_keywords_dataset_singleKW(dataset_dir, device, letter_written=letters, bin_size=params['Bin_size'], inference=True, category=Category)
    # Defined the network architecture layers and the parameters to optimize.
    # Time of the decay is based on the step_size. Every step in the cell dynamic is not equivalent to one input anymore
    layers, tau, opt_params = define_globals(time_size=params['Step_size'], load_model=True)
    # Always TRUE here - if Load_saved_model:
    train_acc, [spks_mid, spks_out] = compute_classification_accuracy(ds_full, layers=layers, tau=tau, early=True)

    # Print spikes
    print("****** File directory of experiment {} *******".format(params['Datain_dirname']))
    spks_midsum = np.sum(spks_mid)
    print("Total Spikes in Middle layer:{}   Shape:{} ".format(spks_midsum, spks_mid.shape))
    spks_outsum = np.sum(spks_out)
    print("Total Spikes in Output layer:{}   Shape:{} ".format(spks_outsum, spks_out.shape))
    # Save Spikes for matlab -- better offline with the script
    save_spikes_variables(spks_mid, spks_out, labels)

    return


def save_spikes_variables(spks_mid, spks_out, labels):
    for i in np.unique(labels):
        kword = letters[i]
        ind_key = (labels == i)
        g_name = ('{}/layer1/spikes_layer1_{}.pkl.lzma'.format(params['Dataout_dirname'], kword))
        with lzma.open(g_name, 'wb') as handle:
            pickle.dump(np.array(spks_mid[ind_key]), handle, protocol=4)
        g_name = ('{}/layer2/spikes_layer2_{}.pkl.lzma'.format(params['Dataout_dirname'], kword))
        with lzma.open(g_name, 'wb') as handle:
            pickle.dump(np.array(spks_out[ind_key]), handle, protocol=4)
    return

if __name__ == "__main__":
    main_training()
    #main_inference()

