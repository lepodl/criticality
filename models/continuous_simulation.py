# -*- coding: utf-8 -*- 
# @Time : 2022/12/9 17:23 
# @Author : lepold
# @File : continuous_simulation.py

import os
import time

import torch
from mpi4py import MPI

from analysis.avalanches import compute_avalanche, fit_powerlaw
from analysis.spike_statistics import *
from generation.read_block import connect_for_block
from models.block_new import block as block_new
from matplotlib import gridspec

"""
gui = np.array([0.5, 0.1, 6., 0.0])  # mean driven, i_mean=0.45, i_sigma=0.15, tau_i=2.5, short inhibitory time constant. tau_gaba=10
gui = np.array([0.5, 0.1, 3., 0.])  # fluctuation driven, i_mean=0.18, i_sigma=0.6, tau_i=2.5, short inhibitory time constant. tau_gaba=10
"""
def assign_parameters(gui_base=(0.5, 0.1, 6., 0.0), n=30):
    total = gui_base[0] + 20 * gui_base[1]
    ampa = np.linspace(0.5, 1., n) * total
    nmda = (total - ampa) / 20.
    param = np.stack([ampa, nmda, np.ones(n) * gui_base[2], np.zeros(n)], axis=1)
    return param


def simulation_via_background_current(block_path="small_block_d100", ratio_index=1, Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    write_path = kwargs.get("write_path", None)
    os.makedirs(write_path, exist_ok=True)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    N = property.shape[0]
    n = 100
    s = int(4 / 5 * N - 4 / 5 * n)
    e = s + n
    w_uij = w_uij.cuda("cuda:0")
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=0.48,
        i_sigma=0.1,
        tau_i=2.5
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(document_iter_time, n), dtype=np.uint8)
    spike_count = np.zeros(shape=(iter_time))
    synaptic_current_total = np.zeros(shape=(document_iter_time, int(4 * n / 5)))
    background_current = np.zeros(shape=(document_iter_time, int(4 * n / 5)))
    synaptic_current = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    for t in range(iter_time):
        # print(t, end='\r')
        B.run(debug_mode=False)
        spike_count[t] = np.sum(B.active.data.cpu().numpy()[s:int(s + 4 / 5 * n)])
        if t >= document_start:
            log_all[t - document_start] = B.active.data.cpu().numpy()[s:e]
            synaptic_current_total[t - document_start] = B.I_ui.sum(axis=0).cpu().numpy()[s:int(s + 4 / 5 * n)]
            background_current[t - document_start] = B.i_background.cpu().numpy()[s: int(s + 4 / 5 * n)]
            synaptic_current[t - document_start] = B.I_ui.mean(axis=-1).cpu().numpy()
    end = time.time()
    print(f"cost time: {end - start:.2f}")
    torch.cuda.empty_cache()
    pcc_log = pearson_cc(log_all[:, :int(4 / 5 * n)], pairs=100)
    cc_log = correlation_coefficent(log_all[:, :int(4 / 5 * n)], bin_width=20)
    mean_fr = mean_firing_rate(log_all[:, :int(4 / 5 * n)])
    cv = coefficient_of_variation(log_all[:, :int(4 / 5 * n)])
    total_current = synaptic_current_total + background_current
    act_idx = np.where(log_all[:, :int(4 / 5 * n)].sum(axis=0) > 20)[0]
    i_std = np.mean(total_current[act_idx].std(axis=1) / total_current[act_idx].mean(axis=1))
    np.savez(os.path.join(write_path, f"result_{ratio_index}.npz"), cc=cc_log, fr=mean_fr, i_std=i_std, cv=cv,
             plv=np.nan, pcc=pcc_log)

    spike_count = spike_count.reshape((-1, 5))
    spike_count = spike_count.sum(axis=1)[1000:]
    if 0.003 < mean_fr < 0.2:
        _, aval_size, aval_dur, _ = compute_avalanche(spike_count, threshold=1, if_spike=False)
        alpha_size, distance_size = fit_powerlaw(aval_size)
        alpha_dur, distance_dur = fit_powerlaw(aval_dur)
    else:
        alpha_size, distance_size, alpha_dur, distance_dur = 0., 0., 0., 0.
    print("mean_fr", mean_fr)
    print("cc: ", cc_log)
    print("pcc: ", pcc_log)

    log_all = log_all[-10000:, ]
    spike_data = [log_all[:, i].nonzero()[0] for i in range(n)]
    fr = instantaneous_rate(log_all, bin_width=20)
    rate_time_series_auto_kernel = gaussian_kernel_inst_rate(log_all, 20, 100)
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_axes([0.1, 0.6, 0.8, 0.3])
    ax.eventplot(spike_data[:int(4 / 5 * n)], lineoffsets=np.arange(1, int(4 / 5 * n) + 1), colors="tab:blue")
    ax.eventplot(spike_data[int(4 / 5 * n):], lineoffsets=np.arange(int(4 / 5 * n) + 1, n + 1), colors="tab:red")
    ax.set_xticks([])
    ax.set_ylabel('neuron')
    info = f"fr: {mean_fr:.3f}, cc: {cc_log:.2f}, cv: {cv:.2f} " + "|" + f" aval size: slope:{alpha_size:.2f} and dis:{distance_size:.2f} " + "|" + f" aval dur: slope:{alpha_dur:.2f} and dis:{distance_dur:.2f}"
    ax.text(0.2, 1.05, info, transform=ax.transAxes)
    ax.text(0.2, 1.05, info, transform=ax.transAxes)
    ax.invert_yaxis()
    ax = fig.add_axes([0.1, 0.32, 0.8, 0.23])
    ax.plot(rate_time_series_auto_kernel, color='0.2')
    ax.plot(fr, color='0.8')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_ylabel("fr(KHz)")
    ax.set_xticks([])
    ax = fig.add_axes([0.1, 0.05, 0.8, 0.23])
    synaptic_current = synaptic_current[-10000:]
    epsc = np.sum(synaptic_current[:, :2], axis=1)
    ipsc = np.sum(synaptic_current[:, 2:], axis=1)
    psc = np.sum(synaptic_current, axis=1)
    ax.plot(np.arange(len(synaptic_current)), epsc, c="b", alpha=0.8, label="Exc")
    ax.plot(np.arange(len(synaptic_current)), ipsc, c="r", alpha=0.8, label="Inh")
    ax.plot(np.arange(len(synaptic_current)), psc, c="k", alpha=0.7, label="Net")
    ax.set_xlim([0, len(synaptic_current)])
    ax.set_xticks(np.linspace(0, len(synaptic_current), 5))
    ax.set_xticklabels(np.linspace(0, document_time, 5))
    ax.set_xlabel("time(ms)")
    ax.set_ylabel("current")
    ax.legend(loc="best")

    fig.savefig(os.path.join(write_path, f"fig_{ratio_index}.png"), dpi=100)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

params = assign_parameters(gui_base=(0.5, 0.1, 6., 0.), n=10)

for id in range(rank, 300, size - 1):
    i = int(id % 10)
    j = int(id // 10)
    simulation_via_background_current("../blocks/2000_neurons_tau_10/d100_w01",
                                      write_path=f"../data/slow_current_mean_driven.npz/trial_{j}",
                                      Time=3000, document_time=2000, delta_t=0.1,
                                      specified_gui=params[i] * 1e-3, ratio_index=i)
    comm.send(1, dest=size - 1)

if rank == size - 1:
    for id in range(size - 1):
        info = comm.recv(source=id)
        assert info == 1
    time.sleep(30)
    colors = ["#4C72B0", "#C44E52", "#000000"]
    cc = np.zeros((10, 30))
    fr = np.zeros((10, 30))
    cv = np.zeros((10, 30))
    i_std = np.zeros((10, 30))
    plv = np.zeros((10, 30))
    pcc = np.zeros((10, 30))
    for j in range(30):
        for i in range(10):
            file = np.load(os.path.join(f"../data/slow_current_mean_driven/trial_{j}", f"result_{i}.npz"))
            cc[i, j] = file['cc']
            fr[i, j] = file['fr'] * 1000
            cv[i, j] = file['cv']
            pcc[i, j] = file['pcc']
            i_std[i, j] = file['i_std']
            plv[i, j] = file['plv']
    fig = plt.figure(figsize=(8, 8))
    axes = dict()
    gs = gridspec.GridSpec(2, 2)
    gs.update(left=0.1, right=0.94, top=0.93, bottom=0.1, wspace=0.2, hspace=0.2)
    axes['A'] = plt.subplot(gs[0, 0])
    axes['C'] = plt.subplot(gs[1, 0])
    axes['B'] = plt.subplot(gs[0, 1])
    axes['D'] = plt.subplot(gs[1, 1])
    ratio = np.linspace(0.5, 1., 10)
    index = np.arange(10, dtype=np.int_)
    axes['A'].errorbar(ratio[index], fr.mean(axis=1)[index], yerr=fr.std(axis=1)[index], fmt="o-", capsize=3.,
                       errorevery=1, c=colors[0], lw=1.5, ecolor=colors[0], elinewidth=1.)
    axes['B'].errorbar(ratio[index], cv.mean(axis=1)[index], yerr=cv.std(axis=1)[index], fmt="o-", capsize=3.,
                       errorevery=1, c=colors[0], lw=1.5, ecolor=colors[0], elinewidth=1.)
    axes['C'].errorbar(ratio[index], i_std.mean(axis=1)[index], yerr=i_std.std(axis=1)[index], fmt="o-", capsize=3.,
                       errorevery=1, c=colors[0], lw=1.5, ecolor=colors[0], elinewidth=1.)
    axes['D'].errorbar(ratio[index], cc.mean(axis=1)[index], yerr=cc.std(axis=1)[index], fmt="o-", capsize=3.,
                       errorevery=1, c=colors[0], lw=1.5, ecolor=colors[0], elinewidth=1.)
    labels = ['Firing rate', 'CV', 'Std (syn_current)', 'Coherence index']
    for label, name in zip(['A', 'B', 'C', 'D'], labels):
        axes[label].set_ylabel(name)
        axes[label].set_xlabel("Fast-acting ratio")
    np.savez(os.path.join("../data/slow_current_mean_driven", "result.npz"), cc=cc, fr=fr, cv=cv,
             i_std=i_std, plv=plv, ratio=ratio, pcc=pcc)
    fig.savefig(os.path.join("../data/slow_current_mean_driven", "result.png"), dpi=100)
