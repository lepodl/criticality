# -*- coding: utf-8 -*- 
# @Time : 2022/12/8 20:57 
# @Author : lepold
# @File : simulation.py
import os
import time

import torch
from tqdm import tqdm

from analysis.spike_statistics import *
from generation.read_block import connect_for_block
from models.block_new import block as block_new


def simulation_via_background_current(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    show_result = kwargs.get("show_result", False)
    ou_param = kwargs.get("ou_param", None)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    N = property.shape[0]
    n = 200
    s = int(4 / 5 * N - 4 / 5 * n)
    e = s + n
    if ou_param is None:
        ou_param = {'i_mean': 0.4, 'i_sigma': 0.1, "tau_i": 5.}
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=ou_param['i_mean'],
        i_sigma=ou_param['i_sigma'],
        tau_i=ou_param['tau_i']
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(iter_time, n), dtype=np.uint8)
    synaptic_current = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    sample_ou = np.zeros(shape=(document_iter_time), dtype=np.float32)
    # synaptic_current_total = np.zeros(shape=(document_iter_time, int(4 * n / 5)))
    for t in tqdm(range(iter_time)):
        # print(t, end='\r')
        B.run(debug_mode=False)
        log_all[t] = B.active.data.cpu().numpy()[s:e]
        if t >= document_start:
            synaptic_current[t - document_start] = B.I_ui.mean(axis=-1).cpu().numpy()
            sample_ou[t - document_start] = B.i_background.data.cpu().numpy()[50]
            # synaptic_current_total[t - document_start] = B.I_syn.cpu().numpy()[s:int(s + 4 / 5 * n)]
    end = time.time()
    print(f"cost time: {end - start:.2f}")
    torch.cuda.empty_cache()
    fr = instantaneous_rate(log_all, bin_width=20)

    if show_result:
        # pcc_log = pearson_cc(log_all[:, :int(4 / 5 * n)], pairs=100)
        cc_log = correlation_coefficent(log_all[:, :int(4 / 5 * n)], bin_width=20)
        mean_fr = mean_firing_rate(log_all[:, :int(4 / 5 * n)])
        cv = coefficient_of_variation(log_all)
        spike_data = [log_all[:, i].nonzero()[0] for i in range(n)]
        print("mean_fr", mean_fr)
        print("cc: ", cc_log)
        # print("pcc: ", pcc_log)
        print("cv:", cv)
        log_all = log_all[-document_iter_time:, ]
        fr = instantaneous_rate(log_all, bin_width=20)
        rate_time_series_auto_kernel = gaussian_kernel_inst_rate(log_all, 20, 100)
        fig = plt.figure(figsize=(10, 7), dpi=100)
        ax = fig.add_axes([0.1, 0.7, 0.8, 0.24])

        ax.eventplot(spike_data, lineoffsets=np.arange(1, int(n) + 1), colors="tab:blue")
        # ax.eventplot(spike_data[int(4 / 5 * n):], lineoffsets=np.arange(int(4 / 5 * n) + 1, n + 1), colors="tab:red")
        ax.set_xticks([])
        ax.set_ylabel('neuron')
        ax.text(0.2, 1.04, f"fr: {mean_fr:.3f}, cc: {cc_log:.2f}, cv: {cv:.2f}, ou_param: {ou_param!r}",
                transform=ax.transAxes)
        ax.invert_yaxis()
        ax = fig.add_axes([0.1, 0.47, 0.8, 0.18])
        ax.plot(rate_time_series_auto_kernel, color='0.2')
        ax.plot(fr, color='0.8')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylabel("fr(KHz)")
        # ax.set_xticks([])
        ax = fig.add_axes([0.1, 0.24, 0.8, 0.18])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        epsc = np.sum(synaptic_current[:, :2], axis=1)
        ipsc = np.sum(synaptic_current[:, 2:], axis=1)
        psc = np.sum(synaptic_current, axis=1)
        ax.plot(np.arange(len(synaptic_current)), epsc, c="b", alpha=0.8, label="Exc")
        ax.plot(np.arange(len(synaptic_current)), ipsc, c="r", alpha=0.8, label="Inh")
        ax.plot(np.arange(len(synaptic_current)), psc, c="k", alpha=0.7, label="Net")
        # ax.set_xticks([])
        ax.legend(loc="best")
        ax.set_ylabel("current")

        ax = fig.add_axes([0.1, 0.06, 0.8, 0.15])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.plot(np.arange(len(sample_ou)), sample_ou, c="g", alpha=0.8, label="ou", lw=1.)
        ax.legend(loc="best")
        ax.set_xticks(np.linspace(0, len(sample_ou), 5))
        ax.set_xticklabels(np.linspace(0, len(sample_ou) / 10, 5))
        ax.set_xlabel("time(ms)")
        ax.set_ylabel("current")
        return fig
    else:
        return fr

def simulation_for_gating_variable(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    show_result = kwargs.get("show_result", False)
    ou_param = kwargs.get("ou_param", None)
    const = kwargs.get("slow_const", None)
    print("slow const", const)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    N = property.shape[0]
    n = 200
    s = int(4 / 5 * N - 4 / 5 * n)
    e = s + n
    if ou_param is None:
        ou_param = {'i_mean': 0.4, 'i_sigma': 0.1, "tau_i": 5.}
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=ou_param['i_mean'],
        i_sigma=ou_param['i_sigma'],
        tau_i=ou_param['tau_i']
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(iter_time, n), dtype=np.uint8)
    jui = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    # synaptic_current_total = np.zeros(shape=(document_iter_time, int(4 * n / 5)))
    for t in tqdm(range(iter_time)):
        # print(t, end='\r')
        if const is not None:
            B.J_ui[1, :] = float(const)
        B.run(debug_mode=False)
        log_all[t] = B.active.data.cpu().numpy()[s:e]
        if t >= document_start:
            jui[t - document_start] = B.J_ui.cpu().numpy()[:, 188]
    end = time.time()
    print(f"cost time: {end - start:.2f}")
    jui = jui * specified_gui
    jui[:, 1] = jui[:, 1] * 10
    torch.cuda.empty_cache()

    if show_result:
        # pcc_log = pearson_cc(log_all[:, :int(4 / 5 * n)], pairs=100)
        cc_log = correlation_coefficent(log_all[:, :int(4 / 5 * n)], bin_width=20)
        mean_fr = mean_firing_rate(log_all[:, :int(4 / 5 * n)])
        cv = coefficient_of_variation(log_all)
        spike_data = [log_all[:, i].nonzero()[0] for i in range(n)]
        print("mean_fr", mean_fr)
        print("cc: ", cc_log)
        # print("pcc: ", pcc_log)
        print("cv:", cv)
        log_all = log_all[-document_iter_time:, ]
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_axes([0.1, 0.52, 0.8, 0.4])

        ax.eventplot(spike_data, lineoffsets=np.arange(1, int(n) + 1), colors="tab:blue")
        # ax.eventplot(spike_data[int(4 / 5 * n):], lineoffsets=np.arange(int(4 / 5 * n) + 1, n + 1), colors="tab:red")
        ax.set_xticks([])
        ax.set_ylabel('neuron')
        ax.text(0.2, 1.04, f"fr: {mean_fr:.3f}, cc: {cc_log:.2f}, cv: {cv:.2f}, ou_param: {ou_param!r}",
                transform=ax.transAxes)
        ax.invert_yaxis()

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.35])
        names = ["fast", "10 slow", "inhi"]
        for i in range(3):
            ax.plot(jui[:, i], label=names[i])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylabel("gating variable")
        ax.legend(loc="best")
        ax.set_xlabel("time")
        return fig
    else:
        return None

def simulation_for_jui(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    show_result = kwargs.get("show_result", False)
    ou_param = kwargs.get("ou_param", None)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    N = property.shape[0]
    n = 200
    s = int(4 / 5 * N - 4 / 5 * n)
    e = s + n
    if ou_param is None:
        ou_param = {'i_mean': 0.4, 'i_sigma': 0.1, "tau_i": 5.}
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=ou_param['i_mean'],
        i_sigma=ou_param['i_sigma'],
        tau_i=ou_param['tau_i']
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_time = int(iter_time / 2)
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    log_all = np.zeros(shape=(iter_time, n), dtype=np.uint8)
    jui = np.zeros(shape=(document_iter_time, 4), dtype=np.float32)
    # synaptic_current_total = np.zeros(shape=(document_iter_time, int(4 * n / 5)))
    for t in tqdm(range(iter_time)):
        # print(t, end='\r')
        B.run(debug_mode=False)
        log_all[t] = B.active.data.cpu().numpy()[s:e]
        if t >= document_start:
            jui[t - document_start] = B.J_ui.cpu().numpy()[:, 188]
    end = time.time()
    print(f"cost time: {end - start:.2f}")
    # jui = jui * specified_gui
    torch.cuda.empty_cache()

    if show_result:
        # pcc_log = pearson_cc(log_all[:, :int(4 / 5 * n)], pairs=100)
        cc_log = correlation_coefficent(log_all[:, :int(4 / 5 * n)], bin_width=20)
        mean_fr = mean_firing_rate(log_all[:, :int(4 / 5 * n)])
        cv = coefficient_of_variation(log_all)
        spike_data = [log_all[:, i].nonzero()[0] for i in range(n)]
        print("mean_fr", mean_fr)
        print("cc: ", cc_log)
        # print("pcc: ", pcc_log)
        print("cv:", cv)
        log_all = log_all[-document_iter_time:, ]
        fr = instantaneous_rate(log_all, bin_width=20)
        rate_time_series_auto_kernel = gaussian_kernel_inst_rate(log_all, 20, 100)
        fig = plt.figure(figsize=(8, 6), dpi=100)
        ax = fig.add_axes([0.1, 0.52, 0.8, 0.4])

        ax.eventplot(spike_data, lineoffsets=np.arange(1, int(n) + 1), colors="tab:blue")
        # ax.eventplot(spike_data[int(4 / 5 * n):], lineoffsets=np.arange(int(4 / 5 * n) + 1, n + 1), colors="tab:red")
        ax.set_xticks([])
        ax.set_ylabel('neuron')
        ax.text(0.2, 1.04, f"fr: {mean_fr:.3f}, cc: {cc_log:.2f}, cv: {cv:.2f}, ou_param: {ou_param!r}",
                transform=ax.transAxes)
        ax.invert_yaxis()

        ax = fig.add_axes([0.1, 0.1, 0.8, 0.35])
        names = ["fast", "slow", "inhi"]
        for i in range(3):
            ax.plot(jui[:, i], label=names[i])
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.set_ylabel("jui")
        ax.legend(loc="best")
        ax.set_xlabel("time")
        return fig
    else:
        return None


def simulation_count(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=0.18,
        i_sigma=0.6,
        tau_i=2.5
    )
    iter_time = int(Time / delta_t)

    if document_time is not None:
        document_iter_time = int(document_time / delta_t)
        document_start = iter_time - document_iter_time
    else:
        document_iter_time = int(iter_time / 2)
        document_start = iter_time - document_iter_time

    spike_count = np.zeros(shape=(document_iter_time, 20), dtype=np.int_)
    indexs = np.ones((20, 400), dtype=np.int_)
    for idx in range(20):
        index = np.random.choice(1600, 400, replace=False)
        indexs[idx] = index
    for tt in range(iter_time):
        # print(tt, end='\r')
        if tt %1000==0:
            print(tt)
        B.run(debug_mode=False)
        if tt >= document_start:
            act = B.active.data.cpu().numpy()
            for idx in range(20):
                index = indexs[idx]
                spike_count[tt - document_start, idx] = act[index].sum()
    spike_count = spike_count.reshape((-1, 5, 20))
    spike_count = spike_count.sum(axis=1)
    os.makedirs(os.path.join("../data/fast_current", "spike_count1"), exist_ok=True)
    np.save(os.path.join("../data/fast_current", "spike_count1" + ".npy"), spike_count)
    torch.cuda.empty_cache()
    end = time.time()
    print(f"cost time: {end - start:.2f}")

def simulation_for_imean_and_raster(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    inhi_tau = kwargs.get("inhi_tau", 10.)
    ou_param = kwargs.get("ou_param", None)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)
    property[:, 20] = inhi_tau

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")
    N = property.shape[0]
    n = 200
    s = int(4 / 5 * N - 4 / 5 * n)
    e = s + n
    if ou_param is None:
        ou_param = {'i_mean': 0.4, 'i_sigma': 0.1, "tau_i": 5.}
    B = block_new(
        node_property=property,
        w_uij=w_uij,
        delta_t=delta_t,
        i_mean=ou_param['i_mean'],
        i_sigma=ou_param['i_sigma'],
        tau_i=ou_param['tau_i']
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
    synaptic_current = np.zeros(shape=(iter_time), dtype=np.float32)
    for t in tqdm(range(iter_time)):
        # print(t, end='\r')
        B.run(debug_mode=False)
        synaptic_current[t] = B.I_syn.cpu().numpy()[s:int(s + 4 / 5 * n)].mean()
        if t >= document_start:
            log_all[t-document_start] = B.active.data.cpu().numpy()[s:e]
    end = time.time()
    print(f"cost time: {end - start:.2f}")
    torch.cuda.empty_cache()
    return log_all, synaptic_current

if __name__ == '__main__':
    path = "../blocks/2000_neurons_tau_10/d100_w01"
    gui = np.array([1., 0.075, 3., 0.0])
    simulation_count(block_path=path, Time=100000, document_time=98000, delta_t=0.1, specified_gui=gui * 1e-3)
