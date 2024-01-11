# @File : grid_simulation.py
# -*- coding: utf-8 -*-
# @Time : 2022/12/9 17:23
# @Author : lepold
# @File : continuous_simulation.py

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpi4py import MPI

from analysis.avalanches import compute_kaapa
from analysis.getpsd import get_power
from generation.read_block import connect_for_block
from models.block_new import block as block_new


def assign_parameters(gui_base=(0.5, 0.1, 6., 0.0), n=30):
    total = gui_base[0] + 20 * gui_base[1]
    ampa = np.linspace(0.6, 1., n) * total
    nmda = (total - ampa) / 20.
    param = np.stack([ampa, nmda, np.ones(n) * gui_base[2], np.zeros(n)], axis=1)
    return param


def simulation_via_background_current(block_path="small_block_d100", Time=3000, delta_t=0.1, **kwargs):
    start = time.time()
    specified_gui = kwargs.get("specified_gui", None)
    document_time = kwargs.get("document_time", None)
    write_path = kwargs.get("write_path", None)
    ou_param = kwargs.get("ou_param", None)

    os.makedirs(write_path, exist_ok=True)
    property, w_uij = connect_for_block(block_path)
    if specified_gui is not None:
        property[:, 10:14] = torch.tensor(specified_gui)

    property = property.cuda("cuda:0")
    w_uij = w_uij.cuda("cuda:0")

    if ou_param is None:
        ou_param = {'i_mean': 0.4, 'i_sigma': 0.1, "tau_i": 5.}
    global i_mean
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

    spike_count = np.zeros(shape=(document_iter_time))
    kappas = []
    powers = []
    for idx in range(len(i_mean)):
        B.clear()
        B.i_mean = i_mean[idx]
        for t in range(iter_time):
            B.run(debug_mode=False)
            if t >= document_start:
                spike_count[t - document_start] = np.sum(B.active.data.cpu().numpy()[:1600])
        end = time.time()
        print(f"done {idx}, cost time: {end - start:.2f}")

        kappa = compute_kaapa(spike_count, threshold=None, no_Bins=10)
        spike_count2 = spike_count.reshape((-1, 5))
        spike_count2 = spike_count2.sum(axis=1)
        power = get_power(spike_count2, fs=2000)[:, 0]
        kappas.append(kappa)
        powers.append(list(power))
    torch.cuda.empty_cache()
    return kappas, powers


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

params = assign_parameters(gui_base=(0.5, 0.1, 3., 0.), n=20)
ou = {'i_mean': 0.18, 'i_sigma': 0.60, "tau_i": 2.5}
i_mean = np.linspace(0.18, 0.28, 20)
for id in range(rank, 20, size - 1):
    kappas, powers = simulation_via_background_current("../blocks/2000_neurons_tau_10/d100_w01",
                                                       write_path=f"../data/grid_search/fast_current2",
                                                       Time=4000, document_time=3500, delta_t=0.1, ou_param=ou,
                                                       specified_gui=params[id] * 1e-3)
    info = {"kappas": kappas, "powers": powers}
    comm.send(info, dest=size - 1)

if rank == size - 1:
    kappas = np.zeros((20, 20), dtype=np.float32)

    delta_powers = np.zeros((20, 20), dtype=np.float32)
    alpha_powers = np.zeros((20, 20), dtype=np.float32)
    beta_powers = np.zeros((20, 20), dtype=np.float32)
    gamma_powers = np.zeros((20, 20), dtype=np.float32)
    for id in range(size - 1):
        info = comm.recv(source=id)
        kappas[id] = np.array(info["kappas"])
        powers = info["powers"]
        delta_powers[id] = np.array(powers)[:, 0]
        alpha_powers[id] = np.array(powers)[:, 1]
        beta_powers[id] = np.array(powers)[:, 2]
        gamma_powers[id] = np.array(powers)[:, 3]
    time.sleep(10)
    fig = plt.figure(figsize=(5, 10), dpi=100)
    ax = fig.add_subplot(1, 2, 1)
    im = ax.imshow(kappas.T, cmap='RdBu_r', interpolation='gaussian')
    cb = fig.colorbar(im, shrink=0.8)
    ax.set_title("Kappa")
    ampa = params[:, 0]
    Delta = i_mean - 0.5

    ax2 = fig.add_subplot(1, 2, 2)
    im2 = ax2.imshow(gamma_powers.T, cmap='RdBu_r', interpolation='gaussian')
    cb2 = fig.colorbar(im2, shrink=0.8)
    ax2.set_title("Gamma")

    yticks = np.linspace(0, 20 - 1, 4, endpoint=True, dtype=np.int8)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{data:.2f}' for data in Delta[yticks]], rotation=60)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{data:.2f}' for data in Delta[yticks]], rotation=60)

    xticks = np.linspace(0, 20 - 1, 4, endpoint=True, dtype=np.int8)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{data:.2f}' for data in ampa[xticks]], )
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([f'{data:.2f}' for data in ampa[xticks]], )

    ax.set_ylabel(r"Delta")
    ax.set_xlabel(r"Ampa")
    ax2.set_ylabel(r"Delta")
    ax2.set_xlabel(r"Ampa")

    fig.tight_layout()
    fig.savefig("../data/grid_search/fast_current2/info.png")
    np.savez("../data/grid_search/fast_current2/info.npz", ampa=ampa, Delta=Delta, kappas=kappas,
             alpha_powers=alpha_powers, beta_powers=beta_powers, gamma_powers=gamma_powers, delta_powers=delta_powers)
    print("Done!")
