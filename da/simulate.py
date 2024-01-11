# -*- coding: utf-8 -*- 
# @Time : 2022/10/11 10:30 
# @Author : lepold
# @File : simulate_cuda.py


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.bold_model_pytorch import BOLD
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from scipy.ndimage import gaussian_filter1d

from analysis.spike_statistics import instantaneous_rate


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def torch_2_numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def sample_in_voxel_splitEI(aal_region, neurons_per_population_base, num_sample_voxel_per_region=1,
                            num_neurons_per_voxel=300):
    base = neurons_per_population_base
    subblk_base = np.arange(len(aal_region)) * 2
    uni_region = np.unique(aal_region)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)
    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)
    count_voxel = 0
    for i in uni_region:
        print("sampling for region: ", i)
        choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
        for choice in choices:
            sample1 = np.random.choice(
                np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), s1,
                replace=False)
            sample2 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), s2,
                replace=False)
            sample = np.concatenate([sample1, sample2])
            sub_blk = np.concatenate(
                [np.ones_like(sample1) * subblk_base[choice], np.ones_like(sample2) * (subblk_base[choice] + 1)])[:,
                      None]
            # print("sample_shape", sample.shape)
            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
            sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
            sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
        count_voxel += 1

    return sample_idx.astype(np.int64)


def run_simulation(ip, block_path, write_path, hp_path, real_bold_path):
    os.makedirs(write_path, exist_ok=True)
    v_th = -50
    aal_region = np.array([0])
    block_model = block_gpu(ip, block_path, 0.1,
                            route_path=None,
                            force_rebase=False, overlap=2)

    total_neurons = int(block_model.total_neurons)
    neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    populations = block_model.subblk_id
    populations_cpu = populations.cpu().numpy()
    total_populations = int(block_model.total_subblks)
    num_voxel = int(total_populations // 2)
    neurons_per_voxel = block_model.neurons_per_subblk.cpu().numpy()

    neurons_per_voxel, _ = np.histogram(populations_cpu, weights=neurons_per_voxel, bins=num_voxel,
                                        range=(0, num_voxel * 2))

    def _update(param):
        nmda = (2.5 - param) / 20
        param = torch.cat([param, nmda], dim=0).cuda()
        for i, ind in enumerate(np.array([10, 11])):
            population_info = np.stack(np.meshgrid(populations_cpu, ind, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            param_here = torch.repeat_interleave(param[[i]], 2)
            alpha = torch.ones(num_voxel, device="cuda:0") * param * 1e8
            alpha = torch.repeat_interleave(alpha, 2)
            beta = torch.ones(self.num_voxel, device="cuda:0") * 1e8
            beta = torch.repeat_interleave(beta, 2)
            block_model.gamma_property_by_subblk(population_info, )

    sample_idx = load_if_exist(sample_in_voxel_splitEI, os.path.join(write_path, "sample_idx"),
                               aal_region=aal_region,
                               neurons_per_population_base=neurons_per_population_base, num_sample_voxel_per_region=1,
                               num_neurons_per_voxel=300)

    sample_number = sample_idx.shape[0]
    print("sample_num:", sample_number)
    assert sample_idx[:, 0].max() < total_neurons
    sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
    block_model.set_samples(sample_idx)

    # hp = np.load(hp_path)
    # hp = hp.reshape((-1, 2))
    # hp = torch.from_numpy(hp.astype(np.float32)).cuda()
    # T = hp.shape[0]
    # _update(hp[-1, :])
    ampa = torch.tensor([2e-3], dtype=torch.float32)
    _update(ampa)

    T = 300
    _ = block_model.run(8000, freqs=True, vmean=False, sample_for_show=False)
    bold = BOLD(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)
    bolds_out = np.zeros((T, num_voxel))
    for j in range(T):
        print(j)
        temp_spike = []
        act_all = np.zeros((800 * 10, total_populations))
        count = 0
        # _update(hp[j, :])
        for return_info in block_model.run(8000, freqs=True, vmean=False, sample_for_show=True, iou=True):
            act, spike, _ = return_info
            temp_spike.append(spike)
            act_all[count] = act.cpu().numpy()
            count += 1
        act_all = act_all.reshape((800, 10, -1))
        act_all = np.sum(act_all, axis=1)
        for idxx in range(800):
            act = act_all[idxx]
            active, _ = np.histogram(populations_cpu, weights=act, bins=num_voxel,
                                     range=(0, num_voxel * 2))
            active = (active / neurons_per_voxel).reshape(-1)
            active = torch.from_numpy(active.astype(np.float32)).cuda()
            out = bold.run(torch.max(active, torch.tensor([1e-05]).type_as(active)))

        Spike = torch.stack(temp_spike, dim=0)
        bolds_out[j, :] = torch_2_numpy(out)
    np.save(os.path.join(write_path, "spike.npy"), Spike)
    np.save(os.path.join(write_path, "bold.npy"), bolds_out)
    block_model.shutdown()
    print("Simulation Done")

    real_bold_path = None
    if real_bold_path is not None:
        bold_y = np.load(real_bold_path)
        bold_y = bold_y[:, :num_voxel]
        # bold_y = 0.005 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
    else:
        bold_y = None


    fig = plt.figure(figsize=(5, 3), dpi=100)
    ax = fig.gca()
    TT = bolds_out.shape[0]
    xrange = np.arange(TT)
    if bold_y is not None:
        ax.plot(xrange, bold_y[:TT, 0], label="real bold", color="r")
    ax.plot(xrange, bolds_out[:TT, 0], label="simulated bold", color="b")
    ax.set_xlabel("time(800ms)")
    ax.set_ylabel("blod")
    ax.set_ylim([0, 0.08])
    fig.savefig(os.path.join(write_path, "bold.png"))
    plt.close(fig)

    sub_log = Spike[-1600:, ]
    fr = instantaneous_rate(sub_log, bin_width=20)
    rate_time_series_auto_kernel = gaussian_filter1d(fr, 20, axis=-1)
    fig = plt.figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax.grid(False)
    ax.set_xlabel('time(ms)')
    ax = fig.add_axes([0.1, 0.35, 0.8, 0.5])
    ax.grid(False)

    n = sub_log.shape[1]
    spike_data = [sub_log[:, i].nonzero()[0] for i in range(n)]
    ax.eventplot(spike_data[:240], lineoffsets=np.arange(0, 240), colors="tab:blue")
    ax.eventplot(spike_data[240:], lineoffsets=np.arange(240, n), colors="tab:red")
    ax.set_xlim([0, 1600])
    ax.set_ylim([0, 300])
    ax.set_xticks([])
    ax.set_ylabel('neuron')
    ax.invert_yaxis()
    ax.set_aspect(1)
    ax = fig.add_axes([0.1, 0.15, 0.8, 0.2])
    ax.grid(False)
    ax.plot(rate_time_series_auto_kernel, color='0.2')
    ax.set_xlim([0, 1600])
    ax.plot(fr, color='0.8')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_ylabel("fr(KHz)")
    fig.savefig(os.path.join(write_path, "raster.png"))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="simulation")
    parser.add_argument("--ip", type=str, default="11.5.4.2:50051")
    parser.add_argument("--block_path", type=str, default="blokc_path/single")
    parser.add_argument("--write_path", type=str, default="write_path")
    parser.add_argument("--hp_path", type=str, default="write_path")
    parser.add_argument("--real_bold_path", type=str, default="/public/home/ssct004t/project/zenglb/Richdynamics_NN/da/da_results/critical_simulation/bold.npy")
    args = parser.parse_args()
    run_simulation(args.ip, args.block_path, args.write_path, args.hp_path, args.real_bold_path)
