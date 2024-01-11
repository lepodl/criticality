# -*- coding: utf-8 -*- 
# @Time : 2023/5/11 16:12 
# @Author : lepold
# @File : supp_cases.py

import unittest

import matplotlib.pyplot as plt
import numpy as np

from analysis.getpsd import get_psd


class Testcase(unittest.TestCase):
    def _test_plot_inhibitory_time_scale_regulating(self):
        from models.simulation import simulation_for_imean_and_raster
        log, syn_current = simulation_for_imean_and_raster(block_path="../blocks/2000_neurons_tau_10/d100_w01",
                                                           Time=4000,
                                                           document_time=3600, delta_t=0.1,
                                                           ou_param={'i_mean': 0.45, 'i_sigma': 0.15, "tau_i": 2.5},
                                                           inhi_tau=15.,
                                                           specified_gui=(2.2 * 1e-3, 0.015 * 1e-3, 3. * 1e-3, 0. * 1e-3))
        np.savez("../data/inhi_15.npz", log=log, syn_current=syn_current)

    def test_plot_raster_and_psd(self):
        file = np.load("../data/inhi_15.npz")
        log = file["log"][-10000:, :]
        n = log.shape[1]
        syn_current = file["syn_current"]
        freqs, power = get_psd(syn_current, fs=10000, resolution=0.5)
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        ax.plot(np.arange(10000), syn_current[-10000:])
        ax.set_xlabel("time")
        ax.set_ylabel("signal")
        ax2.plot(freqs[:100], power[:100])
        ax2.set_xlabel("freqs")
        ax2.set_ylabel("psd")
        fig.savefig("../figs/psd_inhi_15.pdf", dpi=100)
        plt.close(fig)
        fig = plt.figure(figsize=(5, 3))
        ax = fig.gca()
        spike_data = [log[:, i].nonzero()[0] for i in range(log.shape[1])]
        ax.eventplot(spike_data[:int(4 / 5 * n)], lineoffsets=np.arange(1, int(4 / 5 * n) + 1), colors="tab:blue")
        ax.eventplot(spike_data[int(4 / 5 * n):], lineoffsets=np.arange(int(4 / 5 * n) + 1, n + 1), colors="tab:red")
        ax.set_xlabel("Time")
        ax.set_ylabel('neuron')
        fig.savefig("../figs/raster_inhi_15.pdf", dpi=100)
        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
