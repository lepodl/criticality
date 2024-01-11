# -*- coding: utf-8 -*- 
# @Time : 2023/4/27 0:10 
# @Author : lepold
# @File : theory.py


import math
import random
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from random import expovariate

# global tau_e1, tau_e2, tau_i, V_e1, V_e2, V_i, g_L, V_L, C, mu_ext, tau_ext, sigma_ext, V_L, omega
tau_e1 = 2.
tau_e2 = 40.
tau_i = 10.
V_e1 = 0.
V_e2 = 0.
V_i = -70.
g_L = 25 * 1e-3
V_L = -70
C = 0.5
mu_ext = 0.45
sigma_ext = 0.15
tau_ext = 2.
omega = 0.5
V_th = -50

# init
delta_t = 0.1  # ms

def run_LIF(pars, exc_spike=None, inh_spike=None, ou_extern=None):
    # Set parameters
    V_th, V_reset = pars['V_th'], pars['V_reset']
    tau_m, g_L = pars['tau_m'], pars['g_L']
    w_e, w_i = pars['w_e'], pars['w_i']
    g_ampa, g_nmda, g_i = pars['g_u']
    V_ampa, V_nmda, V_i = pars['V_u']
    V_init, E_L = pars['V_init'], pars['E_L']
    dt, T = pars['dt'], pars['T']
    tref = pars['tref']
    C = pars['C']
    tau_ampa, tau_nmda, tau_i = pars['tau_ei']
    init_t = 0.

    # Initialize voltage
    v = np.zeros(int(T / dt), dtype=np.float32)
    v[0] = V_init
    steps = len(v)

    s_exc1 = 0.  # gating variable
    s_exc2 = 0.  # gating variable
    s_inh = 0.  # gating variable
    if exc_spike is not None and exc_spike != 0:
        if isinstance(exc_spike, float):
            mu_e = exc_spike
        elif isinstance(exc_spike, dict):
            mu_e = exc_spike['lambda']
        else:
            raise NotImplementedError
        for j in range(10):
            num = int(int(steps * dt) * mu_e + j * 10)
            spike_e = np.array([expovariate(mu_e) for _ in range(num)], dtype=np.int_) * (1 / dt)
            spike_e = np.add.accumulate(spike_e)
            if spike_e[-1] > steps:
                break

    if inh_spike is not None and inh_spike != 0:
        if isinstance(inh_spike, float):
            mu_i = inh_spike
        elif isinstance(inh_spike, dict):
            mu_i = inh_spike['lambda']
        else:
            raise NotImplementedError
        for j in range(10):
            num = int(int(steps * dt) * mu_i + j * 10)
            spike_i = np.array([expovariate(mu_i) for _ in range(num)], dtype=np.int_) * (1 / dt)
            spike_i = np.add.accumulate(spike_i)
            if spike_i[-1] > steps:
                break

    if ou_extern is not None:
        assert isinstance(ou_extern, dict)
        ou_mean = ou_extern['mean']
        ou_sigma = ou_extern['sigma']
        ou_tau = ou_extern["tau"]
        iou = ou_mean
    else:
        iou = 0.

    # Loop over time
    rec_spikes = []  # record spike times
    tr = -tref  # the count for refractory duration
    iou_all = [iou,]

    i_syn = 0.
    for it in range(1, steps):
        init_t += dt
        if ou_extern is not None:
            iou = iou + (1 - np.exp(-dt / ou_tau)) * (ou_mean - iou) + np.sqrt(
                1 - np.exp(-2 * dt / ou_tau)) * ou_sigma * np.random.randn(1).squeeze()
        else:
            iou = 0.
        main_part = - g_L * (v[it - 1] - E_L)
        iou_all.append(iou)
        C_diff_Vi = main_part + i_syn + iou

        delta_Vi = dt / C * C_diff_Vi
        v_normal = v[it - 1] + delta_Vi

        if init_t <= tr + tref:
            v[it] = V_reset
        else:
            v[it] = v_normal

        if v[it] >= V_th:
            rec_spikes.append(it)
            v[it] = V_th + 10
            tr = init_t
        if exc_spike is not None:
            s_exc1 = s_exc1 * np.exp(-dt / tau_ampa)
            s_exc2 = s_exc2 * np.exp(-dt / tau_nmda)
            if it in spike_e:
                s_exc1 += w_e
                s_exc2 += w_e

        if inh_spike is not None:
            s_inh = s_inh * np.exp(-dt / tau_i)
            if it in spike_i:
                s_inh += w_i
        i_syn = g_ampa * (V_ampa - v[it]) * 1e-3 * s_exc1 + g_nmda * (V_nmda - v[it]) * 1e-3 * s_exc2 + g_i * (
                    V_i - v[it]) * s_inh * 1e-3

    # Get spike times in ms
    rec_spikes = np.array(rec_spikes) * dt
    iou_all = np.array(iou_all)

    return v, rec_spikes, iou_all


def tau_0(g_e1, g_e2, g_i, S_e1, S_e2, S_i):
    return C / (g_L + g_e1 * S_e1 + g_e2 * S_e2 + g_i * S_i)


def V_0(g_e1, g_e2, g_i, S_e1, S_e2, S_i):
    return 1 / (g_L + g_e1 * S_e1 + g_e2 * S_e2 + g_i * S_i) * (
            g_L * V_L + g_e1 * S_e1 * V_e1 + g_e2 * S_e2 * V_e2 + g_i * S_i * V_i + mu_ext)


def sigma_0_square(rho, g_e1, g_e2, g_i, S_e1, S_e2, S_i):
    g_0 = g_L + g_e1 * S_e1 + g_e2 * S_e2 + g_i * S_i
    # V_0_out = V_0(g_e1, g_e2, g_i, S_e1, S_e2, S_i)
    # out = 2 * tau_ext / g_0 * g_e1 ** 2 * omega ** 2 * tau_ext * rho * (V_e1 - V_0_out) ** 2 + 2 * sigma_ext ** 2
    tau_0_out = tau_0(g_e1, g_e2, g_i, S_e1, S_e2, S_i)
    out = sigma_ext / g_0 * math.sqrt(2 * tau_ext / tau_0_out)
    out = out**2
    return out


def system(step=2000, g_e1=1.5e-3, g_e2=0.05e-3, g_i=6e-3, **init_kwargs):
    if init_kwargs:
        S_e1, S_e2, S_i, rho = init_kwargs.values()
    else:
        S_e1 = 0.
        S_e2 = 0.
        S_i = 0.
        rho = 0.1
    S_e1_total = []
    S_e2_total = []
    S_i_total = []
    rho_all = []
    random.seed(100)
    for i in range(step):
        S_e1 = S_e1 * math.exp(-delta_t / tau_e1) + omega * rho + omega * math.sqrt(rho) * random.gauss(0, 1)
        # S_e2 = S_e2 * math.exp(-delta_t / tau_e2) + omega * rho
        S_e2 = 0.
        S_i = S_i * math.exp(-delta_t / tau_i) + omega * rho
        V_0_out = V_0(g_e1, g_e2, g_i, S_e1, S_e2, S_i)
        sigma_0_out = math.sqrt(sigma_0_square(rho, g_e1, g_e2, g_i, S_e1, S_e2, S_i))
        rho = 1 / 2 - 1 / 2 * erf((V_th - V_0_out) / sigma_0_out)
        S_e1_total.append(S_e1)
        S_e2_total.append(S_e2)
        S_i_total.append(S_i)
        rho_all.append(rho)
    return S_e1_total, S_e2_total, S_i_total, rho_all

def basic_test():
    S_e1_total, S_e2_total, S_i_total, rho_all = system(step=3000, g_e1=2e-3, g_e2=0.025e-3, g_i=6e-3, S_e1=2, S_e2=10, S_i=1, rho=0.1)
    T = 1000
    print("rho out", np.array(rho_all[T:]).mean())
    fig, ax = plt.subplots(4, 1, figsize=(5, 3))
    ax = ax.flatten()
    ax[0].plot(S_e1_total[T:], lw=1.)
    ax[0].set_ylabel("s_e1")
    ax[1].plot(S_e2_total[T:], lw=1.)
    ax[1].set_ylabel("s_e2")
    ax[2].plot(S_i_total[T:], lw=1.)
    ax[2].set_ylabel("s_i")
    ax[3].plot(rho_all[T:], lw=1.)
    ax[3].set_ylabel("rho")
    for i in range(4):
        ax[i].spines["right"].set_color(None)
        ax[i].spines["top"].set_color(None)
        ax[i].set_xticks([0, 500, 1000, 1500, 2000])
    fig.tight_layout()
    fig.show()
    # fig.savefig("gating_ou.pdf", dpi=100)
    # fig2 = plt.figure(figsize=(5, 3))
    # fig2.gca().plot(S_e2_total[T+150:T+750])
    # fig2.gca().plot(S_i_total[T+150:T+750])
    # fig2.savefig("gating_ou_clear.pdf", dpi=100)

def plot_approximation():
    mu_ext_all = np.linspace(0.3, 0.6, 30)

    rho_final_all = []
    for i in range(len(mu_ext_all)):
        global mu_ext
        mu_ext = mu_ext_all[i]
        S_e1_total, S_e2_total, S_i_total, rho_all = system(step=3000, g_e1=0.5e-3, g_e2=0.1e-3, g_i=6e-3, S_e1=2, S_e2=10,
                                                            S_i=1, rho=0.1)
        rho_final = np.array(rho_all[-1000:]).mean()
        rho_final_all.append(rho_final)

    rho_final_all = np.array(rho_final_all)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    ax.plot(mu_ext_all, rho_final_all)
    ax.set_xlabel("I_ext")
    ax.set_ylabel("rho")
    fig.show()

def circle_plot():
    ge1 = np.array([2., 1.3]) * 1e-3  # 0.5, 1.3, 1.9,
    ge2 = (2.5 * 1e-3 - ge1) / 20
    fig = plt.figure(figsize=(5, 5))
    ax = fig.gca()
    for a, b in zip(ge1, ge2):
        S_e1_total, S_e2_total, S_i_total, rho_all = system(step=4000, g_e1=a, g_e2=b, g_i=6e-3, S_e1=2,
                                                            S_e2=10,
                                                            S_i=1, rho=0.1)
        ax.plot(S_e1_total[-500:], S_i_total[-500:], label=f"{a:.4f}")

    global tau_i
    tau_i = 20
    for a, b in zip(ge1, ge2):
        S_e1_total, S_e2_total, S_i_total, rho_all = system(step=4000, g_e1=a, g_e2=b, g_i=6e-3, S_e1=2,
                                                            S_e2=10,
                                                            S_i=1, rho=0.1)
        ax.plot(S_e1_total[-500:], S_i_total[-500:], label=f"tau_i{tau_i}")
        break
    ax.set_xlabel("S_e")
    ax.set_ylabel("S_i")
    ax.legend()
    fig.tight_layout()
    # fig.savefig("./circle.pdf", dpi=100)
    fig.show()

if __name__ == '__main__':
    # plot_approximation()
    # basic_test()
    circle_plot()