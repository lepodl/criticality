import itertools
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import powerlaw


def compute_spike_count(X):
    """
    Args:
        X: (np.array): spike train

    Returns:

    """
    return np.sum(X, axis=1)


def compute_avalanche(X: np.array, threshold=None, if_spike=True):
    """Avalanche sizes, durations and interval sizes

        - Set the neural activity =0 if < activity_threshold
        - Slice the array by non-zero value indices
        - Count the number of items in each slices: Duration of avalanches
        - Sum the items in each slices: Size of avalanches
        - Slice the array into zero value indices
        - Count number of items in each slices: Duration of inter avalanche intervals

    Args:
        X (np.array): spike train

        threshold (int, optional): Threshold of number of spikes at each time step. Spike counts below threshold will be set to zero.Defaults to 1.
    Returns:
        spike_count (np.array): Number of spikes at each time step

        avalanche_durations (np.array): Avalanches durations

        avalanche_sizes (np.array): Number of spikes within each avalanche duration

        iai (np.array): Time interval between avalanches
    """
    if if_spike:
        spike_count = np.asarray(compute_spike_count(X))
    else:
        spike_count = X
    if threshold is None:
        threshold = np.median(spike_count) / 2
    spike_count[spike_count < threshold] = 0

    # Avalanche size and duration
    # Get the non zero indices
    aval_idx = np.nonzero(spike_count)[0]

    # Group indices by a consecutiveness
    aval_indices = []
    for k, g in itertools.groupby(enumerate(aval_idx), lambda ix: ix[0] - ix[1]):
        aval_indices.append(list(map(itemgetter(1), g)))

    # Using group indices, pick the correpondning items in the spike_count list
    avalanches = []
    for val in aval_indices:
        avalanches.append(list(spike_count[val]))

    # Avalanche sizes
    avalanche_sizes = [sum(avalanche) for avalanche in avalanches]
    # Avalanche duration
    avalanche_durations = [len(avalanche) for avalanche in avalanches]

    # Inter avalanche intervals

    # Get the indices where spike count =0
    silent_idx = np.where(spike_count == 0)[0]

    silent_indices = []
    # Group indices by consecutiveness
    for k, g in itertools.groupby(enumerate(silent_idx), lambda ix: ix[0] - ix[1]):
        silent_indices.append(list(map(itemgetter(1), g)))
    iai_ = []
    for val in silent_indices:
        iai_.append(list(spike_count[val]))
    # Duration of inter-avalanche intervals
    iai = [len(intervals) for intervals in iai_]

    return spike_count, np.array(avalanche_sizes), np.array(avalanche_durations), np.array(iai)


def compute_kaapa(X, threshold=None, no_Bins=10, tau=1.5):
    """

    Args:
        X: (np.array): spike train
        threshold:
        no_Bins:


    Returns:

    """
    _, aval_size, aval_dur, _ = compute_avalanche(X, threshold, if_spike=False)
    avalanche_size_min = np.min(aval_size)
    avalanche_size_max = np.max(aval_size)
    avs_with_size_threshold = aval_size[
        np.where(np.logical_and(aval_size >= avalanche_size_min, aval_size <= avalanche_size_max))[0]]
    t = np.logspace(np.log10(avalanche_size_min), np.log10(avalanche_size_max), no_Bins)
    theoreticalDistribution = (avalanche_size_max ** (-tau + 1) - avalanche_size_min ** (-tau + 1)) ** (-1) * (t ** (-tau + 1) - avalanche_size_min ** (-tau +1))
    actualDistribution = np.zeros(no_Bins)
    for i in range(no_Bins):
        actualDistribution[i] = (avs_with_size_threshold < t[i]).sum()
    actualDistribution = actualDistribution / len(avs_with_size_threshold)
    kappa = 1 + np.mean(actualDistribution - theoreticalDistribution )
    return kappa


def fit_powerlaw(data):
    """
    fit data to a powerlaw distribution and return fit statistics
    Args:
        data: 1-D np.ndarray

    Returns: ks distance
             alpha
             x_min
             x_max

    """
    fit = powerlaw.Fit(data, discrete=True)
    return fit.alpha, fit.D


def plot_avalanches(X, threshold):
    """
    plot avalanches distribution
    Args:
        X:
        threshold:

    Returns: figure

    """
    _, aval_size, aval_dur, _ = compute_avalanche(X, threshold)
    fig = plt.figure(figsize=(8, 4), dpi=300)
    ax = fig.add_subplot(1, 2, 1)
    fit = powerlaw.Fit(aval_size, xmin=aval_size.min(), xmax=aval_size.max())
    fit.plot_pdf(ax=ax, original_data=False, color="k", lw=1.5)
    info = f"alpha={fit.alpha}\nks={fit.D}"
    ax.text(0.7, 0.85, info, fontsize=8, ha='center', va='center', color='b',
            transform=ax.transAxes)
    ax.set_xlabel("aval size")
    ax.set_ylabel("prob")
    ax = fig.add_subplot(1, 2, 2)
    fit = powerlaw.Fit(aval_dur, xmin=aval_dur.min(), xmax=aval_dur.max())
    fit.plot_pdf(ax=ax, original_data=False, color="k", lw=1.5)
    info = f"alpha={fit.alpha}\nks={fit.D}"
    ax.text(0.7, 0.85, info, fontsize=8, ha='center', va='center', color='b',
            transform=ax.transAxes)
    ax.set_xlabel("aval dur")
    ax.set_ylabel("prob")
    fig.tight_layout()
    return fig


def caculate_DFA(signal, compute_interval, fit_interval, sampling_fre, window_overlap):
    """

    Args:
        signal: np.ndarray, (time_length, num_channel)
        compute_interval: np.ndarray, shape=(2, )
        fit_interval: np.ndarray, shape=(2, )
        sampling_fre:
        window_overlap:

    Returns:

    """
    pass


def total_avalanches(spike, threshold=1):
    aval_size_total = []
    aval_dur_total = []
    for i in range(20):
        seed = np.random.randint(low=0, high=500, size=(200), dtype=np.int32)
        _, aval_size, aval_dur, _ = compute_avalanche(spike[:, seed], threshold)
        aval_size_total.append(aval_size)
        aval_dur_total.append(aval_dur)
    aval_size_total = np.concatenate(aval_size_total, axis=0)
    aval_dur_total = np.concatenate(aval_dur_total, axis=0)
    return aval_size_total, aval_dur_total
