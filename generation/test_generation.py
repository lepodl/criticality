# -*- coding: utf-8 -*- 
# @Time : 2022/8/26 10:14 
# @Author : lepold
# @File : test_generation.py

"""
some Testcase to make block, such as small off-line network for seeking appropriate parameter, and
much bigger block for grid search.

specific_gui = np.array([2.13037975e-02, 1.39240506e-04, 1.58227848e-01, 1.89873418e-02])  # for critical in Critical_NN
specific_gui = np.array([0.01837975, 0.00072405, 0.10759494, 0.02180028])  # for critical in Richdynamics_NN
specific_gui = np.array([0.02046835 0.00030633 0.14240506 0.01986639])  # for critical in Richdynamics_NN  # a new critical parameter (in the experiment of reparameter)
specific_gui = np.array([0.01517722, 0.00136456, 0.18987342, 0.01722925])  # for subcritical
specific_gui = np.array([0.01935443, 0.00052911, 0.09493671, 0.02250352]) # for supercritical
"""

import unittest

from generation.make_block import *


# from mpi4py import MPI


class TestBlock(unittest.TestCase):

    def test_local_generate(self, degree=100):
        prob = np.array([[0.8, 0.2], [0.8, 0.2]],
                        dtype=np.float32)
        gui = np.array([0.104, 0.327, 1.25, 0.]) * 1e-3
        kwords = [{'g_ui': gui,
                   'noise_rate': 0.,
                   'tao_ui': (2, 40, 20, 1),
                   'V_ui': (0, 0, -70, 0),
                   'g_Li': 25 * 1e-3,
                   'V_L': -70,
                   'V_th': -50,
                   'V_reset': -55,
                   'C': 0.5,
                   'T_ref': 2,
                   "size": b}
                  for j, b in enumerate([1600, 400])]
        conn = connect_for_multi_sparse_block(prob, kwords, degree=degree, init_min=0.,
                                              init_max=1, prefix=None)
        merge_dti_distributation_block(conn, "../blocks/2000_neurons_tau_20",
                                       MPI_rank=None,
                                       number=1,
                                       dtype="d100_w01",
                                       debug_block_dir=None)

    def _test_local_coupled_block_generate(self, degree=110):
        prob = np.array([[8/11, 2/11, 1/11, 0.], [8/11, 2/11, 1/11, 0.], [1/11, 0., 8/11, 2/11], [1/11, 0., 8/11, 2/11]],
                        dtype=np.float32)
        gui = np.array([1.0, 0.075,  6., 0.0]) * 1e-3
        kwords = [{'g_ui': gui,
                   'noise_rate': 0.,
                   'tao_ui': (2, 40, 10, 1),
                   'V_ui': (0, 0, -70, 0),
                   'g_Li': 25 * 1e-3,
                   'V_L': -70,
                   'V_th': -50,
                   'V_reset': -55,
                   'C': 0.5,
                   'T_ref': 2,
                   "size": b}
                  for j, b in enumerate([1600, 400, 1600, 400])]
        conn = connect_for_multi_sparse_block(prob, kwords, degree=degree, init_min=0.,
                                              init_max=1., prefix=None)
        merge_dti_distributation_block(conn, "../blocks/4000_neurons_coupled",
                                       MPI_rank=None,
                                       number=1,
                                       dtype="d110_w01",
                                       debug_block_dir=None)


if __name__ == "__main__":
    unittest.main()
