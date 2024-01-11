# -*- coding: utf-8 -*- 
# @Time : 2023/3/23 23:05 
# @Author : lepold
# @File : da_model.py

import os
import prettytable as pt
from cuda.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from models.bold_model_pytorch import BOLD
import time
import torch
import numpy as np
import matplotlib.pyplot as mp
import argparse

mp.switch_backend('Agg')


class DA_MODEL:
    def __init__(self, block_dict: dict, bold_dict: dict, steps=800, ensembles=100, time=400, hp_sigma=1.,
                 bold_sigma=1e-6):
        """
        Mainly for the whole brain model consisting of cortical functional column structure and canonical E/I=4:1 structure.

       Parameters
       ----------
       block_name: str
           block name.
       block_dict : dict
           contains the parameters of the block model.
       bold_dict : dict
           contains the parameters of the bold model.
        """
        self.block = block_gpu(block_dict['ip'], block_dict['block_path'], block_dict['delta_t'],
                               route_path=None, force_rebase=False, cortical_size=2)  # !!!!
        self.noise_rate = block_dict['noise_rate']
        self.delta_t = block_dict['delta_t']
        self.bold = BOLD(bold_dict['epsilon'], bold_dict['tao_s'], bold_dict['tao_f'], bold_dict['tao_0'],
                         bold_dict['alpha'], bold_dict['E_0'], bold_dict['V_0'])
        self.ensembles = ensembles
        self.num_populations = int(self.block.total_subblks)
        print("num_populations", self.num_populations)
        self.num_populations_in_one_ensemble = int(self.num_populations / self.ensembles)
        self.num_neurons = int(self.block.total_neurons)
        self.num_neurons_in_one_ensemble = int(self.num_neurons / self.ensembles)
        self.populations_id = self.block.subblk_id.cpu().numpy()
        print("len(populations_id)", len(self.populations_id))
        self.neurons = self.block.neurons_per_subblk
        self.populations_id_per_ensemble = np.split(self.populations_id, self.ensembles)

        self.T = time
        self.steps = steps
        self.hp_sigma = hp_sigma * torch.tensor([0.01, 0.1])
        self.bold_sigma = bold_sigma

    @staticmethod
    def pretty_print(content):
        screen_width = 80
        text_width = len(content)
        box_width = text_width + 6
        left_margin = (screen_width - box_width) // 2
        print()
        print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
        print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
        print(' ' * left_margin + '|' + content + ' ' * (box_width - text_width - 4) + '|')
        print(' ' * left_margin + '|' + ' ' * (text_width + 2) + '|')
        print(' ' * left_margin + '+' + '-' * (text_width + 2) + '+')
        print()


    @staticmethod
    def log(val, low_bound, up_bound, scale=10):
        assert torch.all(torch.le(val, up_bound)) and torch.all(
            torch.ge(val, low_bound)), "In function log, input data error!"
        return scale * (torch.log(val - low_bound) - torch.log(up_bound - val))

    @staticmethod
    def sigmoid(val, low_bound, up_bound, scale=10):
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(val / scale)
        elif isinstance(val, np.ndarray):
            assert np.isfinite(val).all()
            return low_bound + (up_bound - low_bound) * torch.sigmoid(
                torch.from_numpy(val.astype(np.float32)) / scale).numpy()
        else:
            raise ValueError("val type is wrong!")

    @staticmethod
    def torch_2_numpy(u, is_cuda=True):
        assert isinstance(u, torch.Tensor)
        if is_cuda:
            return u.cpu().numpy()
        else:
            return u.numpy()

    @staticmethod
    def show_bold(W, bold, T, path, brain_num):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            print("show_bold" + str(i))
            fig = mp.figure(figsize=(5, 3), dpi=500)
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot(iteration, bold[:T, i], 'r-')
            ax1.plot(iteration, np.mean(W[:T, :, i, -1], axis=1), 'b-')
            mp.fill_between(iteration, np.mean(W[:T, :, i, -1], axis=1) -
                            np.std(W[:T, :, i, -1], axis=1), np.mean(W[:T, :, i, -1], axis=1)
                            + np.std(W[:T, :, i, -1], axis=1), color='b', alpha=0.2)
            mp.ylim((0.0, 0.08))
            ax1.set(xlabel='observation time/800ms', ylabel='bold', title=str(i + 1))
            mp.savefig(os.path.join(path, "bold" + str(i) + ".png"), bbox_inches='tight', pad_inches=0)
            mp.close(fig)
        return None

    @staticmethod
    def show_hp(hp, T, path, brain_num, hp_num, hp_real=None):
        iteration = [i for i in range(T)]
        for i in range(brain_num):
            for j in range(hp_num):
                print("show_hp", i, 'and', j)
                fig = mp.figure(figsize=(5, 3), dpi=500)
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.plot(iteration, np.mean(hp[:T, :, i, j], axis=1), 'b-')
                if hp_real is None:
                    pass
                else:
                    ax1.plot(iteration, np.tile(hp_real[j], T), 'r-')
                mp.fill_between(iteration, np.mean(hp[:T, :, i, j], axis=1) -
                                np.sqrt(np.var(hp[:T, :, i, j], axis=1)), np.mean(hp[:T, :, i, j], axis=1)
                                + np.sqrt(np.var(hp[:T, :, i, j], axis=1)), color='b', alpha=0.2)
                ax1.set(xlabel='observation time/800ms', ylabel='hyper parameter')
                mp.savefig(os.path.join(path, "hp" + str(i) + "_" + str(j) + ".png"), bbox_inches='tight', pad_inches=0)
                mp.close(fig)
        return None

    def initial_model(self):
        """
        initialize the block model, and then determine the random walk range of hyper parameter,
        -------
        """
        raise NotImplementedError

    def evolve(self, steps=800):
        """
        evolve the block model and obtain prediction observation,
        here we apply the MC method to evolve samples drawn from initial Gaussian distribution.
        -------

        """
        raise NotImplementedError

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        use kalman filter to filter the latent state.
        -------

        """
        raise NotImplementedError


class DA_Small_block_demo(DA_MODEL):
    def __init__(self, block_dict: dict, bold_dict: dict, steps, ensembles, time, hp_sigma,
                 bold_sigma):
        super().__init__(block_dict, bold_dict, steps, ensembles, time, hp_sigma, bold_sigma)
        self.device = "cuda:0"
        aal_region = np.arange(1, dtype=np.int8)
        self.num_voxel_in_one_ensemble = len(aal_region)
        assert self.num_populations_in_one_ensemble == self.num_voxel_in_one_ensemble * 2
        assert self.populations_id.max() == self.num_populations_in_one_ensemble * self.ensembles - 1, "population_id is not correct!"
        self.num_voxel = int(self.num_populations / 2)
        neurons_per_voxel = self.block.neurons_per_subblk.cpu().numpy()

        self.neurons_per_voxel, _ = np.histogram(self.populations_id, weights=neurons_per_voxel, bins=self.num_voxel,
                                            range=(0, self.num_voxel * 2))

        self.hp_num = None
        self.up_bound = None
        self.low_bound = None
        self.hp = None
        self.hp_log = None
        # self.H = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float32, device="cuda:0")
        # self.H = self.H.T
        # self.lmabda = 1.
        # self.H[:, 0] = self.H[:, 0] * self.lmabda
        # self.H = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 1.]],
        #                       dtype=torch.float32, device="cuda:0")
        # self.H = self.H.T

    def __str__(self):
        print("DA FOR REST WHOLE BRAIN")

    @staticmethod
    def random_walk_range(voxels, up_times, low_times):
        temp_up = np.tile(up_times, (voxels, 1))
        temp_low = np.tile(low_times, (voxels, 1))
        return temp_up.reshape((voxels, -1)), temp_low.reshape((voxels, -1))

    def update(self, param_here):
        nmda = (2.5 - param_here[:, :, [0]]) / 20
        param = torch.cat([param_here, nmda], dim=2)
        for i, ind in enumerate(np.array([10, 11])):
            population_info = np.stack(np.meshgrid(self.populations_id, ind, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            temp_param = param[:, :, i].reshape(-1)
            temp_param = torch.from_numpy(temp_param).cuda()
            temp_param = torch.repeat_interleave(temp_param, 2)
            self.block.assign_property_by_subblk(population_info, temp_param)

    def initial_model(self):
        start = time.time()
        self.hp_num = 1
        self.up_bound, self.low_bound = np.array([[2.5 * 1e-3]]), np.array([[0.5 * 1e-3]])
        ampa = np.linspace(0.5, 2.5, 20) * 1e-3
        self.hp = ampa[:, np.newaxis, np.newaxis]
        print(f"in initial_model, hp shape {self.hp.shape}")
        self.hp = torch.from_numpy(self.hp.astype(np.float32)).cuda()

        self.up_bound, self.low_bound = torch.from_numpy(self.up_bound.astype(np.float32)).cuda(), torch.from_numpy(
            self.low_bound.astype(np.float32)).cuda()
        self.update(self.hp)
        print(f"=================Initial DA MODEL done! cost time {time.time() - start:.2f}==========================")

    def filter(self, w_hat, bold_y_t, rate=0.5):
        """
        distributed ensemble kalman filter. for single voxel, it modified both by its own observation and also
        by other's observation with distributed rate.

        Parameters
        ----------
        w_hat  :  store the state matrix, shape=(ensembles, voxels, states)
        bold_y_t : shape=(voxels)
        rate : distributed rate
        """
        ensembles, voxels, total_state_num = w_hat.shape # ensemble, brain_n, hp_num+act+hemodynamic(total state num)
        assert self.ensembles == ensembles
        assert total_state_num == self.hp_num + 6  # +6 --> +7
        w = w_hat.clone()
        w_mean = torch.mean(w_hat, dim=0, keepdim=True)
        w_diff = w_hat - w_mean
        observation = torch.matmul(w_hat, self.H)
        n = observation.shape[-1]
        observation_diff = observation - torch.mean(observation, dim=0, keepdim=True)
        w_cxx = torch.sum(torch.einsum('ijk,ijm->ijkm', observation_diff, observation_diff), dim=0) / (
                    self.ensembles - 1) + torch.eye(n, dtype=torch.float32, device="cuda:0") * self.bold_sigma
        w_cxx_inv = torch.pinverse(w_cxx)
        print("w_cxx_inv", w_cxx_inv[0, :, :].data)
        temp = torch.einsum('ijk,jkm->ijm', observation_diff, w_cxx_inv) / (
                self.ensembles - 1)
        model_noise = self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, voxels, n)).cuda()
        bold_with_noise = bold_y_t + model_noise
        kalman = torch.einsum('ijk,ijm->ijkm', w_diff, temp)
        kalman = torch.sum(kalman, dim=0)
        print("kalman", kalman[0, :, :].data)
        data_diff = bold_with_noise - observation
        w += rate * torch.einsum('jkm,ijm->ijk', kalman, data_diff)
        return w

    def evolve(self, steps=800):
        print("evolve")
        start = time.time()
        out = None
        steps = steps if steps is not None else self.steps
        act_all = np.zeros((steps * 10, self.num_populations))
        count = 0
        for act in self.block.run(steps * 10, freqs=True, vmean=False, sample_for_show=False, iou=True):
            act = act.cpu().numpy()
            act_all[count] = act
            count += 1
        act_all = act_all.reshape((steps, 10, -1))
        act_all = np.sum(act_all, axis=1)
        for idxx in range(steps):
            act = act_all[idxx]
            active, _ = np.histogram(self.populations_id, weights=act, bins=self.num_voxel,
                                     range=(0, self.num_voxel * 2))
            active = (active / self.neurons_per_voxel).reshape(-1)
            active = torch.from_numpy(active.astype(np.float32)).cuda()
            out = self.bold.run(torch.max(active, torch.tensor([1e-05]).type_as(active)))

        print(
            f'active mean: {active.mean().item():.3f},  {active.min().item():.3f} ------> {active.max().item():.3f}')

        bold = torch.stack(
            [self.bold.s, self.bold.q, self.bold.v, self.bold.f_in, out])

        # print("cortical max bold_State: s, q, v, f_in, bold ", bold1.max(1)[0].data)
        print("bold range:", bold[-1].min().data, "------>>", bold[-1].max().data)
        w = torch.cat((self.hp_log, active.reshape([self.ensembles, -1, 1]),
                       bold.T.reshape([self.ensembles, -1, 5])), dim=2)  #  , last_element
        print(f'End evolving, totally cost time: {time.time() - start:.2f}')
        return w

    def run(self, bold_path="whole_brain_voxel_info.npz", write_path="./"):
        total_start = time.time()

        tb = pt.PrettyTable()
        tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
        tb.add_row([1, "name", "da_small_block", "ensembles", self.ensembles])
        tb.add_row([2, "total_populations", self.num_populations, "populations_per_ensemble",
                    self.num_populations_in_one_ensemble])
        tb.add_row([3, "total_neurons", self.num_neurons, "neurons_per_ensemble", self.num_neurons_in_one_ensemble])
        tb.add_row([4, "voxels_per_ensemble", self.num_voxel_in_one_ensemble, "populations_per_voxel", "2"])
        tb.add_row([5, "total_time", self.T, "steps", self.steps])
        tb.add_row([6, "hp_sigma", self.hp_sigma, "bold_sigma", self.bold_sigma])
        tb.add_row([7, "noise_rate(Hz)", self.noise_rate, "bold_range", "None"])
        # tb.add_row([8, "upbound", "(0.6, 0.6)", "lowbound", "(0., 0.)"])
        print(tb)
        self.pretty_print("Init Model")
        self.initial_model()
        self.hp_log = self.log(self.hp, self.low_bound, self.up_bound)
        # self.hp_log = self.hp
        self.pretty_print("Pre-Run")
        _ = self.evolve(steps=800)
        _ = self.evolve(steps=800)
        _ = self.evolve(steps=800)
        w = self.evolve(steps=800)

        bold_y = np.load(bold_path)
        bold_y = bold_y[:, :self.num_voxel_in_one_ensemble]
        bold_y = torch.from_numpy(bold_y.astype(np.float32)).cuda()
        w_save = [self.torch_2_numpy(w, is_cuda=True)]

        self.pretty_print("Begin DA")
        for t in range(self.T - 1):
            print("PROCESSING || %d" % t)
            bold_y_t = bold_y[t].reshape([self.num_voxel_in_one_ensemble, 1])
            signal_t = bold_y_t
            # signal_t = torch.cat([bold_y_t, torch.tensor([[-0.043]], dtype=torch.float32, device="cuda:0") * self.lmabda], dim=1)
            self.hp_log = w[:, :, :self.hp_num] + (self.hp_sigma ** 0.5 * torch.normal(0, 1, size=(self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num))).cuda()
            self.hp = self.sigmoid(self.hp_log, self.low_bound, self.up_bound)
            # self.hp = self.hp_log
            # self.hp = 2 * self.hp_star / 3 + 1 / 3 * self.hp
            # print("self.hp.dtype", self.hp.dtype)
            # self.hp_star = self.hp
            print("hp in one sample: ", self.hp[int(self.ensembles / 2), 0, :self.hp_num].data)
            # print("self.hp.shape ", self.hp.shape)
            self.update(self.hp)

            w_hat = self.evolve()
            w_hat[:, :, (self.hp_num + 1):(self.hp_num + 6)] += (self.bold_sigma ** 0.5 * torch.normal(0, 1, size=(
                self.ensembles, self.num_voxel_in_one_ensemble, 5))).type_as(w_hat)

            w = self.filter(w_hat, signal_t, rate=1.)
            self.bold.state_update(
                w[:, :self.num_voxel_in_one_ensemble, (self.hp_num + 1):(self.hp_num + 5)])
            w_save.append(self.torch_2_numpy(w_hat, is_cuda=True))

        print("\n                 END DA               \n")
        np.save(os.path.join(write_path, "W.npy"), w_save)
        del w_hat, w
        path = write_path + '/show/'
        os.makedirs(path, exist_ok=True)

        w_save = np.array(w_save)
        # self.show_bold(w_save, self.torch_2_numpy(bold_y, is_cuda=True), self.T, path, 1)  # !!!
        self.show_bold(w_save[:, :, :, :-1], self.torch_2_numpy(bold_y, is_cuda=True), self.T, path, 1)  # !!!
        hp_save = self.sigmoid(
            w_save[:, :, :, :self.hp_num].reshape(self.T * self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.torch_2_numpy(self.low_bound), self.torch_2_numpy(self.up_bound))
        # hp_save = w_save[:, :, :, :self.hp_num].reshape(self.T * self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num)
        hp = hp_save.reshape(self.T, self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num)
        hp = hp.mean(axis=1)
        np.save(os.path.join(write_path, "hp.npy"), hp)
        self.show_hp(
            hp_save.reshape(self.T, self.ensembles, self.num_voxel_in_one_ensemble, self.hp_num),
            self.T, path, 1, self.hp_num)
        self.block.shutdown()
        print("\n\n Totally cost time:\t", time.time() - total_start)
        print("=================================================\n")


if __name__ == '__main__':
    block_dict = {"ip": "10.5.4.1:50051",
                  "block_path": "./",
                  "noise_rate": 3,
                  "delta_t": 0.1,
                  "print_stat": False,
                  "froce_rebase": True}
    bold_dict = {"epsilon": 200,
                 "tao_s": 0.8,
                 "tao_f": 0.4,
                 "tao_0": 1,
                 "alpha": 0.2,
                 "E_0": 0.8,
                 "V_0": 0.02}

    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--ip", type=str, default="10.5.4.1:50051")
    parser.add_argument("--print_stat", type=bool, default=False)
    parser.add_argument("--force_rebase", type=bool, default=True)
    parser.add_argument("--block_path", type=str,
                        default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/single")
    parser.add_argument("--write_path", type=str,
                        default="/public/home/ssct004t/project/wenyong36/dti_voxel_outlier_10m/dti_n4_d100/")
    parser.add_argument("--T", type=int, default=450)
    parser.add_argument("--noise_rate", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--hp_sigma", type=float, default=0.1)
    parser.add_argument("--bold_sigma", type=float, default=1e-6)
    parser.add_argument("--ensembles", type=int, default=100)

    args = parser.parse_args()
    block_dict.update(
        {"ip": args.ip, "block_path": args.block_path, "noise_rate": args.noise_rate, "print_stat": args.print_stat,
         "force_rebase": args.force_rebase})
    bold_path = "/public/home/ssct004t/project/zenglb/Richdynamics_NN/da/da_results/critical_simulation/bold.npy"

    da_rest = DA_Small_block_demo(block_dict, bold_dict, steps=args.steps,
                                  ensembles=args.ensembles, time=args.T, hp_sigma=args.hp_sigma,
                                  bold_sigma=args.bold_sigma)
    os.makedirs(args.write_path, exist_ok=True)
    da_rest.run(bold_path=bold_path, write_path=args.write_path)
