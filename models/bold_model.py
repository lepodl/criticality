import numpy as np

class BOLD:
    def __init__(self, epsilon, tao_s, tao_f, tao_0, alpha, E_0, V_0,
                 delta_t=1e-3, init_f_in=None, init_s=None, init_v=None, init_q=None):
        self.epsilon = epsilon
        self.tao_s = tao_s
        self.tao_f = tao_f
        self.tao_0 = tao_0
        self.E_0 = E_0
        self.V_0 = V_0
        self.delta_t = delta_t
        self.div_alpha = 1 / alpha
        self.f_in = init_f_in
        self.s = init_s
        self.v = init_v
        self.q = init_q

    def update(self, f_str, df):
        f = self.__getattribute__(f_str)
        if f is None:
            self.__setattr__(f_str, df * self.delta_t)
        else:
            f += df * self.delta_t

    def run(self, u):
        assert isinstance(u, np.ndarray)
        if self.s is None:
            self.s = np.zeros_like(u)
        if self.q is None:
            self.q = np.zeros_like(u)
        if self.v is None:
            self.v = np.ones_like(u)
        if self.f_in is None:
            self.f_in = np.ones_like(u)
        d_s = self.epsilon * u - self.s/self.tao_s - (self.f_in-1)/self.tao_f
        q_part = np.where(self.f_in > 0, 1 - (1-self.E_0)**(1/self.f_in), np.ones_like(self.f_in))
        self.update('q', (self.f_in * q_part/self.E_0 - self.q * self.v ** (self.div_alpha - 1))/self.tao_0)
        self.update('v', (self.f_in - self.v ** self.div_alpha)/self.tao_0)
        self.update('f_in', self.s)
        self.f_in = np.where(self.f_in>0, self.f_in, np.zeros_like(self.f_in))
        self.update('s', d_s)

        out = self.V_0 * (7 * self.E_0 * (1 - self.q) + 2 * (1 - self.q / self.v) + (2 * self.E_0 - 0.2) * (1 - self.v))
        return np.stack([self.s, self.q, self.v, self.f_in, out])
