import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    n_sample = 10**3
    snr = 25
    n_rx = np.arange(1, 21, dtype=int)
    len_n_rx = len(n_rx)
    s = np.random.rand(n_sample) > 0.5
    x = 2 * s - 1

    eff_eb_ovr_no_sim = np.zeros(len_n_rx)
    eff_eb_ovr_no_the = np.zeros(len_n_rx)
    for k in range(len_n_rx):
        n = 1/np.sqrt(2) * (np.random.randn(n_rx[k], n_sample) + 1j * np.random.randn(n_rx[k], n_sample))
        h = 1/np.sqrt(2) * (np.random.randn(n_rx[k], n_sample) + 1j * np.random.randn(n_rx[k], n_sample))
        xd = np.kron(np.expand_dims(np.ones(n_rx[k]), axis=1), np.expand_dims(x, axis=0))
        y = h * xd + (10**(-snr/20)) * n
        y_hat = np.sum(np.conj(h)*y, 0)
        eff_eb_ovr_no_sim[k] = np.mean(np.abs(y_hat))
        eff_eb_ovr_no_the[k] = n_rx[k]
    
    print(eff_eb_ovr_no_sim)

    plt.figure
    plt.plot(n_rx, 10 * np.log10(eff_eb_ovr_no_sim), label='simulation')
    plt.plot(n_rx, 10 * np.log10(eff_eb_ovr_no_the), label='theoretical result')
    plt.xlim(1, 20)
    plt.ylim(0, 16)
    plt.grid('True')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()