import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    N = 1000000
    snr = np.arange(-2, 10, dtype=int)
    n_snr = len(snr)
    s = np.random.rand(N) > 0.5
    x = 2 * s - 1

    n_error = np.zeros(n_snr)
    the_ber = np.zeros(n_snr)
    y_hat = np.zeros(N)
    for k in range(n_snr):
        # n = 1/np.sqrt(2) * (np.random.randn(N) + 1j * np.random.randn(N))
        n = np.random.randn(N)
        y = x + (10**(-snr[k]/20)) * n
        y_hat[np.real(y) > 0] = 1
        y_hat[np.real(y) <= 0] = -1
        n_error[k] = np.sum(np.abs(x - y_hat)) / 2
        # the_ber[k] = 0.5 * math.erfc(np.sqrt(10**(snr[k]/10)))
        the_ber[k] = 0.5 * math.erfc(np.sqrt((10**(snr[k]/10)/2)))
    
    sim_ber = n_error / N

    plt.figure
    plt.semilogy(snr, sim_ber, label='simulation')
    plt.semilogy(snr, the_ber, label='theoretical result')
    plt.grid('True')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()