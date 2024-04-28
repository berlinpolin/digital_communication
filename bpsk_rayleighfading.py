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
    the_ber_awgn = np.zeros(n_snr)
    y_hat = np.zeros(N)
    for k in range(n_snr):
        n = 1/np.sqrt(2) * (np.random.randn(N) + 1j * np.random.randn(N))
        h = 1/np.sqrt(2) * (np.random.randn(N) + 1j * np.random.randn(N))
        y = h*x + (10**(-snr[k]/20)) * n
        y_est = y/h
        y_hat[np.real(y_est) > 0] = 1
        y_hat[np.real(y_est) <= 0] = -1
        n_error[k] = np.sum(np.abs(x - y_hat)) / 2

        e_n = 10**(snr[k]/10)
        the_ber[k] = 0.5 * (1 - np.sqrt(e_n/(e_n + 1)))
        the_ber_awgn[k] = 0.5 * math.erfc(np.sqrt(e_n))

    sim_ber = n_error / N

    print(snr)
    print(the_ber)

    plt.figure
    plt.semilogy(snr, sim_ber, label='simulation')
    plt.semilogy(snr, the_ber, label='theoretical result')
    plt.semilogy(snr, the_ber_awgn, label='theoretical result (AWGN)')
    plt.grid('True')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()