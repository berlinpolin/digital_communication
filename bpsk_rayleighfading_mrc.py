import numpy as np
import math
import matplotlib.pyplot as plt

def main():
    N = 1000000
    snr_db = np.arange(-2, 10, dtype=int)
    n_snr = len(snr_db)
    n_rx = np.arange(1, 3, dtype=int)
    # n_rx = 1
    len_n_rx = len(n_rx)
    s = np.random.rand(N) > 0.5
    x = 2 * s - 1

    n_error = np.zeros([len_n_rx, n_snr])
    for j in range(len_n_rx):
        y_hat = np.zeros(N)
        for k in range(n_snr):
            n = 1/np.sqrt(2) * (np.random.randn(n_rx[j], N) + 1j * np.random.randn(n_rx[j], N))
            h = 1/np.sqrt(2) * (np.random.randn(n_rx[j], N) + 1j * np.random.randn(n_rx[j], N))
            xd = np.kron(np.expand_dims(np.ones(n_rx[j]), axis=1), np.expand_dims(x, axis=0))
            y = h * xd + (10**(-snr_db[k]/20)) * n
            y_est = np.sum(np.conj(h)*y, 0) / np.sum(np.conj(h) * h, 0)
            y_hat[np.real(y_est) > 0] = 1
            y_hat[np.real(y_est) <= 0] = -1
            n_error[j, k] = np.sum(np.abs(x - y_hat)) / 2
    
    sim_ber = n_error / N
    snr = 10**(snr_db/10)
    lambda_snr = np.sqrt(snr/(snr + 1))
    the_ber_n1 = 0.5 * (1 - lambda_snr)
    # the_ber_n2 = 3/(4*(snr**2))
    the_ber_n2 = (((1-lambda_snr)/2)**2) * (1+2*(((1+lambda_snr)/2)))
    # p = 1/2 - 1/2*((1+1/snr)**(-1/2))
    # the_ber_n2 = (p**2)*(1+2*(1-p))
    

    print(sim_ber[0,:])
    print(the_ber_n1)
    print(sim_ber[1,:])
    print(the_ber_n2)
    plt.figure
    plt.semilogy(snr, sim_ber[0,:], 'b', label='n_rx = 1 (simulation)')
    plt.semilogy(snr, sim_ber[1,:], 'g', label='n_rx = 2 (simulation)')
    plt.semilogy(snr, the_ber_n1, 'r--', label='n_rx = 1 (theoretical)')
    plt.semilogy(snr, the_ber_n2, 'y--', label='n_rx = 2 (theoretical)')
    plt.grid('True')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()