"""
Based on MALTLAB code provided at https://dsplog.com/2008/08/26/ofdm-rayleigh-channel-ber-bpsk/
"""
import numpy as np
import matplotlib.pyplot as plt


def main():
    n_fft = 64
    n_dsc = 52  # number of data subcarriers
    n_bit_per_symbol = 52
    sz_cp = 16
    sz_sym = n_fft + sz_cp
    n_sym = 10**4  # number of symbols
    bnr = np.arange(0, 36, dtype=int)  # bit to noise ratio
    n_bnr = len(bnr)

    # converting to symbol to noise ratio
    snr = bnr + 10 * np.log10(n_dsc / n_fft) + 10 * np.log10(n_fft / sz_sym)

    n_error = np.zeros(n_bnr)
    for k in range(n_bnr):
        ip_bit = np.random.rand(n_bit_per_symbol * n_sym) > 0.5  # random 1's and 0's
        ip_mod = 2 * ip_bit - 1  # BPSK modulation 0 --> -1, 1 --> +1
        ip_mod = ip_mod.reshape(
            n_sym, n_bit_per_symbol
        )  # grouping into multiple symbols

        # assigning modulated symbols to subcarriers from [-26 to -1, +1 to +26]
        xf = np.zeros([n_sym, n_fft])
        xf[:, 6 : 6 + n_bit_per_symbol // 2] = ip_mod[:, : n_bit_per_symbol // 2]
        xf[:, 7 + n_bit_per_symbol // 2 : 7 + n_bit_per_symbol] = ip_mod[
            :, n_bit_per_symbol // 2 :
        ]

        # taking FFT, the term (n_fft/sqrt(n_dsc)) is for normalizing the power of transmit symbol to 1
        xt = (n_fft / np.sqrt(n_dsc)) * np.fft.ifft(np.fft.fftshift(xf, axes=1))

        # appending cylic prefix
        if sz_cp > 0:
            xt = np.append(xt[:, -sz_cp:], xt, axis=1)

        # multipath channel
        n_tap = 10
        ht = (
            1
            / np.sqrt(2)
            * 1
            / np.sqrt(n_tap)
            * (np.random.randn(n_sym, n_tap) + 1j * np.random.randn(n_sym, n_tap))
        )

        # computing and storing the frequency response of the channel, for use at recevier
        hf = np.fft.fftshift(np.fft.fft(ht, n=n_fft), axes=1)

        # convolution of each symbol with the random channel
        xht = np.zeros([n_sym, sz_sym + n_tap - 1], dtype=complex)
        for l in range(hf.shape[0]):
            xht[l, :] = np.convolve(ht[l, :], xt[l, :])

        xht = xht.flatten()

        # Gaussian noise of unit variance, 0 mean
        nt = (1 / np.sqrt(2)) * (
            np.random.randn(n_sym * (sz_sym + n_tap - 1))
            + 1j * np.random.randn(n_sym * (sz_sym + n_tap - 1))
        )

        # adding noise, the term sqrt(sz_sym/n_fft) is to account for the wasted energy due to cyclic prefix
        yt = np.sqrt(sz_sym / n_fft) * xht + 10 ** (-snr[k] / 20) * nt

        # receiver
        yt = yt.reshape(
            n_sym, sz_sym + n_tap - 1
        )  # formatting the received vector into symbols
        yt = yt[:, sz_cp:sz_sym]  # removing cyclic prefix

        # converting to frequency domain
        yf = (np.sqrt(n_dsc) / n_fft) * np.fft.fftshift(np.fft.fft(yt), axes=1)

        # equalization by the known channel frequency response
        yf = yf / hf

        # extracting the required data subcarriers
        y_mod = np.zeros_like(ip_mod, dtype=complex)
        y_mod[:, : n_bit_per_symbol // 2] = yf[:, 6 : 6 + n_bit_per_symbol // 2]
        y_mod[:, n_bit_per_symbol // 2 :] = yf[
            :, 7 + n_bit_per_symbol // 2 : 7 + n_bit_per_symbol
        ]

        # BPSK demodulation
        # +ve value --> 1, -ve value --> -1
        ip_mod_hat = 2 * np.floor(np.real(y_mod / 2)) + 1
        ip_mod_hat[ip_mod_hat > 1] = +1
        ip_mod_hat[ip_mod_hat < -1] = -1

        # converting modulated values into bits
        ip_bit_hat = (ip_mod_hat + 1) / 2
        ip_bit_hat = ip_bit_hat.flatten()

        # counting the errors
        n_error[k] = np.sum(np.abs(ip_bit - ip_bit_hat))

    sim_ber = n_error / (n_sym * n_bit_per_symbol)
    bnr_lin = 10 ** (bnr / 10)
    theory_ber = 0.5 * (1 - np.sqrt(bnr_lin / (bnr_lin + 1)))

    plt.figure()
    plt.semilogy(bnr, theory_ber, "bs-", linewidth=2, label="Rayleigh-Theory")
    plt.semilogy(bnr, sim_ber, "mx-", linewidth=2, label="Rayleigh-Simulation")
    plt.xlim(0, 35)
    plt.ylim(10 ** (-5), 1)
    plt.grid(True)
    plt.xlabel("Eb/No, dB")
    plt.ylabel("Bit Error Rate")
    plt.title("BER for BPSK using OFDM in a 10-tap Rayleigh channel")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
