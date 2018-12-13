import numpy as np
import matplotlib.pyplot as plt
from random import gauss
from numpy import sin, pi, arange
from pg_dsp.utils.py_libs.fixp import Quantizer
from scipy.io.wavfile import write
from verif.moving_average_env import moving_average_sim


def add_white_gaussian_noise(signal, magnitude):
    with_noise = []
    gaussian_noise = [gauss(0.0, 1.0) for i in range(len(signal))]
    for i, sample in enumerate(signal):
        with_noise.append(sample + magnitude*gaussian_noise[i])

    return with_noise


def uint_to_int(samples):
    converted_list = []
    for i, sample in enumerate(samples):
        if sample > 2**15:
            converted_list.append(sample - 2**16)
        else:
            converted_list.append(sample)

    return converted_list


def example_synthetic(nsamples,
                      sample_rate_hz=44100,
                      window_size=20):

    t = arange(nsamples) / sample_rate_hz

    x = 0.8*sin(2*pi*100*t)
    x = add_white_gaussian_noise(x, 0.2)

    # quantizer for hardware input preparation
    q = Quantizer(
        round_mode='floor',
        overflow_mode='saturate',
        fix_format=(16, 15))

    # scale input to avoid clipping
    scaled = x/np.max(np.abs(x))

    # prepare input and taps for hardware simulation
    quantized_input = q.quantize(list(scaled))

    # quantized_input = uint_to_int(quantized_input)

    cfg = (q.quantize(1/window_size), window_size)

    wav_input = np.asarray(quantized_input, dtype=np.int16)
    write('moving_average_input.wav', sample_rate_hz, wav_input)

    res = moving_average_sim(din=quantized_input,
                             cfg=cfg,
                             sample_width=16)

    wav_res = np.asarray(res, dtype=np.int16)[:, 0]

    # res = uint_to_int(res)
    quantized_input = uint_to_int(quantized_input)

    x = np.ndarray.tolist(arange(len(quantized_input)))
    y = quantized_input
    plt.plot(x, y, label='input signal')
    x = np.ndarray.tolist(arange(len(wav_res)))
    y = wav_res
    plt.plot(x, y, label='output signal')
    plt.legend()
    plt.show()

    print(f'Result length: {len(res)}')

    write('moving_average_output.wav', sample_rate_hz, wav_res)


if __name__ == "__main__":
    example_synthetic(nsamples=10000, window_size=50)
