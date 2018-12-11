import numpy as np
from numpy import sin, pi, arange
from pg_dsp.utils.py_libs.fixp import Quantizer
from scipy.io.wavfile import write
from verif.moving_average_env import moving_average_sim


def example_synthetic(nsamples,
                      sample_rate_hz=44100):

    t = arange(nsamples) / sample_rate_hz
    x = 0.8*sin(2*pi*100*t) \
        + 0.2*sin(2*pi*3000*t)

    # quantizer for hardware input preparation
    q = Quantizer(
        round_mode='floor',
        overflow_mode='saturate',
        fix_format=(16, 15))

    # scale input to avoid clipping
    scaled = x/np.max(np.abs(x))

    # prepare input and taps for hardware simulation
    quantized_input = q.quantize(list(scaled))

    cfg = (0.2, 5)

    wav_input = np.asarray(quantized_input, dtype=np.int16)
    write('moving_average_input.wav', sample_rate_hz, wav_input)

    res = moving_average_sim(din=quantized_input,
                             cfg=cfg,
                             sample_width=16)

    wav_res = np.asarray(res, dtype=np.int16)[:, 0]

    print(f'Result length: {len(res)}')

    write('moving_average_output.wav', sample_rate_hz, wav_res)


if __name__ == "__main__":
    example_synthetic(nsamples=50000)
