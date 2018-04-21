from scipy.ndimage import interpolation
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import argparse

import utils


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-o", "--output", required=False)
    parser.add_argument("-w", "--window", required=True, type=int)
    parser.add_argument("-z", "--zoom", required=True, type=int)
    parser.add_argument('-b', '--bound', nargs='+', required=True, type=int)
    return parser.parse_args(argv)


class Hydrophone():
    def __init__(self, path, window, bound, zoom, output=None):
        self.path = path
        self.bound = bound
        self.zoom = zoom
        self.window = window
        self.output = output

    def open_raw(self):
        return utils.open_raw(self.path)

    def process(self, data):
        C = data[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]
        C = (C - np.mean(C)) / np.std(C)
        signal = []
        window_avr_i = 0
        for i in range(len(C) - self.zoom):
            zoomed = interpolation.zoom(C[i:i + 2, :], self.zoom, order=3)
            cor_array = np.correlate(zoomed[0, :], zoomed[-1, :], "full")
            cor = np.argmax(cor_array)
            cor -= len(cor_array) / 2
            if i > 0:
                cor += signal[-1] + 1 / 2

            if i > self.window:
                window_avr_i += 1
                signal[window_avr_i] -= np.mean(signal[-self.window:])

            signal.append(cor)
        signal = signal[:-self.window]
        return signal

    def plot_graphs(self, data):
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(np.fft.fft(data))[:len(data) // 4])
        plt.savefig(self.output)

    def run(self):
        data = self.process(self.open_raw())
        self.plot_graphs(data)


if __name__ == "__main__":
    args = vars(get_args())
    if not args['output']:
        args['output'] = '../figs/fig_{}.png'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    Hydrophone(**args).run()
