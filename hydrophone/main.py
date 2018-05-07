from scipy.ndimage import interpolation
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LinearRegression

import utils


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="")
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

    def crop(self, data):
         return data[self.bound[0]:self.bound[1], self.bound[2]:self.bound[3]]
        
    def process(self, data):
        out = []
        data = (data - np.mean(data)) / np.std(data)
        for i in range(len(data) - self.zoom):
            zoomed = interpolation.zoom(data[i:i + 2, :], self.zoom, order=3)
            cor_array = np.correlate(zoomed[0, :], zoomed[-1, :], "same")
            cor_array = minmax_scale(cor_array)
            cor_array = cor_array[len(cor_array)//4:-len(cor_array)//4]
            cor = np.sum(np.arange(len(cor_array)) * cor_array) / np.sum(cor_array)
            cor -= len(cor_array) / 2
            if i > 0:
                cor += out[-1] + 1/2
            out.append(cor)
        return out

    def unbias_by_regression(self, data):
        x = np.arange(len(data))
        reg = LinearRegression()
        reg.fit(np.transpose([x]), data)
        return data - reg.coef_ * x - np.mean(data)

    def unbias_by_regression_chunked(self, data):
        out = []
        for chunk in np.array_split(np.array(data), self.window):
            out.extend(self.unbias_by_regression(chunk))
        return out

    def unbias_by_moving_avr(self, data):
        cumsum = np.cumsum(np.insert(data, 0, 0)) 
        moving_avr = (cumsum[self.window:] - cumsum[:-self.window]) / float(self.window)
        return data[:-self.window+1] - moving_avr

    def plot_graphs(self, data):
        plt.subplot(2, 1, 1)
        plt.plot(data)
        plt.subplot(2, 1, 2)
        plt.plot(np.abs(np.fft.fft(data))[2:len(data) // 4])
        plt.savefig(self.output)

    def run(self):
        data = self.open_raw()
        data = self.crop(data)
        data = self.process(data)
        data = self.unbias_by_moving_avr(data)
        self.plot_graphs(data)
        return data


if __name__ == "__main__":
    args = vars(get_args())
    if not args['output']:
        args['output'] = '../figs/fig_{}.png'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    Hydrophone(**args).run()
