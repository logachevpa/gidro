import numpy as np
from os import listdir
from os.path import isfile, join
import re
import scipy.ndimage as ndimage
import codecs
from PIL import Image
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib

params = {
    'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
    'axes.titlesize': 8,
    'font.size': 8,
    'legend.fontsize': 8, 
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.figsize': [20, 10],
    'font.family': 'serif'
}
matplotlib.rcParams.update(params)

def open_raw(path):
    with open(path, 'rb') as f:
        x = f.read()
        a = np.fromstring(x, dtype=np.uint16)
        header = a[:298]
        a = a[298:]
        width = header[10]
        length = int(a.size // width)
        b = a.reshape((length, width ))
        return b