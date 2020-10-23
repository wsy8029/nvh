import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class Spectrogram(object):
    def __init__(self):
        pass
        
    def spec_array(self, arr):
        plt.rcParams["figure.figsize"] = (2.24,2.24)
        plt.axis('off') # "invisable" axis in plot
        plt.xticks([]), plt.yticks([])
        plt.use_sticky_edges = True
        plt.margins(0)
        plt.specgram(list(arr), NFFT=10000, Fs=10, noverlap=5, detrend='mean', mode='psd')
        fig = plt.figure(1, tight_layout=True)
        fig.canvas.draw()
        fig.tight_layout(pad=0)
    #     plt.close()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape((3,) + fig.canvas.get_width_height()[::-1])
    #     return np.array(fig.canvas.renderer._renderer)
        return data



def main():
    sp = Spectrogram()

    datas = []
    file = "./sample_data.txt"
    with open(file, "r") as f:
        header = f.readline()
        while 1:
            line = f.readline()
            if not line:
                break
            tmp = line.strip().split('\t')
            freq = list(map(float, tmp[4:]))
            label = int(tmp[0])
    #         label = tmp[0

            datas.append([freq,label])

    # sp.spectrogram(d)
    spar = sp.spec_array(datas[0][0])
    print(spar.shape)



if __name__ == "__main__":
    main()
