from spectrogram import Spectrogram
import dask as dd

sp = Spectrogram()
d = sp.preprocessing("./sample_data.txt")
sp.graph(d)



dd.Dataframe
