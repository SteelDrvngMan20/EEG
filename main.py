# -*- coding: utf-8 -*-
"""
Created on Feb 2021

@author: Natalia
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io.wavfile as waves
import scipy
import csv
import pandas as pd

file = 'JohnHenry_BYB_Recording_2022-01-07_10.18.30.wav'
fs, data = waves.read(file)

length_data=np.shape(data)
length_new=length_data[0]*0.05
ld_int=int(length_new)
from scipy import signal
data_new=signal.resample(data,ld_int)



plt.figure('Spectrogram')
d, f, t, im = plt.specgram(data_new, NFFT= 256, Fs=500, noverlap=250)
plt.ylim(0,90)
plt.colorbar(label= "Power/Frequency")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.savefig("firstplot.png")

matrixf=np.array(f).T
np.savetxt('Frequencies.csv', matrixf)
df = pd.read_csv("Frequencies.csv", header=None, index_col=None)
df.columns = ["Frequencies"]
df.to_csv("Frequencies.csv", index=False)

position_vector=[]
length_f=np.shape(f)
l_row_f=length_f[0]
for i in range(0, l_row_f):
    if f[i]>=7 and f[i]<=12:
        position_vector.append(i)

length_d=np.shape(d)
l_col_d=length_d[1]
AlphaRange=[]
for i in range(0,l_col_d):
    AlphaRange.append(np.mean(d[position_vector[0]:max(position_vector)+1,i]))


def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

plt.figure('AlphaRange')
y=smoothTriangle(AlphaRange, 100)
plt.plot(t, y)
plt.xlabel('Time [s]')
plt.xlim(0,max(t))
plt.savefig("secondplot.png")


