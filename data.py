import csv
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import glob
import os,sys
import matplotlib.colors as mcolors


import plotly.graph_objects as go

import numpy as np
import pandas as pd
import scipy

#from scipy import signal

sns.set_theme(style="white", palette=None)


def read_and_plot():
    y_collapsed = pd.read_pickle(r'pos_collapsed.pickle')
    slope_collapsed = pd.read_pickle(r'slope_collapsed.pickle')
    #print(y_collapsed)
    #print(slope_collapsed)
    to_pandas_pos = {}
    to_pandas_slopes = {}
    # print(f"filtered lines 0 = {filtered_lines[0,0]}")
    entries = 0
    for i, xpos in enumerate(y_collapsed):
        hit = False
        if entries < 3:
            to_pandas_pos[entries] = [(y_collapsed[i],i)]
            to_pandas_slopes[entries] = [(slope_collapsed[i],i)]
            entries+=1
            hit = True
        else:
            for entry in to_pandas_pos:
                if entry in to_pandas_pos and abs(y_collapsed[i] - to_pandas_pos[entry][-1][0]) < 25 and (i-to_pandas_pos[entry][-1][1])<5:
                    to_pandas_pos[entry].append((y_collapsed[i],i))
                    to_pandas_slopes[entry].append((slope_collapsed[i],i))
                    hit = True
        if hit == False:
            to_pandas_pos[entries] = [(y_collapsed[i],i)]
            to_pandas_slopes[entries] = [(slope_collapsed[i],i)]
            entries+=1
                
    print(f"to pandas pos: {to_pandas_pos}")
    print(f"to pandas pos: {to_pandas_slopes}")
    tval_pos = {}
    yval = {}
    sval = {}
    tval_slope = {}
    for i, elem in enumerate(to_pandas_pos):
            tval_pos[i] = [s[1] for s in to_pandas_pos[elem]]
            yval[i] = [s[0] for s in to_pandas_pos[elem]]

    for j, elem2 in enumerate(to_pandas_slopes):
            tval_slope[j] = [s[1] for s in to_pandas_slopes[elem2]]
            sval[j] = [s[0] for s in to_pandas_slopes[elem2]]


    style = "seaborn-talk"
    mpl.style.use(style)

    for key in tval_pos:
        if len(yval[key])>100:
            plt.plot(tval_pos[key], yval[key])
    plt.title('ORGANIC Position vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Position')
    plt.show()

    for key in tval_pos:
        if len(yval[key])>100 and yval[key][0] > 75:
            filt = scipy.signal.savgol_filter(yval[key], len(yval[key]), 4)
            plt.plot(tval_pos[key], filt)
    plt.title('SAVGOL poly fit Position vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Position')
    plt.show()

    for key in tval_pos:
        if len(yval[key])>100 and yval[key][0] > 50:
            filt = scipy.signal.savgol_filter(yval[key],75, 1)
            plt.plot(tval_pos[key], filt)
    plt.title('75 frame smoothing Position vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Position')
    plt.show()

    for key in tval_slope:
        if len(sval[key])>100:
            plt.plot(sval[key])
    plt.title('ORGANIC Entry angle vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Entry angle')
    plt.show()

    for key in tval_slope:
        if len(sval[key])>100:
            filt = scipy.signal.savgol_filter(sval[key],75, 1)
            plt.plot(filt)
    plt.title('75 frame smoothing Entry angle vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Entry angle')
    plt.show()

    for key in tval_slope:
        if len(sval[key])>100:
            filt = scipy.signal.savgol_filter(sval[key],len(sval[key]), 4)
            plt.plot(filt)
    plt.title('SAVGOL poly fit Entry angle vs Time')
    plt.xlabel('Frame #')
    plt.ylabel('Entry angle')
    plt.show()
    
    for key in yval:
        if len(yval[key])>100:
            plt.plot(yval[key], sval[key])
    plt.title('ORGANIC Entry angle vs position')
    plt.xlabel('position')
    plt.ylabel('entry angle')
    plt.show()

    for key in yval:
        if len(yval[key])>100:
            filt_y = scipy.signal.savgol_filter(yval[key],len(yval[key]), 4)
            filt_s = scipy.signal.savgol_filter(sval[key],len(sval[key]), 4)
            plt.plot(filt_y, filt_s)
    plt.title('SAVGOL polyfit Entry angle vs position')
    plt.xlabel('position')
    plt.ylabel('entry angle')
    plt.show()

    for key in yval:
        if len(yval[key])>100:
            filt_y = scipy.signal.savgol_filter(yval[key],75, 1)
            filt_s = scipy.signal.savgol_filter(sval[key],75, 1)
            plt.plot(filt_y, filt_s)
    plt.title('75 frame moving average Entry angle vs position')
    plt.xlabel('position')
    plt.ylabel('entry angle')
    plt.show()