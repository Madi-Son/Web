import csv
import cv2 as cv
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
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
                #print(abs(y_collapsed[i] - to_pandas_pos[entry][-1][0]))
                #print((i-to_pandas_pos[entry][-1][1]))
                if entry in to_pandas_pos and (y_collapsed[i] - to_pandas_pos[entry][-1][0]) < 25 and (y_collapsed[i] - to_pandas_pos[entry][-1][0]) > -25 and (i-to_pandas_pos[entry][-1][1])<5 and (i-to_pandas_pos[entry][-1][1])>0:
                    to_pandas_pos[entry].append((y_collapsed[i],i))
                    to_pandas_slopes[entry].append((slope_collapsed[i],i))
                    hit = True
        if hit == False:
            to_pandas_pos[entries] = [(y_collapsed[i],i)]
            to_pandas_slopes[entries] = [(slope_collapsed[i],i)]
            entries+=1
                
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

    # images = []
    # sequence = sorted(glob.glob(f"output_sequence/*.png")) #create a sequence of images where the images are represented by their filename
    # for image in sequence:
    #     images.append(cv.imread(str(image))) #create a list of images where the images are represented by the actual image arrays
    # i = 0
    # for key in yval:
    #     for elem in yval[key]:
    #         print(elem)
    #         print(i,j)
    #         if (i+j) < len(images)-1:
    #             image[i+j] = cv.rectangle(image[i+j], (185, elem), (200, elem + 25), (0, 255, 255), 5)
    #     i+=1


    style = "seaborn-talk"
    mpl.style.use(style)


    # -- PLOTS --
    min_frames = 75

    # for key in tval_slope:
    #     print(sval[key])
    #     if len(sval[key])>min_frames:
    #         plt.plot(sval[key], alpha = 0.25)

    # for key in yval:
    #     if len(yval[key])>min_frames:
    #         entry_angle = -1*(theta - np.arctan((np.array(yval[key])*np.cos(theta)/pixel_to_inch)/(H0-np.array(yval[key])*np.sin(theta)/pixel_to_inch)))
    #         plt.plot(entry_angle)
    # plt.title('theory for entry angle plotted over actual data')
    # plt.xlabel('Time')
    # plt.ylabel('Entry Angle')
    # plt.show()
    # print(tval_pos)
    # print(sval)
    # print(yval)
    # ppi = 84.3333
    # V = 0.16
    # L = 18
    # Lc = 0.5
    # Rc = 0.5
    # W = 9
    # C = Lc + Rc - W

    # tval = np.linspace(0,250,100)
    # uval = (C*np.exp(-tval * V/L)-Rc) + Lc +1
    # vval = V*np.sin(uval)

    ppi = 84.3333
    a = 0.45
    V = 0.5*a
    L = 18
    Lc = 0
    Rc = 9
    W = 9
    tval = np.linspace(0,175,50)
    uval1 = -1*(np.array(tval)*(a*(Rc-Lc))/(L+np.array(tval)*a)+Lc)+7
    uval2 = -1*(np.array(tval)*(V*(Rc-Lc))/(L+np.array(tval)*V)+Lc)+7
    vval = V*np.sin(uval1)

    #Plot position vs total time series
    for key in tval_pos:
        if len(yval[key])>min_frames:
            yplot = [(num / ppi) for num in yval[key]]
            plt.plot(yplot, alpha = 0.5)
    line, = plt.plot(tval,uval1, 'bo')
    line, = plt.plot(tval,uval2, 'go')
    line.set_label('Derived model')
    plt.title('Position vs Time')
    plt.xlabel('Time (frame)')
    plt.ylabel('Position (in)')
    plt.show()

    # # create the figure and axes objects
    # fig, ax = plt.subplots()

    # # function that draws each frame of the animation
    # # x = []
    # # y = []
    # # tflat = []
    # # pflat = []
    # # for key in tval_pos:
    # #         for l in range(len(tval_pos[key])):
    # #             tflat.append(tval_pos[key][l])
    # #             pflat.append(yval[key][l])
    # # sortflat_t = sorted(tflat)
    # # sortflat_y = [x for _, x in sorted(zip(tflat, pflat))]
    # # def animate(i):
    # #     t = sortflat_t[i]
    # #     pt = sortflat_y[i] # grab a random integer to be the next y-value in the animation
    # #     x.append(t)
    # #     y.append(pt)

    # #     ax.clear()
    # #     ax.plot(x, y, 'ro')
    # #     # ax.set_xlim([0,20])
    # #     # ax.set_ylim([0,10])ß

    # #     # run the animation
    # # ani = FuncAnimation(fig, animate, interval=2)

    # # plt.show()
    # # #and with smoothing
    # # for key in tval_pos:
    # #     if len(yval[key])>min_frames:
    # #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    # #         plt.plot(tval_pos[key], filt)
    # # plt.title('25 frame smoothing Position vs Time')
    # # plt.xlabel('Frame #')
    # # plt.ylabel('Position (px)')
    # # plt.show()

    # #Plot position vs collapsed time series
    # for key in tval_pos:
    #     if len(yval[key])>min_frames:
    #         plt.plot(yval[key])
    # plt.title('Position vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Position (px)')
    # plt.show()
    # #and with smoothing
    # for key in tval_pos:
    #     if len(yval[key])>min_frames:
    #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    #         plt.plot(filt)
    # plt.title('25 frame smoothing Position vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Position (px)')
    # plt.show()

    # #Plot entry angle vs total time series
    # for key in tval_slope:
    #     if len(sval[key])>min_frames:
    #         plt.plot(tval_slope[key],sval[key])
    # plt.title('Entry Angle vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Entry Angle (rad)')
    # plt.show()
    # #and with smoothing
    # for key in tval_slope:
    #     if len(sval[key])>min_frames:
    #         filt = scipy.signal.savgol_filter(sval[key],25, 1)
    #         plt.plot(tval_slope[key],filt)
    # plt.title('25 frame smoothing Entry Angle vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Entry Angle (rad)')
    # plt.show()

    # #Plot entry angle vs collapsed time series
    # for key in tval_slope:
    #     if len(sval[key])>min_frames:
    #         plt.plot(sval[key])
    # plt.title('Entry Angle vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Entry Angle (rad)')
    # plt.show()
    # #and with smoothing
    # for key in tval_slope:
    #     if len(sval[key])>min_frames:
    #         filt = scipy.signal.savgol_filter(sval[key],25, 1)
    #         plt.plot(filt)
    # plt.title('25 frame smoothing Entry Angle vs Time')
    # plt.xlabel('Frame #')
    # plt.ylabel('Entry Angle (rad)')
    # plt.show()

    # #plot slope vs position
    # for key in tval_slope:
    #     if len(sval[key])>min_frames:
    #         plt.plot(yval[key],sval[key])
    # plt.title('Entry Angle vs Position')
    # plt.xlabel('Position (px)')
    # plt.ylabel('Entry angle (rad)')
    # plt.show()
    # #and with smoothing
    # for key in tval_slope:
    #     if len(sval[key])>100:
    #         filt_pos = scipy.signal.savgol_filter(yval[key],25, 1)
    #         filt_slope = scipy.signal.savgol_filter(sval[key],25, 1)
    #         plt.plot(filt_pos, filt_slope)
    # plt.title('25 frame smoothing Entry Angle vs Position')
    # plt.xlabel('Position (px)')
    # plt.ylabel('Entry angle (rad)')
    # plt.show()



  
    #Get velocity data from postions
    dt = 1
    for key in tval_pos:
        if len(yval[key])>min_frames and yval[key][0] > 50:
            filt = scipy.signal.savgol_filter(yval[key],25, 1)
            velocity = (np.diff(filt) / dt)/ppi
            plt.plot(velocity)
            plt.plot(tval, vval, 'go')
    plt.title('Velocity from Position (25 frame smoothing on position only)')
    plt.xlabel('Frame #')
    plt.ylabel('Wrinkle Velocity (px/frame)')
    plt.show()

    # #Get velocity data from postionsn and plot on complete time series
    # dt = 0.1
    # for key in tval_pos:
    #     if len(yval[key])>min_frames and yval[key][0] > 50:
    #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    #         velocity = (np.diff(filt) / dt)
    #         plt.plot(tval_pos[key][1:], velocity)
    # plt.title('Velocity from Position (25 frame smoothing on position only)')
    # plt.xlabel('Frame #')
    # plt.ylabel('Wrinkle Velocity (px/frame)')
    # plt.show()

    # #Get velocity data from postions and smooth
    # dt = 0.1
    # for key in tval_pos:
    #     if len(yval[key])>min_frames and yval[key][0] > 50:
    #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    #         velocity = (np.diff(filt) / dt)
    #         vel = scipy.signal.savgol_filter(velocity,25, 1)
    #         plt.plot(vel)
    # plt.title('Velocity from Position (25 frame smoothing on position and velocity sets)')
    # plt.xlabel('Frame #')
    # plt.ylabel('Wrinkle Velocity (px/frame)')
    # plt.show()

    # #Get velocity data from postions and smooth and plot agains complete time series
    # dt = 0.1
    # for key in tval_pos:
    #     if len(yval[key])>min_frames and yval[key][0] > 50:
    #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    #         velocity = (np.diff(filt) / dt)
    #         vel = scipy.signal.savgol_filter(velocity,25, 1)
    #         plt.plot(tval_pos[key][1:],vel)
    # plt.title('Velocity from Position (25 frame smoothing on position and velocity sets)')
    # plt.xlabel('Frame #')
    # plt.ylabel('Wrinkle Velocity (px/frame)')
    # plt.show()

    # #Get velocity data from postions and smooth and plot against position series
    # dt = 0.1
    # for key in tval_pos:
    #     if len(yval[key])>min_frames and yval[key][0] > 50:
    #         filt = scipy.signal.savgol_filter(yval[key],25, 1)
    #         velocity = (np.diff(filt) / dt)
    #         vel = scipy.signal.savgol_filter(velocity,25, 1)
    #         plt.plot(vel, sval[key][1:])
    # plt.title('Velocity vs Entry angle')
    # plt.xlabel('Entry Angle (rad)')
    # plt.ylabel('Wrinkle Velocity (px/frame)')
    # plt.show()

    