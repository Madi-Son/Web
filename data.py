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
sns.set_theme(style="white", palette=None)

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
    if entries < 4:
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
        # elif (entries-1) in to_pandas_pos and abs(y_collapsed[i] - to_pandas_pos[entries-1][-1]) < 25:
        #     to_pandas_pos[entries-1].append(y_collapsed[i])
        #     to_pandas_slopes[entries-1].append(slope_collapsed[i])
        #     hit = True
        # elif (entries-2) in to_pandas_pos and abs(y_collapsed[i] - to_pandas_pos[entries-2][-1]) < 25:
        #     to_pandas_pos[entries-2].append(y_collapsed[i])
        #     to_pandas_slopes[entries-2].append(slope_collapsed[i])
        #     hit = True
        # elif (entries-3) in to_pandas_pos and abs(y_collapsed[i] - to_pandas_pos[entries-3][-1]) < 25:
        #     to_pandas_pos[entries-3].append(y_collapsed[i])
        #     to_pandas_slopes[entries-3].append(slope_collapsed[i])
        #     hit = True
        # elif (entries-4) in to_pandas_pos and abs(y_collapsed[i] - to_pandas_pos[entries-4][-1]) < 25:
        #     to_pandas_pos[entries-4].append(y_collapsed[i])
        #     to_pandas_slopes[entries-4].append(slope_collapsed[i])
        #     hit = True
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
    if len(yval[key])>50:
        plt.plot(yval[key])
plt.title('Position vs Time')
plt.xlabel('Frame #')
plt.ylabel('Position')
plt.show()

for key in tval_slope:
    if len(sval[key])>50:
        plt.plot(sval[key])
plt.title('Entry angle vs Time')
plt.xlabel('Frame #')
plt.ylabel('Entry angle')
plt.show()
print(f"sval = {sval}")
for key in yval:
    if len(yval[key])>50:
        plt.plot(yval[key], sval[key])


plt.title('Entry angle vs position')
plt.xlabel('position')
plt.ylabel('entry angle')
plt.show()