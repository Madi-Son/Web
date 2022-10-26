#webs_v1
#Robert Hutton, sam@samhutton.net
#Dept. of Mechanical Engineering - University of Nevada, Reno

import csv
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pims
import trackpy as tp

mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')

@pims.pipeline
def gray(image):
    return image[1, 1, 1]  # Take just the green channel

frames = gray(pims.open('8in_misalign_1mil_75percent.MOV'))

frames
print(frames[0])  # the first frame
plt.imshow(frames[0])
/Users/roberthutton/Desktop/Lab Work/wrinkles/webs/webs_v1/test2.mp4
# def open_workbook(workbook):
#     workbook = pd.read_excel(workbook)
#     workbook.head()
#     print(workbook)


# class web_data:
#     def __init__(self, file):
#         self.time = file

# open_workbook('Master Web Data 3.xlsx')
