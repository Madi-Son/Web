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

class ImageSeries():
    def __init__(self, location): #use init function to define the sequence of images as self.sequence using glob.glob in the directory where the images are
        self.images = []
        self.sequence = sorted(glob.glob(f"{location}/*.png")) #create a sequence of images where the images are represented by their filename
        for image in self.sequence:
            self.images.append(cv.imread(str(image))) #create a list of images where the images are represented by the actual image arrays

class ProcessImage(ImageSeries):
    def blur(self, kernel_size): #function to apply a gaussian blur to the image
        self.crop_frame = cv.Canny(self.images[round(len(self.images)/2)], 255/3, 255) #grab the middle frame as our crop set
        blurred_images = []
        for image in self.images: #step through each image in the input series
            blurred_images.append(cv.blur(image,(kernel_size,kernel_size))) #use openCV gaussian blur to apply gaussian blur
        self.images = blurred_images
        for i, image in enumerate(self.images): #loop through each image in the current series
            cv.imwrite('output_sequence/'+f"output_blurred{i}.png", image)

    def gamma(self, gamma_size): #produce a gamma table based on the input value and use the table to change the gamma of the image
        invGamma = 1 / gamma_size #invert the gamma value to build table from
        table = [((i / 255) ** invGamma) * 255 for i in range(256)] #construct gamma table
        table = np.array(table, np.uint8) #store table as an array
        gamma_adjust = []
        for image in self.images: #apply the gamma table to each image in the sequence by using openCV LUT
            gamma_adjust.append(cv.LUT(image, table))
        self.images = gamma_adjust

    def sharpen(self, sharpening_kernel_key): #sharpen the images by a given kernel key by convolving the image with a sharpening kernel
        sharpened_images = []
        for image in self.images:   
            kernel = np.array([ [-1,-1,-1], #the kernel is constructed of -1's with the sharpening key in the center
                            [-1, sharpening_kernel_key,-1],
                            [-1,-1,-1]])
            sharpened_images.append*(cv.filter2D(image, -1, kernel)) #apply the convulution to the image
            self.images = sharpened_images
            
    def canny(self, threshold1, threshold2): #canny edge filter grabs edges from each image in the series and dumps the edges into self.edges
        i = 0 
        self.canny_edges = []
        for image in self.images: 
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #image must be this format for canny edge from openCV
            self.canny_edges.append(cv.Canny(image, threshold1, threshold2)) #append self.cannyedges along with self.images
            # cv.imwrite('output_sequence/'+f"canny_{i}.png", self.canny_edges[i])
            i += 1
        for i, image in enumerate(self.canny_edges): #loop through each image in the current series
            cv.imwrite('output_sequence/'+f"output_canny{i}.png", image)

    def crop_span(self): #crops the frames produced by canny algorith based on the top and bottom edges of the span, uses spacing of gradients
        x_center = int(1920/2)  #assume input footage is 1920x1080
        y,x = self.crop_frame.shape
        self.span_top_edge = 0 #initally, set the crop parameters to the frame size (no crop)
        self.span_bottom_edge = 1080
        for j in range(int(1080/2)):
            if self.crop_frame[j, x_center] > 125: #perform a linear search for >0 pixel values in the canny image
                self.span_top_edge = j #the last >0 pixel value bocomes the top edge
        for l in range(int(1080/2)):
            if self.crop_frame[1079-l,x_center] > 125: #same logic here as above but inverted to search from the bottom
                self.span_bottom_edge = 1080 - l
        crop2 = []
        crop1 = []
        
        black = [np.zeros((self.span_bottom_edge-self.span_top_edge, 1920, 3), dtype = "uint8")]*len(self.images)
        for image in self.canny_edges:
            crop1.append(image[self.span_top_edge+25:self.span_bottom_edge, 185:(215+250)])
        self.canny_edges = crop1
        for image in self.images:
            crop2.append(image[self.span_top_edge:self.span_bottom_edge, 0:1920])
        self.images = crop2

        i = 0
        for image in self.canny_edges: 
            # cv.imwrite('output_sequence/'+f"canny_cropped_{i}.png", self.canny_edges[i])
            i += 1
        for i, image in enumerate(self.images): #loop through each image in the current series
            cv.imwrite('output_sequence/'+f"output_cropped{i}.png", image)
        
    def hough_linesP(self):
        self.linesP = []
        # for image in self.canny_edges:
        #     self.linesP.append(cv.HoughLinesP(image, rho = 1, theta = np.pi/360, threshold = thresh, minLineLength = min_len, maxLineGap = min_sep)) #for each image, append the lines to a blank list self.lines
        for image in self.canny_edges:
            #self.linesP.append(cv.HoughLinesP(image, rho=1, theta=np.pi/500, threshold=50, minLineLength=25, maxLineGap=25)) #for each image, append the lines to a blank list self.lines
            self.linesP.append(cv.HoughLinesP(image, rho=1, theta=np.pi/1080, threshold=50, minLineLength=150, maxLineGap=100)) #for each image, append the lines to a blank list self.lines
            
    def overlay_lines(self): #here we draw the lines on the original set of images
        padded = []
        for image in self.images:
            padded.append(cv.copyMakeBorder(image, self.span_top_edge, 1080-self.span_bottom_edge, 0, 0, cv.BORDER_CONSTANT,value=[0,0,0]))
        self.images = padded 
        i = 0
        self.slopes = []
        filtered_lines = {}
        for image in self.images:
            self.slopes.append([])
            if self.linesP[i] is not None:
                taken_points = []
                free_line = True
                j = 0
                for line in self.linesP[i]:
                    x1, y1, x2, y2 = line[0]
                    slope = (y1-y2)/(x2-x1)
                    if x1<x2 and y2<y1 and slope>0 and slope<(np.pi/2):
                        for item in taken_points:
                            if abs(y1 - item) < 50:
                                free_line = False
                        if free_line == True:
                            taken_points.append(y1)
                            cv.line(image, (x1+185, y1+self.span_top_edge+25), (x2+185, y2+self.span_top_edge+25), (0, 0, 255), 2)
                            #image = cv.rectangle(image, (x1+185-50, y1+self.span_top_edge+25), (x2+185+50, y2+self.span_top_edge+25), (0, 255, 255), 1)
                            filtered_lines[i,j] = [slope, y1]
                    j+=1
            i+=1
        for i, image in enumerate(self.images): #loop through each image in the current series
            cv.imwrite('output_sequence/'+f"output_lines{i}.png", image)

        tracking = {}
        key_last = None
        k = 0
        collapsed = []
        self.y_collapsed = []
        self.slope_collapsed = []

        for key in filtered_lines:
            print(f"filtered lines {filtered_lines[key]}")
            collapsed.append(tuple(filtered_lines[key]))
            self.y_collapsed.append(filtered_lines[key][1])
            self.slope_collapsed.append(filtered_lines[key][0])
            if key[0] == key_last:
                tracking[key[0]].extend(filtered_lines[key])

            else:
                tracking[key[0]] = (filtered_lines[key])
            k+=1
            key_last = key[0]
            
        print(collapsed)
        with open('collapsed_pairs.pickle', 'wb') as f:
            pickle.dump(collapsed, f)

        with open('pos_collapsed.pickle', 'wb') as g:
            pickle.dump(self.y_collapsed, g)

        with open('slope_collapsed.pickle', 'wb') as k:
            pickle.dump(self.slope_collapsed, k)
        
        
    def write_series(self): #method is called at the end of an operation chain to write the final images to a new directory
        for i, image in enumerate(self.images): #loop through each image in the current series
            cv.imwrite('output_sequence/'+f"output_{i}.png", image) #write the new image to the directory with incrementing, sequential filenames
        
        for j in range(len(self.slopes)-1):
            for i in range(len(self.slopes[j])-1):
                plt.plot(self.slopes[j][i])

    def draw_line_of_entry(self, position):
        for image in self.images:
            cv.line(image, (position,0), (position,1080), (255, 255, 0), 2)