from mpl_toolkits import mplot3d
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

style = "seaborn-talk"
plt.style.use(style)

#Misalign: Theta = pi/2 - cos^-1(displacement/ 38.8 cm)   
#Twist: Theta = pi/2 - cos^-1(displacement/ 21.7 cm) 
def misalign_disp(disp):
    return np.pi/2 - np.arccos((disp/21.70))

def machine_velocity(v):
    if v == 75:
        return 0.86
    elif v == 80:
        return 0.945
    elif v == 85:
        return 1.03
    elif v == 90:
        return 1.115
    elif v == 95:
        return 1.2
    return 0

def aspect_ratio(w):
    return w/18

#plot machine velocity calibration
xval = [75,80,85,90,95]
yval = [0.86,0.945,1.03,1.115,1.2]
print(xval, yval)
plt.plot(xval, yval, 'bo-')
plt.title('Web velocity as a function of % max voltage')
plt.xlabel('% Voltage')
plt.ylabel('Actual Velocity (cm/s)')
plt.show()

#plot critical angle vs tension
xval = [6,6,6,7,7,7,8,8,8,9,9,9]
yval = [misalign_disp(0.2), misalign_disp(0.22), misalign_disp(0.21), misalign_disp(0.24), misalign_disp(0.27), misalign_disp(0.26), misalign_disp(0.29), misalign_disp(0.30), misalign_disp(0.31), misalign_disp(0.33), misalign_disp(0.34), misalign_disp(0.35)]
yerr = np.pi/2 - np.arccos((0.025/21.70))
print(xval, yval)
plt.errorbar(xval, yval, yerr,fmt = 'o', mfc='blue', ecolor = 'red')
plt.title('Critical Misalignment vs Web Tension, 8" width, 1mil thickness')
plt.xlabel('Tension (lbs)')
plt.ylabel('Critical misalign angle (rad)')
plt.show()

#plot critical angle vs machine velocity
xval = [machine_velocity(75),machine_velocity(75),machine_velocity(75),machine_velocity(80),machine_velocity(80),machine_velocity(80),machine_velocity(85),machine_velocity(85),machine_velocity(85),machine_velocity(90),machine_velocity(90),machine_velocity(90),machine_velocity(95),machine_velocity(95),machine_velocity(95)]
yval = [misalign_disp(0.31), misalign_disp(0.31), misalign_disp(0.32), misalign_disp(0.31), misalign_disp(0.29), misalign_disp(0.3), misalign_disp(0.30), misalign_disp(0.29), misalign_disp(0.32), misalign_disp(0.30), misalign_disp(0.29), misalign_disp(0.31), misalign_disp(0.30), misalign_disp(0.29), misalign_disp(0.28)]
print(xval, yval)
plt.errorbar(xval, yval, yerr,fmt = 'o', mfc='blue', ecolor = 'red')
plt.title('Critical Misalignment vs Web Velocity, 8" width, 1mil thickness')
plt.xlabel('Machine Velocity (cm/s)')
plt.ylabel('Critical misalign angle (rad)')
plt.show()

#plot critical angle vs aspect ratio
xval = [aspect_ratio(6),aspect_ratio(6),aspect_ratio(6),aspect_ratio(7),aspect_ratio(7),aspect_ratio(7),aspect_ratio(8),aspect_ratio(8),aspect_ratio(8),aspect_ratio(9),aspect_ratio(9),aspect_ratio(9)]
yval = [misalign_disp(0.22), misalign_disp(0.25), misalign_disp(0.27), misalign_disp(0.35), misalign_disp(0.33), misalign_disp(0.34), misalign_disp(0.35), misalign_disp(0.37), misalign_disp(0.37), misalign_disp(0.36), misalign_disp(0.39), misalign_disp(0.40)]
print(xval, yval)
plt.errorbar(xval, yval, yerr,fmt = 'o', mfc='blue', ecolor = 'red')
plt.title('Critical Misalignment vs Aspect Ratio, 8" width, 1mil thickness')
plt.xlabel('Span aspect ratio (W/L)')
plt.ylabel('Critical misalign angle (rad)')
plt.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')

# # Data for a three-dimensional line

# xline = np.linspace(0, 0.01, 10)
# yline = np.linspace(0, 0.01, 10)
# zline = []
# for i in range(10):
#     zline.append(-(1/3)*np.exp(np.pi**2*np.array(yline))*np.cos(np.pi*np.array(xline)))
# print(zline)
# ax.plot3D(xline, yline, zline, 'gray')

# plt.show()

# from numpy import exp,arange
# from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

# # the function that I'm going to plot
# def z_func(x,y):
#  return (1-(x**2+y**3))*exp(-(x**2+y**2)/2)
 
# x = arange(0,30,0.1)
# y = arange(0,30,0.1)
# X,Y = meshgrid(x, y) # grid of point
# Z = z_func(X, Y) # evaluation of the function on the grid

# im = imshow(Z,cmap=cm.RdBu) # drawing the function
# # adding the Contour lines with labels
# cset = contour(Z,arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
# clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
# colorbar(im) # adding the colobar on the right
# # latex fashion title
# title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
# show()