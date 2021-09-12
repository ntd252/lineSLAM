import floor_segmentation as fs
import db_main as db
import cv2
import numpy as np
from math import pi, cos, sin, sqrt
import matplotlib.pyplot as plt


line_list = np.array([[ 548, -267,  588,   -60],
 [ 584,   91,  579,   17],
 [ 605,  189,  584,   35]])

temp = line_list
y1_y2 = np.array([temp[:,1],temp[:,3]]) #get y1 y2 out of the line list
y1_y2 = np.abs(y1_y2)
index_min_y1_y2 = np.argmin(y1_y2, axis=0) #0 for y1, 1 for y2
index_min_y1_y2[index_min_y1_y2 == 1] += 2 #get index for y2 in line form [x1 y1 x2 y2]
index_min_y1_y2[index_min_y1_y2 == 0] += 1  #get index for y1 in line form [x1 y1 x2 y2
index_min = np.column_stack((index_min_y1_y2-1, index_min_y1_y2)) #merge index of (x, y)
temp_filter = np.take_along_axis(temp,index_min,1)    #get (x,y) from line list
print("temp_filter", temp_filter)


A = line_list[:,1] - line_list[:,3]
B = line_list[:,2] - line_list[:,0]
C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]
r_meas = np.divide(np.abs(C), np.sqrt(A*A+B*B))
alpha_meas = (np.arctan2(-B*C,-A*C) + pi) % (2*pi) - pi
print("r_meas", r_meas)

print("alpha_meas", alpha_meas)

list = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
index_list = np.array([0,2])

index_test1 = index_list*2 + 3
index_test2 = index_list*2 + 4
index_test = np.concatenate((index_test1, index_test2))

print("index_test12", index_test1, index_test2)
print("index_test", index_test)

list = np.delete(list, index_test, None)
print("list after: ", list)

d = 150.0
xR = 4.93740296e+02
yR = -8.25360113e+01
theta = -1.71336444e+00



'''
line = np.array([[146,-501,45,-370],
                [454, -897, 321, -725],
                [ 465, -911,  290, -686],
                [ 428, -864,  333, -741],
                [ 221, -598,  109, -454],
                [ 321, -725,  292, -688],
                [ 287, -683,  237, -618],
                [ 465, -911,  420, -853],
                [ 128, -478,   83, -420]])
'''
'''
x1 = line[:,0]
y1 = line[:,1]
x2 = line[:,2]
y2 = line[:,3]
x = np.hstack((x1, x2))
y = np.hstack((y1, y2))
print(x)
print(y)
plt.scatter(x, y)
plt.show()
'''