from os import startfile
import numpy as np
from math import pi
from timeit import default_timer as timer

def check_overlap(line1, line2):
    combination = np.array([line1,
                         line2,
                         [line1[0], line1[1], line2[0], line2[1]],
                         [line1[0], line1[1], line2[2], line2[3]],
                         [line1[2], line1[3], line2[0], line2[1]],
                         [line1[2], line1[3], line2[2], line2[3]]])
    distance = np.sqrt((combination[:,0] - combination[:,2])**2 + (combination[:,1] - combination[:,3])**2)
    max = np.amax(distance)
    overlap = distance[0] + distance[1] - max
    endpoint = combination[np.argmax(distance)]
    return (overlap >= 0), endpoint

overlap1 = np.array([[1,4,5,5], [7,7,3,5]])
overlap2 = np.array([[8,1,12,2], [9,2,11,3]])

non_overlap = np.array([[1,2,5,3], [6,3,9,4]])
overlap, points = check_overlap(overlap2[0], overlap2[1])
print("overlap", overlap)
print("endponts", points)

start = timer()
overlap1 = np.append(overlap1, [overlap2[0]], axis=0)
stop = timer()
nparraytime = stop-start
print("overlap1", overlap1)

test_list = []
start = timer()
test_list.append(overlap2[0])
stop = timer()
print("python list time", nparraytime - (stop-start))

test_list.append(overlap2[1])
print("test_list", test_list[0])


line_list = np.array([[1,0,0,1],[0,2,2,2],[2,0,2,2],[-1,0,0,-1]])
test_state = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
state = np.column_stack((test_state[3::2], test_state[4::2]))
compare = np.array([+20,-10])
substract = np.abs(state - compare)
index = np.nonzero(np.logical_and(substract[:,0] < 17,substract[:,1] < 20))

A = line_list[:,1] - line_list[:,3]
B = line_list[:,2] - line_list[:,0]
C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]

x_L = 0
y_L = 0

x_o = (B*(B*x_L - A*y_L) - A*C)/(A*A + B*B)
y_o = (A*(A*y_L - B*x_L) - B*C)/(A*A + B*B)

theta = 0
dx = x_o - x_L
dy = y_o - y_L
r = np.sqrt(dx*dx + dy*dy)
alpha = (np.arctan2(dy, dx) - theta + pi) % (2*pi) - pi

rm = np.divide(np.abs(C), np.sqrt(A*A+B*B))
alpham = (np.arctan2(-B,-A) + pi) % (2*pi) - pi
stackm = np.column_stack((rm, alpham))
print("stackm", stackm)
