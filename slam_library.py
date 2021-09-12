#Library for line matching, includes:
# Check if 2 lines are overlap
# Check if 2 lines are close to each other
# Check if detected lines from floor segmentation result are new or already detected
# Merge 2 lines if they are detected to be the same line.

from math import sin, cos, pi, sqrt
import numpy as np
import cv2

def distance_point(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
    return (overlap >= -230), endpoint

def get_close_lines(robot):

    slope = 0.57
    #get current robot location
    theta = robot.state[2]
    x_L = robot.state[0] + robot.scanner_displacement*cos(theta)
    y_L = robot.state[1] + robot.scanner_displacement*sin(theta)
    line_in_state = np.column_stack((robot.state[3::2], robot.state[4::2])) #[[r, alpha],...] from state
    a_matrix = np.cos(line_in_state[:,1])
    b_matrix = np.sin(line_in_state[:,1])
    r_state_in_cam = np.abs(x_L*a_matrix + y_L*b_matrix - line_in_state[:,0])
    close_lines = np.nonzero(r_state_in_cam < 500)

    #convert robot.lines list from world coordinate to camera coordinate system
    x1 = cos(theta)*robot.lines[:,0] + sin(theta)*robot.lines[:,1] - x_L#x_cam1
    y1 = -sin(theta)*robot.lines[:,0] + cos(theta)*robot.lines[:,1] - y_L#y_cam1
    x2 = cos(theta)*robot.lines[:,2] + sin(theta)*robot.lines[:,3] - x_L#x_cam2
    y2 = -sin(theta)*robot.lines[:,2] + cos(theta)*robot.lines[:,3] - y_L#y_cam2
    line_list = np.column_stack((x1, y1, x2, y2)).astype(int)


    y1_y2 = np.array([line_list[:,1],line_list[:,3]]) #get y1 y2 out of the line list
    #upper maximum
    index_max_y1_y2 = np.argmax(y1_y2, axis=0) #0 for y1, 1 for y2
    index_max_y1_y2[index_max_y1_y2 == 1] += 2 #get index for y2 in line form [x1 y1 x2 y2]
    index_max_y1_y2[index_max_y1_y2 == 0] += 1  #get index for y1 in line form [x1 y1 x2 y2
    index_max = np.column_stack((index_max_y1_y2-1, index_max_y1_y2)) #merge index of (x, y)
    max_filter = np.take_along_axis(line_list,index_max,1)    #get (x,y) from line list
    x_positive = np.nonzero(max_filter[:,0] > 0)
    out_of_upper = np.nonzero((max_filter[:,1] - slope*max_filter[:,0]) > 0)
    out_of_upper = np.intersect1d(x_positive, out_of_upper)

    #negative minimum
    index_max_y1_y2 = np.argmin(y1_y2, axis=0) #0 for y1, 1 for y2
    index_max_y1_y2[index_max_y1_y2 == 1] += 2 #get index for y2 in line form [x1 y1 x2 y2]
    index_max_y1_y2[index_max_y1_y2 == 0] += 1  #get index for y1 in line form [x1 y1 x2 y2
    index_max = np.column_stack((index_max_y1_y2-1, index_max_y1_y2)) #merge index of (x, y)
    max_filter = np.take_along_axis(line_list,index_max,1)    #get (x,y) from line list
    x_positive = np.nonzero(max_filter[:,0] > 0)
    out_of_lower = np.nonzero((max_filter[:,1] + slope*max_filter[:,0]) < 0)
    out_of_lower = np.intersect1d(x_positive, out_of_lower)

    out_of_range = np.intersect1d(out_of_upper, out_of_lower)


    point1 = line_list[:,0:2]
    point2 = line_list[:,2:4]
    point1_in_range_up = (point1[:,1] - slope*point1[:,0]) < 0
    point1_in_range_low = (point1[:,1] + slope*point1[:,0]) > 0
    point1_in_range = np.logical_and(point1_in_range_up, point1_in_range_low)
    point1_positive = point1[:,0] > 0
    point1_in_range_positive = np.logical_and(point1_in_range, point1_positive)

    point2_in_range_up = (point2[:,1] - slope*point2[:,0]) < 0
    point2_in_range_low = (point2[:,1] + slope*point2[:,0]) > 0
    point2_in_range = np.logical_and(point2_in_range_up, point2_in_range_low)
    point2_positive = point2[:,0] > 0
    point2_in_range_positive = np.logical_and(point2_in_range, point2_positive)

    point_in_range = np.array(np.nonzero(np.logical_or(point1_in_range_positive, point2_in_range_positive))) #index of line which has at least 1 point in the range
    
    in_range_radius = np.concatenate((out_of_range, point_in_range[0]))
    robot.current_close_lines = np.intersect1d(in_range_radius, close_lines)
    #print("robot.current_close_line", robot.current_close_lines)


def get_observations(line_list_cam, robot,
                     max_r_distance, max_alpha_distance):

    a = line_list_cam[:,1] - line_list_cam[:,3]
    b = line_list_cam[:,2] - line_list_cam[:,0]
    c = line_list_cam[:,0]*line_list_cam[:,3] - line_list_cam[:,2]*line_list_cam[:,1]
    r_meas = np.divide(np.abs(c), np.sqrt(a*a+b*b))
    alpha_meas = (np.arctan2(-b*c,-a*c) + pi) % (2*pi) - pi
    measurement = np.column_stack((r_meas, alpha_meas))
        
    # Compute scanner pose from robot pose.
    theta = robot.state[2]
    x_L = robot.state[0] + robot.scanner_displacement*cos(theta)
    y_L = robot.state[1] + robot.scanner_displacement*sin(theta)
    #convert line_list from camera coordinate to world coordinate system
    X1 = cos(theta)*line_list_cam[:,0] - sin(theta)*line_list_cam[:,1] + x_L#x1
    Y1 = sin(theta)*line_list_cam[:,0] + cos(theta)*line_list_cam[:,1] + y_L#y1
    X2 = cos(theta)*line_list_cam[:,2] - sin(theta)*line_list_cam[:,3] + x_L#x2
    Y2 = sin(theta)*line_list_cam[:,2] + cos(theta)*line_list_cam[:,3] + y_L#y2
    line_list = np.column_stack((X1, Y1, X2, Y2)).astype(int)

    A = line_list[:,1] - line_list[:,3]
    B = line_list[:,2] - line_list[:,0]
    C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]
    r = np.divide(np.abs(C), np.sqrt(A*A+B*B))
    alpha = (np.arctan2(-B*C,-A*C) + pi) % (2*pi) - pi
    obs_inworld_list = np.column_stack((r, alpha)) #an array of observed lines in (r,alpha) form to world frame

    # For every detected cylinder which has a closest matching pole in the
    # cylinders that are part of the current state, put the measurement
    # (distance, angle) and the corresponding cylinder index into the result list.
    result = []

    line_in_state = np.column_stack((robot.state[3::2], robot.state[4::2])) #[[r, alpha],...] from state
    a_matrix = np.cos(line_in_state[:,1])
    b_matrix = np.sin(line_in_state[:,1])
    r_state_in_cam = np.abs(x_L*a_matrix + y_L*b_matrix - line_in_state[:,0])
    xo_matrix = b_matrix*(x_L*b_matrix - y_L*a_matrix) + a_matrix*line_in_state[:,0]
    yo_matrix = a_matrix*(y_L*a_matrix - x_L*b_matrix) + b_matrix*line_in_state[:,0]
    dx_matrix = xo_matrix - x_L
    dy_matrix = yo_matrix - y_L
    alpha_state_in_cam = (np.arctan2(dy_matrix, dx_matrix) - theta + np.pi) % (2*pi) - pi
    ralpha_state_in_cam = np.column_stack((r_state_in_cam, alpha_state_in_cam))

    get_close_lines(robot)

    for line_i in range(len(obs_inworld_list)):
        
        '''
        compare = np.abs(line_in_state - obs_inworld_list[line_i])
        index_list = np.nonzero(np.logical_and(compare[:,0] < local_r_distance, compare[:,1] < local_alpha_distance)) #index of state lines close to observed line
        '''
        compare = np.abs(ralpha_state_in_cam - measurement[line_i])
        index_list = np.nonzero(np.logical_and(compare[:,0] < max_r_distance, compare[:,1] < max_alpha_distance)) #index of state lines close to observed line
        
        obs_r, obs_alpha = measurement[line_i]

        #check if observed line and state lines are overlap
        best_index = -1
        best_distance = max_r_distance * max_r_distance
        best_endpoints = np.array([-1,-1,-1,-1])
        for line_index in index_list[0]:
            #print("state.lines", robot.lines[line_index])
            #print("state.r,alpha", robot.state[3+2*line_index:5+2*line_index])
            overlap, endpoints = check_overlap(robot.lines[line_index], line_list[line_i])
            if overlap:
                state_r, state_alpha = ralpha_state_in_cam[line_index]
                dist_2 = (state_r - obs_r)**2 + ((max_r_distance/max_alpha_distance)*(state_alpha - obs_alpha))**2
                if dist_2 < best_distance:
                    best_distance = dist_2
                    best_index = line_index
                    best_endpoints = endpoints

        result.append((measurement[line_i], line_list[line_i], best_endpoints, best_index))

    return result
