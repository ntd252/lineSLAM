#Generate path prediction from encoder data

from math import sin, cos, pi
from lego_robot import *
import numpy as np

# This function takes the old (x, y, heading) pose and the motor ticks
# (ticks_left, ticks_right) and returns the new (x, y, heading).
def filter_step(old_pose, motor_ticks, ticks_to_mm_L, ticks_to_mm_R, robot_width,
                scanner_displacement):
    left_tick = motor_ticks[0]*ticks_to_mm_L
    right_tick = motor_ticks[1]*ticks_to_mm_R
    # Find out if there is a turn at all.
    if left_tick == right_tick:
        # No turn. Just drive straight.
        # --->>> Use your previous implementation.
        # Think about if you need to modify your old code due to the
        # scanner displacement?
        theta = old_pose[2]
        x = old_pose[0] + left_tick*cos(theta)
        y = old_pose[1] + left_tick*sin(theta)
        return (x, y, theta)

    else:
        # Turn. Compute alpha, R, etc.

        # --->>> Modify your previous implementation.
        # First modify the the old pose to get the center (because the
        #   old pose is the LiDAR's pose, not the robot's center pose).
        # Second, execute your old code, which implements the motion model
        #   for the center of the robot.
        # Third, modify the result to get back the LiDAR pose from
        #   your computed center. This is the value you have to return.
        theta = old_pose[2]
        x = old_pose[0] - scanner_displacement * cos(theta)
        y = old_pose[1] - scanner_displacement * sin(theta)
        alpha = (right_tick - left_tick)/ robot_width
        R = left_tick / alpha  #approximation with small alpha

        center = np.array([[0.0, 0.0]]).T
        pose = np.array([[x,y]]).T
        theta_matrix = np.array([[sin(theta), -cos(theta)]]).T

        center = pose - (R+robot_width/2)*theta_matrix
        theta = (theta + alpha) % (2*pi)
        theta_matrix = np.array([[sin(theta), -cos(theta)]]).T

        pose = center + (R+robot_width/2)*theta_matrix

        pose = pose + scanner_displacement*np.array([[cos(theta), sin(theta)]]).T
        return (pose[0][0], pose[1][0], theta)

if __name__ == '__main__':
    # Empirically derived distance between scanner and assumed
    # center of robot.
    scanner_displacement = 0.0

    # Empirically derived conversion from ticks to mm.
    ticks_to_mm_L = 10.5896
    ticks_to_mm_R = 10.5896

    # Measured width of the robot (wheel gauge), in mm.
    robot_width = 150.0

    # Measured start position.
    pose = (1700.0, 1000.0, 180.0 / 180.0 * pi)
    #pose = (0.0, 0.0, 0.0 / 180.0 * pi)

    # Read data.
    logfile = LegoLogfile()
    logfile.read("motor_test1.txt")
    #logfile.read("motor05s.txt")

    # Loop over all motor tick records generate filtered position list.
    filtered = []
    for ticks in logfile.motor_ticks:
        pose = filter_step(pose, ticks, ticks_to_mm_L, ticks_to_mm_R, robot_width,
                           scanner_displacement)
        filtered.append(pose)

    # Write all filtered positions to file.
    f = open("poses_from_ticks.txt", "w")
    for pose in filtered:
        f.write("F %f %f %f\n" % pose)
    f.close()
