#Main thread of SLAM alg
#   Define SLAM class with calculating function
#   Initilize camera in Raspberry Pi module
#   Process images from camera with floor segmentation alg
#   Perform SLAM matching alg to determine if 2 lines are the same
#   Calculate Extended Kalman Filter for SLAM
#   Generate map image after finishing

from math import sin, cos, pi, atan2, sqrt
import numpy as np
import sys
import cv2
from slam_library import get_observations
from floor_segmentation import pose_estimate

from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

left_count = 0
right_count = 0

def r_function(x, y, theta, r_m, alpha_m, d):
    x_L = x + d*cos(theta)
    y_L = y + d*sin(theta)
    r = abs(x_L*cos(alpha_m) + y_L*sin(alpha_m) - r_m)
    return r

def alpha_function(x, y, theta, r_m, alpha_m, d):
    x_L = x + d*cos(theta)
    y_L = y + d*sin(theta)

    A = cos(alpha_m)
    B = sin(alpha_m)
    x_o = B*(B*x_L - A*y_L) + A*r_m
    y_o = A*(A*y_L - B*x_L) + B*r_m
    dx = x_o - x_L
    dy = y_o - y_L
    alpha = (atan2(dy, dx) - theta + pi) % (2*pi) - pi
    return alpha

class ExtendedKalmanFilterSLAM:
    def __init__(self, state, covariance,
                 robot_width, scanner_displacement,
                 control_motion_factor, control_turn_factor,
                 measurement_distance_stddev, measurement_angle_stddev):
        # The state. This is the core data of the Kalman filter.
        self.state = state
        self.predict_state = []
        self.covariance = covariance

        # Some constants.
        self.robot_width = robot_width
        self.scanner_displacement = scanner_displacement
        self.control_motion_factor = control_motion_factor
        self.control_turn_factor = control_turn_factor
        self.measurement_distance_stddev = measurement_distance_stddev
        self.measurement_angle_stddev = measurement_angle_stddev

        # Currently, the number of landmarks is zero.
        self.number_of_landmarks = 0
        self.lines = np.empty((0,4),dtype=np.int16)
        self.observed_time = np.empty((0,1),dtype=np.int16)
        self.current_close_lines = []

    @staticmethod
    def g(state, control, w):
        x, y, theta = state
        l, r = control
        if r != l:
            alpha = (r - l) / w
            rad = l/alpha
            g1 = x + (rad + w/2.)*(sin(theta+alpha) - sin(theta))
            g2 = y + (rad + w/2.)*(-cos(theta+alpha) + cos(theta))
            g3 = (theta + alpha + pi) % (2*pi) - pi
        else:
            g1 = x + l * cos(theta)
            g2 = y + l * sin(theta)
            g3 = theta

        return np.array([g1, g2, g3])

    @staticmethod
    def dg_dstate(state, control, w):
        theta = state[2]
        l, r = control
        if r != l:
            alpha = (r-l)/w
            theta_ = theta + alpha
            rpw2 = l/alpha + w/2.0
            m = np.array([[1.0, 0.0, rpw2*(cos(theta_) - cos(theta))],
                       [0.0, 1.0, rpw2*(sin(theta_) - sin(theta))],
                       [0.0, 0.0, 1.0]])
        else:
            m = np.array([[1.0, 0.0, -l*sin(theta)],
                       [0.0, 1.0,  l*cos(theta)],
                       [0.0, 0.0,  1.0]])
        return m

    @staticmethod
    def dg_dcontrol(state, control, w):
        theta = state[2]
        l, r = tuple(control)
        if r != l:
            rml = r - l
            rml2 = rml * rml
            theta_ = theta + rml/w
            dg1dl = w*r/rml2*(sin(theta_)-sin(theta))  - (r+l)/(2*rml)*cos(theta_)
            dg2dl = w*r/rml2*(-cos(theta_)+cos(theta)) - (r+l)/(2*rml)*sin(theta_)
            dg1dr = (-w*l)/rml2*(sin(theta_)-sin(theta)) + (r+l)/(2*rml)*cos(theta_)
            dg2dr = (-w*l)/rml2*(-cos(theta_)+cos(theta)) + (r+l)/(2*rml)*sin(theta_)
            
        else:
            dg1dl = 0.5*(cos(theta) + l/w*sin(theta))
            dg2dl = 0.5*(sin(theta) - l/w*cos(theta))
            dg1dr = 0.5*(-l/w*sin(theta) + cos(theta))
            dg2dr = 0.5*(l/w*cos(theta) + sin(theta))

        dg3dl = -1.0/w
        dg3dr = 1.0/w
        m = np.array([[dg1dl, dg1dr], [dg2dl, dg2dr], [dg3dl, dg3dr]])
            
        return m

    def predict(self, control):
        """The prediction step of the Kalman filter."""
        # covariance' = G * covariance * GT + R
        # where R = V * (covariance in control space) * VT.
        # Covariance in control space depends on move distance.
        G3 = self.dg_dstate(self.state, control, self.robot_width)
        left, right = control
        left_var = (self.control_motion_factor * left)**2 +\
                   (self.control_turn_factor * (left-right))**2
        right_var = (self.control_motion_factor * right)**2 +\
                    (self.control_turn_factor * (left-right))**2
        control_covariance = np.diag([left_var, right_var])
        V = self.dg_dcontrol(self.state, control, self.robot_width)
        R3 = V.dot(control_covariance).dot(V.T)

        # --->>> Put here your previous code to compute the new
        #        covariance and state.
        N = self.number_of_landmarks
        G = np.zeros((3+2*N,3+2*N))
        G[0:3,0:3] = G3
        G[3:,3:] = np.eye(2*N)

        R = np.zeros((3+2*N,3+2*N))
        R[0:3,0:3] = R3

        self.covariance = G.dot(self.covariance).dot(G.T) + R
        # state' = g(state, control)
        state3 = self.g(self.state[0:3], control, self.robot_width)
        self.predict_state.append(state3)
        self.state[0:3] = state3

    def add_landmark_to_state(self, line_coords):
        """Enlarge the current state and covariance matrix to include one more
           landmark, which is given by its initial_coords (an (r, alpha) tuple).
           Returns the index of the newly added landmark."""
        #line_coords is the full form of line coordinates (x1, y1, x2, y2)
        
        # --->>> Put here your previous code to augment the robot's state and
        #        covariance matrix.
        index = self.number_of_landmarks
        self.number_of_landmarks += 1

        cov_row, cov_col = self.covariance.shape
        new_covariance = np.zeros((3+2*self.number_of_landmarks,3+2*self.number_of_landmarks), dtype=float)
        new_covariance[0:cov_row,0:cov_row] = self.covariance
        new_covariance[cov_row:,cov_col:] = np.diag([10**10,3.14])
        self.covariance = new_covariance

        A = line_coords[1] - line_coords[3]
        B = line_coords[2] - line_coords[0]
        C = line_coords[0]*line_coords[3] - line_coords[2]*line_coords[1]
        r = abs(C)/sqrt(A*A + B*B)
        alpha = (atan2(-B*C,-A*C) + pi) % (2*pi) - pi
        
        new_state = np.hstack((self.state, [r, alpha]))
        self.state = new_state
        self.lines = np.append(self.lines, [line_coords], axis=0)
        self.observed_time = np.append(self.observed_time,0)

        return index

    @staticmethod
    def h(state, landmark, scanner_displacement):
        """Takes a (x, y, theta) state and a (r, alpha) landmark, and returns the
           measurement (range, bearing) in camera coordinate frame."""
        r = r_function(state[0], state[1], state[2], landmark[0], landmark[1], scanner_displacement)
        alpha = alpha_function(state[0], state[1], state[2], landmark[0], landmark[1], scanner_displacement)

        return np.array([r, alpha])
    
    @staticmethod
    def dh_dstate(state, landmark, scanner_displacement):
        delta = 1e-7
        rx1 = r_function(state[0]+delta, state[1], state[2], landmark[0], landmark[1], scanner_displacement)
        rx2 = r_function(state[0]-delta, state[1], state[2], landmark[0], landmark[1], scanner_displacement)
        dr_x = (rx1 - rx2)/(2*delta)

        ry1 = r_function(state[0], state[1]+delta, state[2], landmark[0], landmark[1], scanner_displacement)
        ry2 = r_function(state[0], state[1]-delta, state[2], landmark[0], landmark[1], scanner_displacement)
        dr_y = (ry1 - ry2)/(2*delta)

        rtheta1 = r_function(state[0], state[1], state[2]+delta, landmark[0], landmark[1], scanner_displacement)
        rtheta2 = r_function(state[0], state[1], state[2]-delta, landmark[0], landmark[1], scanner_displacement)
        dr_theta = (rtheta1 - rtheta2)/(2*delta)

        alphax1 = alpha_function(state[0]+delta, state[1], state[2], landmark[0], landmark[1], scanner_displacement)
        alphax2 = alpha_function(state[0]-delta, state[1], state[2], landmark[0], landmark[1], scanner_displacement)
        dalpha_x = (alphax1 - alphax2)/(2*delta)

        alphay1 = alpha_function(state[0], state[1]+delta, state[2], landmark[0], landmark[1], scanner_displacement)
        alphay2 = alpha_function(state[0], state[1]-delta, state[2], landmark[0], landmark[1], scanner_displacement)
        dalpha_y = (alphay1 - alphay2)/(2*delta)

        alphatheta1 = r_function(state[0], state[1], state[2]+delta, landmark[0], landmark[1], scanner_displacement)
        alphatheta2 = r_function(state[0], state[1], state[2]-delta, landmark[0], landmark[1], scanner_displacement)
        dalpha_theta = (alphatheta1 - alphatheta2)/(2*delta)

        r_rm1 = r_function(state[0], state[1], state[2], landmark[0]+delta, landmark[1], scanner_displacement)
        r_rm2 = r_function(state[0], state[1], state[2], landmark[0]-delta, landmark[1], scanner_displacement)
        dr_rm = (r_rm1 - r_rm2)/(2*delta)
    
        r_alpham1 = r_function(state[0], state[1], state[2], landmark[0], landmark[1]+delta, scanner_displacement)
        r_alpham2 = r_function(state[0], state[1], state[2], landmark[0], landmark[1]-delta, scanner_displacement)
        dr_alpham = (r_alpham1 - r_alpham2)/(2*delta)

        alpha_rm1 = alpha_function(state[0], state[1], state[2], landmark[0]+delta, landmark[1], scanner_displacement)
        alpha_rm2 = alpha_function(state[0], state[1], state[2], landmark[0]-delta, landmark[1], scanner_displacement)
        dalpha_rm = (alpha_rm1 - alpha_rm2)/(2*delta)

        alpha_alpham1 = alpha_function(state[0], state[1], state[2], landmark[0], landmark[1]+delta, scanner_displacement)
        alpha_alpham2 = alpha_function(state[0], state[1], state[2], landmark[0], landmark[1]-delta, scanner_displacement)
        dalpha_alpham = (alpha_alpham1 - alpha_alpham2)/(2*delta)

        H3 = np.array([[dr_x, dr_y, dr_theta],
                      [dalpha_x, dalpha_y, dalpha_theta]])
        
        H_ext = np.array([[dr_rm, dr_alpham],
                          [dalpha_rm, dalpha_alpham]])
        return H3, H_ext

    def correct(self, measurement, landmark_index):
        """The correction step of the Kalman filter."""
        # Get (x_m, y_m) of the landmark from the state vector.

        landmark = self.state[3+2*landmark_index : 3+2*landmark_index+2]
        H3, H_ext = self.dh_dstate(self.state, landmark, self.scanner_displacement)


        #Full H matrix
        H = np.zeros((2,3+2*self.number_of_landmarks))
        H[0:2,0:3] = H3
        H[0:2,3+2*landmark_index:5+2*landmark_index] = H_ext
        
        # This is the old code from the EKF - no modification necessary!
        Q = np.diag([self.measurement_distance_stddev**2,
                  self.measurement_angle_stddev**2])
        K = np.dot(self.covariance, H.T).dot(np.linalg.inv(H.dot(self.covariance).dot(H.T) + Q))

        innovation = np.array(measurement) - self.h(self.state, landmark, self.scanner_displacement)
        innovation[1] = (innovation[1] + pi) % (2*pi) - pi
        self.state = self.state + np.dot(K, innovation)
        self.covariance = np.dot(np.eye(np.size(self.state)) - np.dot(K, H), self.covariance)
        self.observed_time[landmark_index] += 1

    def update_line(self, endpoints, state_line_index):
        #update line endpoints from corrected (r,alpha) in self.state
        A = cos(self.state[3+2*state_line_index+1])
        B = sin(self.state[3+2*state_line_index+1])
        #rm = self.state[3+2*state_line_index]
        x1 = B*(B*endpoints[0] - A*endpoints[1]) + A*self.state[3+2*state_line_index]
        y1 = A*(A*endpoints[1] - B*endpoints[0]) + B*self.state[3+2*state_line_index]
        x2 = B*(B*endpoints[2] - A*endpoints[3]) + A*self.state[3+2*state_line_index]
        y2 = A*(A*endpoints[3] - B*endpoints[2]) + B*self.state[3+2*state_line_index]

        self.lines[state_line_index] = np.array([x1, y1, x2, y2]).astype(int)   

    def delete_wrong_line(self):
        index_of_obs_1 = np.nonzero(self.observed_time == 1)
        index_of_close_1 = np.intersect1d(index_of_obs_1, self.current_close_lines)

        delete_list1 = index_of_close_1*2 + 3
        delete_list2 = index_of_close_1*2 + 4
        delete_list = np.concatenate((delete_list1, delete_list2))

        self.state = np.delete(self.state, delete_list)

        self.covariance = np.delete(self.covariance, delete_list, axis=0)
        self.covariance = np.delete(self.covariance, delete_list, axis=1)

        self.lines = np.delete(self.lines, index_of_close_1, axis=0)
        self.observed_time = np.delete(self.observed_time, index_of_close_1, axis=0)

        self.number_of_landmarks -= len(index_of_close_1)            
    
def left_tick(channel):
    global left_count
    left_count += 1

def right_tick(channel):
    global right_count
    right_count += 1

def interrupt_init():
    LEFT_ENCODER = 11
    RIGHT_ENCODER = 7
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LEFT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RIGHT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(LEFT_ENCODER, GPIO.FALLING, 
        callback=left_tick)
    GPIO.add_event_detect(RIGHT_ENCODER, GPIO.FALLING, 
        callback=right_tick)

if __name__ == '__main__':
    # Robot constants.
    SCANNER_DISPLACEMENT = 150.0
    TICKS_TO_MM_L = 10.5896
    TICKS_TO_MM_R = 10.5896
    ROBOT_WIDTH = 150.0
    left_count_previous = 0
    right_count_previous = 0

    left_encoder_read = 0
    right_encoder_read = 0

    # Cylinder extraction and matching constants.
    max_r_distance = 200.0
    max_alpha_distance = 0.35

    map_size = 8000

    # Filter constants.
    control_motion_factor = 0.35  # Error in motor control.
    control_turn_factor = 0.6  # Additional error due to slip when turning.
    measurement_distance_stddev = 600.0  # Distance measurement error of cylinders.
    measurement_angle_stddev = 45. / 180.0 * pi  # Angle measurement error.

    # Arbitrary start position.
    initial_state = np.array([0.0, 0.0, 0.0])
    # Covariance at start position.
    initial_covariance = np.zeros((3,3))
    

    #camera startup
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    camera.exposure_mode = 'sports'
    rawCapture = PiRGBArray(camera, size=(640, 480))
    sleep(2) # allow the camera to warmup

    #init interrupt for reading odometry
    interrupt_init()
    # Setup filter.
    kf = ExtendedKalmanFilterSLAM(initial_state, initial_covariance,
                                  ROBOT_WIDTH, SCANNER_DISPLACEMENT,
                                  control_motion_factor, control_turn_factor,
                                  measurement_distance_stddev,
                                  measurement_angle_stddev)

    # Read data.
    start = timer()
    encoder_log = open("frame/motor.txt", "w")
    count = 0
    try:
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array

            left_encoder_read = left_count
            right_encoder_read = right_count

            #data logging
            frame_name = "frame/frame" + str(count) + ".jpg"
            cv2.imwrite(frame_name, image)
            count += 1
            encoder_log.write('M {} {}\n'.format(left_encoder_read, right_encoder_read))
            #end data logging

            filtered_line_list = pose_estimate(image)
            # clear the stream in preparation for the next frame
            rawCapture.truncate(0)
            control = np.array([left_encoder_read-left_count_previous, right_encoder_read-right_count_previous]) * [TICKS_TO_MM_L, TICKS_TO_MM_R]
            kf.predict(control)

            # Correction.
            observations = get_observations(filtered_line_list, kf, max_r_distance, max_alpha_distance)
            for obs in observations:
                measurement, line_world, endpoints, state_line_index = obs
                old_state_line_index = state_line_index
                if state_line_index == -1:
                    state_line_index = kf.add_landmark_to_state(line_world)
                kf.correct(measurement, state_line_index)
                if old_state_line_index != -1:
                    kf.update_line(endpoints, state_line_index)
                print("running correction")
            kf.delete_wrong_line()
            print("--------------")  
            
            left_count_previous = left_encoder_read
            right_count_previous = right_encoder_read
            # End of EKF SLAM - from here on, data is written.
            '''
            stop = timer()
            if (stop-start) > 40:
                print("closing...")
                encoder_log.close()
                camera.close()
                GPIO.cleanup()
                print("closed!")
                break
            '''
    except:
        print("closing...")
        encoder_log.close()
        camera.close()
        GPIO.cleanup()
        print("closed!")

    finally:
        print("Saving map image...")
        final_map = np.zeros((map_size,map_size),dtype=np.uint8)
        for wall_index in range(len(kf.lines)):
            if kf.observed_time[wall_index] < 2:
                continue
            int_wall = kf.lines[wall_index].astype(int)

            x1 = int(map_size/2) + int_wall[0]
            y1 = int(map_size/2) - int_wall[1]
            x2 = int(map_size/2) + int_wall[2]
            y2 = int(map_size/2) - int_wall[3]

            cv2.line(final_map, (x1, y1), (x2, y2), color=255, thickness=5)
        cv2.imwrite("map.jpg", final_map)
        print("SLAM is finished!")
        sys.exit()
