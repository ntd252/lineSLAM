#Floor segmentation using line detection

import warnings
import numpy as np
from math import pi, exp, sqrt
import cv2
from skimage.draw import line

#from matplotlib import pyplot as plt
from timeit import default_timer as timer

#draw line on black image for (n,1,4) array
def draw_white_lines(img, lines, color=255, thickness=5):
    
    if lines is None:
        return
    for line in lines: 
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

#draw line on black image for (n,4) array
def draw_white_2d(img, lines, color=255, thickness=1):
    
    if lines is None:
        return
    for line in lines: 
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return img

#get vertical line list from hough lines result
def vertical_line(lines):
    filter_index = np.nonzero(np.logical_and(lines[:,0] < 630, lines[:,2] < 630))
    a = lines[filter_index]
    x = np.fabs(a[:,0] - a[:,2])
    y = np.fabs(a[:,1] - a[:,3])
    tan = np.divide(y,x)
    vertical_list = a[(tan==np.inf) | (tan>=6.0)] #>80degrees
    line_image = np.zeros((480, 640), dtype=np.uint8)
    '''
    draw_white_2d(line_image, vertical_list)
    cv2.imshow("vertical_line", line_image)
    '''
    return vertical_list

#get horizontal line list from hough lines result
def horizontal_line(lines):
    filter_index = np.nonzero(np.logical_and(lines[:,1] > 90, lines[:,3] > 90))
    a = lines[filter_index]
    x = np.fabs(a[:,0] - a[:,2])
    y = np.fabs(a[:,1] - a[:,3])
    tan = np.divide(y,x)
    horizontal_list = a[(tan!=np.inf) & (tan<=6.0)] #<82degrees
    line_image = np.zeros((480, 640), dtype=np.uint8)
    '''
    draw_white_2d(line_image, horizontal_list)
    cv2.imshow("horizontal_line", line_image)
    '''
    return horizontal_list

#perform line detection using Hough transform
def line_Hough(image, blur_image):
    #blur_image = cv2.GaussianBlur(gray,(5,5),5)
    #blur_image = cv2.bilateralFilter(gray,7,40,40) #reduce noise and increse edge sharp
    canny = cv2.Canny(blur_image, 20, 55)
    #general lines from normal gray image
    lines = cv2.HoughLinesP(
        canny,
        rho = 1,
        theta = np.pi/180,
        threshold = 60,
        lines = np.array([]),
        minLineLength = 30,
        maxLineGap = 20
    )

    #detect lines from s channel is HSV color space
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1]
    ret,th_s = cv2.threshold(s, 110, 255, cv2.THRESH_BINARY)

    canny_s = cv2.Canny(th_s, 30, 90)

    lines_s = cv2.HoughLinesP(
        canny_s,
        rho = 1,
        theta = np.pi/180,
        threshold = 60,
        lines = np.array([]),
        minLineLength = 30,
        maxLineGap = 20
    )
    
    if lines_s is not None and lines is not None:
        line_list = lines #np.concatenate((lines, lines_s))
    else:
        line_list = lines
    '''
    line_list = lines
    
    line_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    draw_white_lines(
        line_image,
        line_list,
        thickness=1
    )
    cv2.imshow("raw_line", line_image)
    
    #remove un-necessary lines in image border area
    a = line_list[:,0]
    x = np.fabs(a[:,0] - a[:,2])
    y = np.fabs(a[:,1] - a[:,3])
    tan = np.divide(y,x)
    vertical_list = (tan==np.inf) | (tan>=1.2) 
    invalid_range1 = (a[:,1] > image.shape[0] * 0.85) & (a[:,0] > image.shape[1]*0.35) & (a[:,0] < image.shape[1] * 0.7) #point in invalid range
    invalid_range2 = (a[:,3] > image.shape[0] * 0.85) & (a[:,2] > image.shape[1]*0.35) & (a[:,2] < image.shape[1] * 0.7) #point in invalid range
    
    invalid_range = np.logical_or(invalid_range1,invalid_range2)
    invalid_line = np.logical_and(vertical_list, invalid_range)

    valid_line = a[~invalid_line]
    '''
    filterd_line = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
    draw_white_2d(filterd_line,valid_line,thickness=1)    
    cv2.imshow("filtered_line", filterd_line)
    '''
    #return np.array([filterd_line])
    return valid_line

#convert line coordinates in image to camera coordinate
def bird_view(line_list):
    #line_example = np.array([[18,189, 315, 162],[315,162,613,238]])
    line_final = np.empty((0,4))
    #final = np.zeros((2000,3000),dtype=np.uint8)

    dmax = 2500
    range_max = 1500
    pts = np.array([(1145, 746), (1573, 746), (1624, 870), (1110, 870)], dtype='float32')
    dst = np.array([(range_max-47, dmax-547), (range_max+47, dmax-547), (range_max+47, dmax-453), (range_max-47, dmax-453)], dtype='float32')

    matrix = cv2.getPerspectiveTransform(pts, dst)

    k = 2592 / 640
    for line in line_list:
        line_point1 = np.array([[[line[0],line[1]]]], dtype=float) * k
        line_point2 = np.array([[[line[2],line[3]]]],dtype=float)* k

        point1 = cv2.perspectiveTransform(line_point1, matrix)
        point2 = cv2.perspectiveTransform(line_point2, matrix)

        transformed_line = np.hstack((point1[0,0,:], point2[0,0,:]))
        line_final = np.append(line_final, [transformed_line], axis=0)
        
    int_line = line_final.astype(int)
    x1 = dmax - int_line[:,1]
    y1 = range_max - int_line[:,0]
    x2 = dmax - int_line[:,3]
    y2 = range_max - int_line[:,2]
    return np.column_stack((x1,y1,x2,y2))
   
#calculate transform matrix using cali_border.jpg   
def birdview_test():
    image = cv2.imread("Media/cali_border.jpg")
    dmax = 2000
    range_max = 1000
    pts = np.array([(1145, 746), (1573, 746), (1624, 870), (1110, 870)], dtype='float32')
    dst = np.array([(range_max-47, dmax-547), (range_max+47, dmax-547), (range_max+47, dmax-453), (range_max-47, dmax-453)], dtype='float32')

    matrix = cv2.getPerspectiveTransform(pts, dst)

    test_point = np.array([[[241,184],[401,24]]], dtype=float)
    test_point *= 4.05
    warped_test_point = cv2.perspectiveTransform(test_point, matrix)
    print("perspective result",warped_test_point)
    '''
    warped = cv2.warpPerspective(image, matrix,(1800,2500))  
    cv2.imwrite("Media/warped_cali.jpg", warped)
    warped = cv2.resize(warped, (0,0), fx=0.5, fy=0.5)
    cv2.imshow("warped", warped)
    '''

#calculate threshold from Otsu method and average with local minumum
def thres_otsu(image):
    #gray image for input
    blur = cv2.GaussianBlur(image,(5,5),0)
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])
    
    hist_norm = hist.ravel()/hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    hist_smooth = smooth(hist[:,0],10,'hanning')
            #find min of smooth_hist arround custom_th value with window size
    window_size = 20
    index_start = thresh - window_size
    index_stop = thresh + window_size
    slice = hist_smooth[0 if index_start< 0 else index_start:255 if index_stop > 255 else index_stop]
    index_min = np.where(slice == slice.min())
    return int(0.75*thresh + 0.25*index_min[0][0])

#calculate threshold from Otsu method and average with local minumum
def thres_otsu2(blur):
    #gray image for input
    thresh,th0 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    hist = cv2.calcHist([blur],[0],None,[256],[0,256])

    hist_smooth = smooth(hist[:,0],10,'hanning')
            #find min of smooth_hist arround custom_th value with window size=-30 -> 10
    index_start = int(thresh) - 30
    index_stop = int(thresh) + 10
    slice = hist_smooth[0 if index_start< 0 else index_start:255 if index_stop > 255 else index_stop:2]

    index_min = np.argmin(slice)*2 + index_start

    return int(0.7*thresh + 0.3*index_min)

#Generate distance image to calculate structure score
def structure_score(image, gray, blur):
    wall_darker = True
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    '''
    #using custom otsu function
    thres_value = thres_otsu(gray)

    ret,th1 = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)
    '''

    #using built in function for testing speed
    #blur = cv2.GaussianBlur(gray,(5,5),0)
    thres_value = thres_otsu2(blur)
    ret,th1 = cv2.threshold(gray, thres_value, 255, cv2.THRESH_BINARY)
    #cv2.imshow("structure_th", th1)

    #if wall is darker than floor then invert the image
    if wall_darker is not True:
        invert_th1 = cv2.bitwise_not(th1)
    invert = cv2.bitwise_not(cv2.Canny(th1,20,60))
    distance_image = cv2.distanceTransform(invert, cv2.DIST_L2, 3)
    cv2.normalize(distance_image, distance_image, 0, 100.0, cv2.NORM_MINMAX)
    #cv2.imshow("structure distance", distance_image)
    return distance_image
        
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[0:256]

#Merge nearby vertical lines
def merge_vertical(frame, line_list, r_size, alpha_size, overlap_size):
    x_center = frame.shape[1] / 2
    y_center = frame.shape[0] / 2
    A = line_list[:,1] - line_list[:,3]
    B = line_list[:,2] - line_list[:,0]
    C = (line_list[:,0]-x_center)*(line_list[:,3]-y_center) - (line_list[:,2]-x_center)*(line_list[:,1]-y_center)
    r = np.divide(np.abs(C), np.sqrt(A*A+B*B))
    alpha = (np.arctan2(-B*C,-A*C) + pi) % (2*pi) - pi
    r_alpha = np.column_stack((r, alpha))

    merged = np.zeros(len(r_alpha), dtype=np.uint8)
    line_group = np.empty((0,4), dtype=np.int32)
    group_count = 0
    #print("full list", line_list)
    #print("r_alpha", r_alpha)
    for line_index in range(len(r_alpha)):
        if merged[line_index] == 0: #if line hasn't been merged yet
            line_group = np.append(line_group, [line_list[line_index]], axis=0)
 
            for line_index2 in range(line_index+1,len(r_alpha)):
                if merged[line_index2] == 0:
                    dr = abs(r_alpha[line_index,0] - r_alpha[line_index2,0])
                    dalpha = abs(r_alpha[line_index,1] - r_alpha[line_index2,1])
                    #print("dr", line_index, line_index2, dr)
                    #print("dalpha", line_index, line_index2, dalpha)
                    if (dr<r_size) and (dalpha<alpha_size):
                        #if lines are close, check if they are overlap
                        #print("line_list", line_index, line_list[line_index])
                        #print("line_index2", line_index2, line_list[line_index2])
                        overlap, endpoints = check_overlap(line_group[group_count], line_list[line_index2], overlap_size)
                        if overlap:
                            #print("line_group before", line_group)
                            line_group[group_count] = endpoints
                            merged[line_index2] = 1
                            #print("line_group after", line_group)
            merged[line_index] = 1
            group_count += 1
    return line_group

#Generate distance image to calculate bottom score.
def bottom_score(image, vertical_line_list):
    #a = vertical_line_list[:,0] #if vertical_line_list is an (n,1,4) array like hough_list
    #a = vertical_line_list
    a0 = merge_vertical(image, vertical_line_list, r_size=30, alpha_size=0.15, overlap_size=-100)
    a = merge_vertical(image, a0, r_size=30, alpha_size=0.16, overlap_size=-50)
    y1_y2 = np.array([a[:,1],a[:,3]]) #get y1 y2 out of the line list

    index_max_y1_y2 = np.argmax(y1_y2, axis=0) #0 for y1, 1 for y2

    index_max_y1_y2[index_max_y1_y2 == 1] += 2 #get index for x2
    index_max_y1_y2[index_max_y1_y2 == 0] += 1  #get index for x1

    index = np.column_stack((index_max_y1_y2-1, index_max_y1_y2)) #merge index of (x, y)

    a_filter = np.take_along_axis(a,index,1)    #get (x,y) from line list
    sorted = a_filter[a_filter[:,0].argsort()]  #sort (x,y) by x (vertical lines from left to right)
    filter_sort = sorted[sorted[:,1] < (image.shape[0]*0.9)] #filter lines which are in the valid region

    x_eval = np.linspace(0,image.shape[1],41)
    sigma = 8
    delta_x = x_eval[:,None] - filter_sort[:,0]
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, filter_sort[:,1])

    '''
    plt.plot(filter_backup[:,0], filter_backup[:,1],'bo-')
    plt.plot(x_eval,y_eval,'ro-')
    plt.show()
    '''
    
    all_point = np.column_stack((x_eval,y_eval))
    smooth_point = all_point[~np.isnan(all_point).any(axis=1),:] #remove invalid value by weight divided by 0, causes y_eval 'nan' value
    smooth_point = np.int32(smooth_point)
    smooth_bottom = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
    smooth_point = smooth_point.reshape((-1,1,2))
    cv2.polylines(smooth_bottom,[smooth_point],False,(255))
    smooth_bottom[0:10,:] = 0
    if not smooth_bottom.any():
        smooth_bottom[image.shape[0]-1,:] = 255


    #cv2.imshow("smooth bottom", smooth_bottom)

    invert = cv2.bitwise_not(smooth_bottom)
    distance_image = cv2.distanceTransform(invert, cv2.DIST_L2, 3)
    cv2.normalize(distance_image, distance_image, 0, 100.0, cv2.NORM_MINMAX)
    #cv2.imshow("distance of bottom", distance_image)
    
    return distance_image
    
#Generate distance image to calculate homogeneous score
def homo_score(image):
    original_h = image.shape[0]
    original_w = image.shape[1]

    resized_image = cv2.resize(image, (128,96))
    engine = cv2.hfs.HfsSegment_create(96,128)
    engine.setMinRegionSizeI(80)
    engine.setMinRegionSizeII(180)
    # perform segmentation
    # now "res" is a matrix of indices
    # change the second parameter to "True" to get a rgb image for "res"
    res = engine.performSegmentGpu(resized_image, False)

    normalizedImg = np.zeros((96,128),dtype=np.uint8)
    normalizedImg = cv2.normalize(res, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    normalizedImg = np.uint8(normalizedImg)


    h = resized_image.shape[0]
    w = resized_image.shape[1]
    sample_point = np.int32([[0.5*w,0.8*h],[0.35*w,0.85*h],[0.65*w,0.85*h]])
    nearby = np.int32([[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]])
    sample_pixels = sample_point[:,None] + nearby #sample points and 8 nearby pixels
    reshape_sample_pixels = np.reshape(sample_pixels,(27,2)) #total 27 sample points

    floor_value_sample = normalizedImg[reshape_sample_pixels[:,1],reshape_sample_pixels[:,0]]
    floor_value = np.bincount(floor_value_sample).argmax() #get most frequency value of floor

    floor_image = np.zeros(normalizedImg.shape, dtype=np.uint8)
    floor_image[np.where(normalizedImg==floor_value)] = 255
    floor_upscale = cv2.resize(floor_image, (original_w,original_h), interpolation = cv2.INTER_NEAREST) #affect computational speed 
    #CV_INTER_NN (default, fastest)
    #CV_INTER_LINEAR (slower) 
    # CV_INTER_CUBIC (slowest)

    #cv2.imshow("floor_image", floor_upscale)


    invert = cv2.bitwise_not(cv2.Canny(floor_upscale,20,60))
    distance_image = cv2.distanceTransform(invert, cv2.DIST_L2, 3)
    cv2.normalize(distance_image, distance_image, 0, 100.0, cv2.NORM_MINMAX)
    return distance_image

#Check if 2 lines are overlap
def check_overlap(line1, line2, overlap_size):
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
    
    #find the fitting line x = by + c through 4 points
    points_x = np.array([line1[0], line1[2], line2[0], line2[2]])
    points_y = np.array([line1[1], line1[3], line2[1], line2[3]])
    matrix = np.vstack([points_y, np.ones(len(points_x))]).T
    b, c = np.linalg.lstsq(matrix, points_x, rcond=None)[0]
    a = -1
    x1 = (b*(b*endpoint[0] - a*endpoint[1]) - a*c)/(1+b*b)
    y1 = (a*(a*endpoint[1] - b*endpoint[0]) - b*c)/(1+b*b)
    x2 = (b*(b*endpoint[2] - a*endpoint[3]) - a*c)/(1+b*b)
    y2 = (a*(a*endpoint[3] - b*endpoint[2]) - b*c)/(1+b*b)
    endpoint = np.array([x1, y1, x2, y2]).astype(int)

    return (overlap >= overlap_size), endpoint #replace 0 with the value of distance between 2 collinear lines

#General line merging function (for both horizontal and vertical lines)
def mergeLine(line_list, r_size, alpha_size, overlap_size):
    A = line_list[:,1] - line_list[:,3]
    B = line_list[:,2] - line_list[:,0]
    C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]
    r = np.divide(np.abs(C), np.sqrt(A*A+B*B))
    alpha = (np.arctan2(-B*C,-A*C) + pi) % (2*pi) - pi
    r_alpha = np.column_stack((r, alpha))

    merged = np.zeros(len(r_alpha), dtype=np.uint8)
    line_group = np.empty((0,4), dtype=np.int32)
    group_count = 0
    #print("full list", line_list)
    #print("r_alpha", r_alpha)
    for line_index in range(len(r_alpha)):
        if merged[line_index] == 0: #if line hasn't been merged yet
            line_group = np.append(line_group, [line_list[line_index]], axis=0)
 
            for line_index2 in range(line_index+1,len(r_alpha)):
                if merged[line_index2] == 0:
                    dr = abs(r_alpha[line_index,0] - r_alpha[line_index2,0])
                    dalpha = abs(r_alpha[line_index,1] - r_alpha[line_index2,1])
                    #print("dr", line_index, line_index2, dr)
                    #print("dalpha", line_index, line_index2, dalpha)
                    if (dr<r_size) and (dalpha<alpha_size):
                        #if lines are close, check if they are overlap
                        #print("line_list", line_index, line_list[line_index])
                        #print("line_index2", line_index2, line_list[line_index2])
                        overlap, endpoints = check_overlap(line_group[group_count], line_list[line_index2], overlap_size)
                        if overlap:
                            #print("line_group before", line_group)
                            line_group[group_count] = endpoints
                            merged[line_index2] = 1
                            #print("line_group after", line_group)
            merged[line_index] = 1
            group_count += 1
    return line_group

#Filter too short lines
def length_filter(line_list, min_dist):
    dx = line_list[:,0] - line_list[:,2]
    dy = line_list[:,1] - line_list[:,3]
    length = dx*dx + dy*dy
    length = np.sqrt(length)
    long_line = line_list[np.nonzero(length >= min_dist)]
    return long_line

#Perform line detection and calculate score for each line
def pose_estimate(frame):
    #frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),4)

    line_list = line_Hough(frame,blur)
    
    vertical_list = vertical_line(line_list)
    horizontal_list = horizontal_line(line_list)

    dist1 = structure_score(frame,gray,blur)
    dist2 = homo_score(frame)
    dist3 = bottom_score(frame, vertical_list)
    '''
    dist_file = open('dist_file.txt', 'w')
    for each_row in range(dist2.shape[0]):
        for each_col in range(dist2.shape[1]):
            dist_file.write(str(dist2[each_row,each_col]))
            dist_file.write(" ")
        dist_file.write("\n")
    dist_file.close()
    '''
    count = 0

    structure = 0.0
    homo = 0.0
    bottom = 0.0

    sig_struct = 1.0 #0.241712146
    sig_homo = 10.0 #4.154150685
    sig_bottom = 18.16686174
    total = 0.0
    final_list = np.empty((0,4), dtype=np.int32)
    #fhandle = open('score_log.txt', 'a')
    
    for hline in horizontal_list:

        rr, cc = line(hline[1], hline[0], hline[3], hline[2])
        rr_step2 = rr[::2]
        cc_step2 = cc[::2]
        count = len(rr_step2)
        structure = dist1[rr_step2, cc_step2].sum()/count
        homo = dist2[rr_step2, cc_step2].sum()/count
        bottom = dist3[rr_step2, cc_step2].sum()/count

        structure = exp(-structure*structure/(2*sig_struct*sig_struct))/sqrt(2*pi)
        homo = exp(-homo*homo/(2*sig_homo*sig_homo))/sqrt(2*pi)
        bottom = exp(-bottom*bottom/(2*sig_bottom*sig_bottom))/sqrt(2*pi)

        total = structure*0.3 + homo*0.5 + bottom*0.2
        
        if total >= 0.290:
            final_list = np.append(final_list, [hline], axis=0)
 
        '''
        fhandle.write("F{} ".format(frame_count))
        fhandle.write("{} {} {} {}\n".format(hline, structure, homo, bottom))
        
        struct_color = 255 - int(structure/100*255)
        bottom_color = 255 - int(bottom/100*255)
        homo_color = 255 - int(homo/100*255)

        '''
    
    '''
    struct_image = np.zeros((frame.shape[0],frame.shape[1], 3),dtype=np.uint8)
    homo_image = np.zeros((frame.shape[0],frame.shape[1],3),dtype=np.uint8)
    bottom_image = np.zeros((frame.shape[0],frame.shape[1], 3),dtype=np.uint8)
    merged_line = mergeLine(final_list)
    pose_line = bird_view(merged_line, r_size=13, alpha_size=0.115, overlap=-35)

    final_image = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
    draw_white_2d(final_image, final_list)
    cv2.imshow("final_image", final_image)

    merged_image = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
    draw_white_2d(merged_image, merged_line)
    cv2.imshow("merge_image", merged_image)
    '''
    #final_image = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
    #draw_white_2d(final_image, final_list)
    #print("final list", final_list)
    #cv2.imshow("final_line", final_image)
    bird_lines = bird_view(final_list)
    merged_line1 = mergeLine(bird_lines, r_size=30, alpha_size=0.16, overlap_size=-100)
    merged_line = mergeLine(merged_line1, r_size=25, alpha_size=0.15, overlap_size=-90)
    long_line = length_filter(merged_line, 150)
    
    #print("pose_line", pose_line)
    return long_line

#Main function for testing purposes in videos.
if __name__ == '__main__':
    capture = cv2.VideoCapture('../opencv/Media/floor2.mp4')
    frame_count = 0
    while True:
        ret, frame = capture.read()
        if ret is False:
            capture = cv2.VideoCapture('Media/floor2.mp4')
            continue

        frame_count += 1
        if frame_count > 287:
            cv2.waitKey(0)
        start = timer()
        pose_estimate(frame)
        end = timer()
        print("fps",1.0/(end - start))
        

        if cv2.waitKey(50) == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    