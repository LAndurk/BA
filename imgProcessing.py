import cv2
import numpy as np
import helperFunctions as hf


def get_segmented_img(img, thresh_type, adaptive_thresh, para_smoothing,
					  para_threshold, para_open, para_close):

    try:  # 3 channel pic as input
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:  # 1 channel pic as input
        img_gray = img

    # Smoothing
    img_smooth = cv2.GaussianBlur(img_gray, (para_smoothing, para_smoothing), 0)

    # Threshold
    if not adaptive_thresh:  # global threshold
        if thresh_type < 0:  # inverted
            img_smooth = cv2.bitwise_not(img_smooth)
        ret, img_thresh = cv2.threshold(img_smooth, 0, 255, cv2.THRESH_OTSU)
    else:  # dynamic threshold
        if thresh_type < 0:  # inverted
            thresh_opencv = cv2.THRESH_BINARY_INV
        else:
            thresh_opencv = cv2.THRESH_BINARY
        img_thresh = cv2.adaptiveThreshold(img_smooth, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresh_opencv,
                                           para_threshold, 0)

    # Opening
    kernel_open = np.ones((para_open, para_open), np.uint8)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel_open)
    # Closing
    kernel_close = np.ones((para_close, para_close), np.uint8)
    img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel_close)

    # show pictures
    numpy_horizontal_concat = np.concatenate((img_smooth, img_thresh, img_open,
											  img_close), axis=1)
    hf.resize_and_show("all img proc steps", numpy_horizontal_concat, 2)

    return img_close


def detect_damage(segmented_img):

    # initialize result variable
    result = True

    # make 3 channel picture out of 1 channel for coloured output
    img_output = cv2.cvtColor(segmented_img, cv2.COLOR_GRAY2BGR)

    # size of contours, approximation of contours, ignored border area
    min_contour_size = 2000
    approx_contour = 30
    ignored_border = 150

    pixel_per_axis = (segmented_img.shape[0] + segmented_img.shape[1]) / 2
    scale = pixel_per_axis / 2250
    print(scale)

    min_contour_size *= scale
    approx_contour *= scale
    ignored_border *= scale

    # find contours
    contours, hierarchy = cv2.findContours(segmented_img, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)

    # loop through all contours bigger than minimum size
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > min_contour_size:

            # draw contours
            cv2.drawContours(img_output, contours[i], -1, (0, 255, 0), 2)

            # calculate and draw approximation
            approx = cv2.approxPolyDP(contours[i], approx_contour, True)
            cv2.drawContours(img_output, [approx], -1, (255, 0, 0), 2)

            # count points at border
            points_at_border = 0
            for point in approx:
                y = point[0, 0]
                x = point[0, 1]
                if x < ignored_border or 
				   x > (img_output.shape[0] - ignored_border) or 
				   y < ignored_border or 
				   y > (img_output.shape[1] - ignored_border):
				   
                    points_at_border += 1

            # set result variable
            if points_at_border < 4:
                result = False

    return result, img_output
