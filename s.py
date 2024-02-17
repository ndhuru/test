# import the required modules
import cv2
import numpy as np


# define a function to draw the lane lines on the original image
def draw_lane_lines(image, lines):
    # create a blank image with the same shape as the original image
    line_image = np.zeros_like(image)
    # check if there are any lines detected
    if lines is not None:
        # loop over the lines
        for line in lines:
            # get the coordinates of the line endpoints
            x1, y1, x2, y2 = line.reshape(4)
            # draw the line on the blank image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    # return the line image
    return line_image


def draw_centerline(image, lines):
    # create a blank image with the same shape as the original image
    center_image = np.zeros_like(image)
    # check if there are any lines detected
    if lines is not None:
        # initialize lists to store the coordinates of the outer lines
        outer_lines = []
        # loop over the lines
        for line in lines:
            # get the coordinates of the line endpoints
            x1, y1, x2, y2 = line.reshape(4)
            # calculate the slope of the line
            slope = (y2 - y1) / (x2 - x1)
            # ignore the vertical lines
            if np.abs(slope) > 100:  # avoid division by zero
                continue
            # classify the lines into outer lines based on slope magnitude
            outer_lines.append((x1, y1, x2, y2, slope))

        # check if there are enough outer lines to calculate the centerline
        if len(outer_lines) >= 2:
            # sort the outer lines based on slope magnitude
            sorted_outer_lines = sorted(outer_lines, key=lambda x: np.abs(x[4]), reverse=True)
            # take the first two outer lines
            outer1, outer2 = sorted_outer_lines[:2]
            # calculate the midpoint between the endpoints of the outer lines
            mid_x = int((outer1[0] + outer2[0]) / 2)
            mid_y = int((outer1[1] + outer2[1]) / 2)
            # calculate the slope of the centerline
            center_slope = (outer1[4] + outer2[4]) / 2
            # calculate the y-intercept of the centerline using the midpoint
            center_intercept = mid_y - center_slope * mid_x
            try:
                # calculate the endpoints of the centerline based on image height and slope
                height, width = image.shape[:2]
                center_x1 = int((height - center_intercept) / center_slope)
                center_x2 = int((height / 2 - center_intercept) / center_slope)
                # draw the centerline on the blank image
                cv2.line(center_image, (center_x1, height), (center_x2, int(height / 2)), (0, 255, 0), 10)
            except OverflowError:
                # handle overflow error gracefully
                pass
    # return the center image
    return center_image


# define a function to mask the region of interest on the image
def region_of_interest(image):
    # get the height and width of the image
    height, width = image.shape[:2]
    # define the vertices of the square to mask
    top_left = (int(width * 0.25), int(height * 0.25))
    bottom_right = (int(width * 0.75), int(height * 0.75))
    # create a blank image with the same shape as the original image
    mask = np.zeros_like(image)
    # fill the square with white color
    cv2.rectangle(mask, top_left, bottom_right, (255, 255, 255), -1)
    # apply bitwise and operation to the original image and the mask
    masked_image = cv2.bitwise_and(image, mask)
    # draw the region of interest outline on the image
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 2)
    # return the masked image
    return masked_image


# capture the video from the camera
cap = cv2.VideoCapture(0)

# check if the camera is opened
if not cap.isOpened():
    print("error opening the camera")

# loop until the user presses 'q' or the video ends
while cap.isOpened():
    # read a frame from the video
    ret, frame = cap.read()
    # check if the frame is valid
    if ret:
        # draw region of interest outline on the frame
        region_of_interest(frame)
        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # apply Canny edge detection to find the edges
        canny = cv2.Canny(blur, 50, 150)
        # apply the region of interest mask to the edge image
        cropped = region_of_interest(canny)
        # apply Hough transform to find the lines
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=100)
        # draw the lane lines on the original frame
        line_image = draw_lane_lines(frame, lines)
        # draw the centerline on the original frame
        center_image = draw_centerline(frame, lines)
        # combine the original frame with the line images
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        combo_image = cv2.addWeighted(combo_image, 0.8, center_image, 1, 1)
        # show the result on the screen
        cv2.imshow("result", combo_image)
        # wait for 1 millisecond for user input
        key = cv2.waitKey(1)
        # if the user presses 'q', break the loop
        if key == ord('q'):
            break
    else:
        break

# release the camera and destroy the windows
cap.release()
cv2.destroyAllWindows()



# sources:
# https://geeksforgeeks.org/python-play-a-video-using-opencv/
# https://github.com/adityagandhamal/road-lane-detection/blob/master/detection_on_vid.py
# https://pyimagesearch.com/2021/01/19/image-masking-with-opencv/
# https://www.geeksforgeeks.org/program-find-slope-line/
# https://www.geeksforgeeks.org/python-nested-loops/
