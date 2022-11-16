import cv2
import numpy as np

import toolbox

RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_canny = toolbox.make_canny(frame)
        # img_masked = toolbox.region_of_interest(img_canny)
        img_masked = img_canny
        hough_lines = cv2.HoughLinesP(img_masked, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=5)
        # img_lines = toolbox.draw_lines(frame, hough_lines)
        lane_lines, steering_line = toolbox.compute_average_lines(hough_lines, frame.shape)
        img_lanes = toolbox.draw_lines(frame, lane_lines, color=RED)
        img_steering = toolbox.draw_lines(frame, steering_line, color=GREEN)
        print(toolbox.steering_command(steering_line))

        output = cv2.addWeighted(frame, 0.8, img_lanes + img_steering, 1, 1)
        cv2.imshow('frame', output)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
