import cv2

import toolbox


if __name__ == "__main__":
    cap = cv2.VideoCapture('slow_raw.mp4')
    print(cap.isOpened())
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        overlay, steering_line = toolbox.run_yellow_segmentation_pipeline(frame)
        if steering_line is None:
            overlay, steering_line = toolbox.run_lane_detection_pipeline(frame)

        print(toolbox.steering_command(steering_line))
        output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)
        cv2.imshow('frame', output)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
