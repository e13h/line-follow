import os
import sys
import cv2

import toolbox


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) != 2:
        print(f'USAGE: python {os.path.basename(__file__)} <input_video_file.mp4>')
        exit(-1)
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        raise ValueError(f"{filename!r} is not a valid file.")

    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        print(type(frame))
        if not ret:
            print("Can't receive frame (stream end?). Exiting...")
            break
        output = toolbox.run_cv2_pipeline(frame)
        cv2.imshow('frame', output)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
