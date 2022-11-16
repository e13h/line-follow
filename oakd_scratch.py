import cv2
import numpy as np
import depthai as dai

import toolbox

RED = (0, 0, 255)  # BGR
GREEN = (0, 255, 0)


if __name__ == "__main__":
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)

    xoutRgb.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(300, 300)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    # Linking
    camRgb.preview.link(xoutRgb.input)

    # Connect to device and start pipeline
    with dai.Device(pipeline, usb2Mode=True) as device:
        print('Connected cameras: ', device.getConnectedCameras())
        print('USB speed: ', device.getUsbSpeed().name)
        if device.getBootloaderVersion() is not None:
            print('Bootloader version: ', device.getBootloaderVersion())

        # Output queue will be used to get the rgb frames from the output defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        while True:
            inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
            frame = inRgb.getCvFrame()  # Retrieve 'bgr' (opencv format) frame
            # print(frame.shape)
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
            cv2.imshow("frame", output)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
