import cv2
import depthai as dai

import toolbox


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

            overlay, steering_line = toolbox.run_yellow_segmentation_pipeline(frame)
            if steering_line is None:
                overlay, steering_line = toolbox.run_lane_detection_pipeline(frame)

            print(toolbox.steering_command(steering_line))
            output = cv2.addWeighted(frame, 0.8, overlay, 1, 1)
            cv2.imshow("frame", output)
            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyAllWindows()
