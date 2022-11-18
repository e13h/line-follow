import numpy as np
import cv2
import math


def make_canny(frame, thresh_low = 100, thresh_high = 200):
    # img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    return img_canny

def create_roi(img: np.ndarray, height_factor: float = 0.6, width_factor: float = 0.75) -> np.ndarray:
    img_height, img_width = img.shape[0], img.shape[1]
    bounds = np.array([[
        [0, img_height],
        [img_width, img_height],
        [width_factor * img_width, height_factor * img_height],
        [(1-width_factor) * img_width, height_factor * img_height]]], dtype=np.int32)
    return bounds

def mask_roi(img: np.ndarray, roi: np.ndarray) -> np.ndarray:
    BLACK = (255, 255, 255)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, roi, color=BLACK)
    # return mask
    masked_image = cv2.bitwise_and(img, mask) 
    return masked_image

def draw_lines(img: np.ndarray, lines: np.ndarray, color: tuple = (0, 0, 255), thickness: int = 2):
    mask_lines = np.zeros_like(img)
    if lines is None:
        print('No lines to draw!')
        return mask_lines

    for points in lines:
        x1, y1, x2, y2 = points[0]
        cv2.line(mask_lines, (x1, y1), (x2, y2), color=color, thickness=thickness)
    return mask_lines

def draw_roi(img: np.ndarray, roi: np.ndarray, color: tuple = (0, 0, 255)) -> np.ndarray:
    off_by_1_indices = roi[:, list(range(1, roi.shape[1])) + [0]]
    roi_lines = np.concatenate((roi, off_by_1_indices), axis=2)
    return draw_lines(img, roi_lines.reshape(-1, 1, 4), color=color)

def show_image(name, img):
    cv2.imshow(name,img)
    cv2.waitKey(0)

def get_coordinates(line_parameters: np.ndarray, img_height: int, line_height_factor: float = 0.6):
    slope = line_parameters[0]
    if math.isclose(slope, 0.0, abs_tol=1.0e-7):
        print('Slope is basically 0!')
        return None
    intercept = line_parameters[1]
    y1 = img_height
    y2 = line_height_factor * img_height
    x1 = int((y1-intercept) / slope)
    x2 = int((y2-intercept) / slope)
    coordinates = [x1, int(y1), x2, int(y2)]
    return coordinates

def compute_average_lines(lines, img_shape: tuple):
    left_lane_lines = []
    right_lane_lines = []
    left_weights = []
    right_weights = []
    if lines is None:
        return None, None
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if x2 == x1:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # implementing polyfit to identify slope and intercept
        slope, intercept = parameters
        length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if slope < 0:
            left_lane_lines.append([slope,intercept])
            left_weights.append(length)   
        else:
            right_lane_lines.append([slope,intercept])
            right_weights.append(length)

    left_line = define_line(left_lane_lines, img_shape[0])#, left_weights)
    right_line = define_line(right_lane_lines, img_shape[0])#, right_weights)

    if left_line is not None and right_line is not None:
        return np.array([[left_line], [right_line]]), make_steering_line(left_line, right_line, img_shape)
    elif left_line is not None:
        return np.array([[left_line]]), make_steering_line(left_line, None, img_shape)
    elif right_line is not None:
        return np.array([[right_line]]), make_steering_line(None, right_line, img_shape)
    else:
        return None, None

def define_line(lines: list, img_height: int, weights: list = None) -> np.ndarray:
    if not lines:
        return None
    if weights:
        average_line = np.dot(weights, lines) / np.sum(weights)
    else:
        average_line = np.average(lines, axis=0)
    points = get_coordinates(average_line, img_height)
    return points

def make_steering_line(left_line: list, right_line: list, img_shape: tuple, line_height_factor: float = 0.6):
    if left_line is None:
        average_delta_x = right_line[2] - right_line[0]
    elif right_line is None:
        average_delta_x = left_line[2] - left_line[0]
    else:
        average_delta_x = ((left_line[2]-left_line[0]) + (right_line[2]-right_line[0])) / 2
    y1 = img_shape[0]
    y2 = int(line_height_factor * img_shape[0])
    x1 = int(img_shape[1] / 2)
    x2 = int(x1 + average_delta_x)
    return np.array([[[x1, y1, x2, y2]]])

def steering_command(steering_line: np.ndarray) -> float:
    TUNING_FACTOR = 1.0
    if steering_line is None:
        return 0.0
    x1, _, x2, _ = steering_line.flatten()
    return (x2 - x1) * TUNING_FACTOR

def run_cv2_pipeline(frame: np.ndarray) -> np.ndarray:
    # Colors are in BGR order
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 234, 255)

    img_canny = make_canny(frame)
    roi = create_roi(frame)
    img_masked = mask_roi(img_canny, roi)
    # return img_masked
    hough_lines = cv2.HoughLinesP(img_masked, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=5)
    img_lines = draw_lines(frame, hough_lines, color=BLUE)
    # return img_lines
    lane_lines, steering_line = compute_average_lines(hough_lines, frame.shape)
    img_lanes = draw_lines(frame, lane_lines, color=RED)
    img_steering = draw_lines(frame, steering_line, color=GREEN)
    img_roi = draw_roi(frame, roi, color=YELLOW)
    print(steering_command(steering_line))

    layers = img_lines + img_lanes + img_steering + img_roi
    output = cv2.addWeighted(frame, 0.8, layers, 1, 1)
    return output
