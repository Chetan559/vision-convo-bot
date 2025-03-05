from ultralytics import YOLO
import cv2
import numpy as np
import webcolors
from Color import closest_color_name  # Import the function from Color.py

def get_color_name(rgb_color):
    try:
        color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        color_name = closest_color_name(rgb_color)
    return color_name

def closest_position(x1, y1, x2, y2, image_width, image_height):
    positions = {
        "Top-Left": (0, 0),
        "Top-Right": (image_width, 0),
        "Bottom-Left": (0, image_height),
        "Bottom-Right": (image_width, image_height),
        "Center": (image_width // 2, image_height // 2)
    }

    object_center_x = (x1 + x2) // 2
    object_center_y = (y1 + y2) // 2
    
    min_distance = float('inf')
    closest_pos_name = None
    for pos_name, pos_coords in positions.items():
        distance = np.sqrt((object_center_x - pos_coords[0]) ** 2 + (object_center_y - pos_coords[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_pos_name = pos_name

    return closest_pos_name

def object_predictor(image):
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Predict on the image
    detection_output = model.predict(source=image, conf=0.25, save=False)

    image_height, image_width, _ = image.shape

    detection_details = []

    for detection in detection_output[0].boxes.data:
        class_id = int(detection[5])
        class_name = model.names[class_id]
        
        x1, y1, x2, y2 = map(int, detection[:4])
        
        exact_position = closest_position(x1, y1, x2, y2, image_width, image_height)
        
        roi = image[y1:y2, x1:x2]
        
        average_color_per_row = np.average(roi, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8(average_color)

        average_color_rgb = average_color[::-1]

        color_name = get_color_name(tuple(average_color_rgb))
        
        rgb_color_list = tuple(int(c) for c in average_color_rgb)
        
        detection_details.append({
            "class_name": class_name,
            "exact_position": exact_position,
            "rgb_color": rgb_color_list,
            "color_name": color_name
        })
        
    return detection_details


