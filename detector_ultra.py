import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model for object detection
model = YOLO('yolov8n.pt')  # Use a pretrained YOLOv8 model, or specify the path to your own model 

def detect_persons(image, conf_threshold=0.5):
    """
    Perform object detection on a single image to detect persons using YOLO with post-processing.
    
    Args:
        image (np.array): The input image/frame.
        conf_threshold (float): The confidence threshold to filter weak detections.
        
    Returns:
        list: List of bounding boxes and scores for detected persons.
    """
    results = model(image)  # Run inference on the image
    
    bboxes = []
    scores = []
    
    # for result in results.xyxy[0]:  # xyxy format gives the bounding box coordinates
    #     x1, y1, x2, y2, score, class_id = result
        
    #     # Filter only for 'person' class detections and apply confidence threshold
    #     if int(class_id) == 0 and score >= conf_threshold:  # Assuming 0 is the class ID for 'person'
    #         bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    #         scores.append(float(score))
    
    # return bboxes, scores
    for result in results:  # Iterate over all detection results
        for box in result.boxes.data.tolist():  # Extract the bounding box data
            x1, y1, x2, y2, score, class_id = box

            # Filter only for 'person' class detections and apply confidence threshold
            if int(class_id) == 0 and score >= conf_threshold:  # Assuming 0 is the class ID for 'person'
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))
                scores.append(float(score))
    
    return bboxes, scores

if __name__ == '__main__':
    image = cv2.imread('Images/people.jpg')
    bboxes, scores = detect_persons(image, conf_threshold=0.5)
    
    for (x, y, x2, y2) in bboxes:
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Score: {scores[bboxes.index((x, y, x2, y2))]:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 