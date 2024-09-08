import cv2
import numpy as np

# Load the cascade classifier for full-body detection
detector = cv2.CascadeClassifier('cascade/fullbody.xml')

def detect_persons(image):
    """
    Perform object detection on a single image to detect persons using Haar cascades.
    
    Args:
        image (np.array): The input image/frame.
        
    Returns:
        list: List of bounding boxes and scores for detected persons.
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = detector.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    bboxes = []
    scores = []
    
    # Since Haar cascades do not provide scores, we will not use scores here
    for (x, y, w, h) in detections:
        bboxes.append((x, y, x + w, y + h))
        scores.append(1.0)  # Dummy score, as Haar cascades do not provide confidence scores
    
    return bboxes, scores

# Example usage
if __name__ == '__main__':
    image = cv2.imread('Images/people.jpg')
    bboxes, scores = detect_persons(image)
    
    for (x, y, x2, y2) in bboxes:
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()