import cv2
import numpy as np

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Invalid name! Available trackers: ')
        for t in tracker_types:
            print(t)
    return tracker

def track_objects(tracker_type, frame, bboxes):
    """
    Initialize trackers for given bounding boxes.
    
    Args:
        tracker_type (str): Type of the tracker.
        frame (np.array): The input frame.
        bboxes (list): List of bounding boxes.
        
    Returns:
        list: List of initialized trackers.
    """
    trackers = []
    
    for bbox in bboxes:
        if bbox[2] > 0 and bbox[3] > 0:  # Ensure non-zero width and height
            tracker = create_tracker_by_name(tracker_type)
            if tracker is not None:
                # Ensure frame is in correct format
                frame = np.asarray(frame, dtype=np.uint8)
                tracker.init(frame, bbox)
                trackers.append(tracker)
        else:
            print(f"Skipping invalid bbox: {bbox}")
    
    return trackers

def update_trackers(trackers, frame):
    """
    Update the trackers with the new frame.
    
    Args:
        trackers (list): List of trackers.
        frame (np.array): The input frame.
        
    Returns:
        list: List of updated bounding boxes.
    """
    bboxes = []
    frame = np.asarray(frame, dtype=np.uint8)  # Ensure frame is in correct format
    for tracker in trackers:
        ok, bbox = tracker.update(frame)
        if ok:
            bboxes.append(bbox)
    return bboxes