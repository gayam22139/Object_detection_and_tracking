import cv2
from detector import detect_persons

video = cv2.VideoCapture('Videos/walking.avi')

if not video.isOpened():
    print('Error while loading the video!')
    exit()

while True:
    ok, frame = video.read()
    if not ok:
        print('Error reading the video frame or end of video!')
        break
    
    bboxes, scores = detect_persons(frame)
    print(f"Detected bounding boxes: {bboxes}")
    
    for (x, y, x2, y2) in bboxes:
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Detection Test', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
