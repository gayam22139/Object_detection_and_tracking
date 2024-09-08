import cv2
from detector import detect_persons
from tracker import track_objects, update_trackers
# from pytube import YouTube
import yt_dlp
import os
import subprocess

from random import randint


def download_youtube_video(youtube_url, output_filename):
    """
    Download a YouTube video to a file using yt-dlp.
    
    Args:
        youtube_url (str): URL of the YouTube video.
        output_filename (str): Path to save the downloaded video.
    """
    # ydl_opts = {
    #     'format': 'bestvideo+bestaudio/best',
    #     'outtmpl': output_filename,
    # }
    ydl_opts = {
        'format' : 'mp4',
        'outtmpl' : output_filename,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    print(f"Video downloaded to {output_filename}")


def manual_reference_detection(frame):
    bboxes = []
    colors = []

    while True:
        bbox = cv2.selectROI('MultiTracker', frame)
        
        if bbox == (0, 0, 0, 0):
            print('No object selected, press Q to quit.')
            break

        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print('Press Q to quit and start tracking')
        print('Press any other key to select the next object')
        
        # Wait for a key press
        k = cv2.waitKey(0) & 0xFF
        print(f"Key pressed: {k}")  # Debug line to check the key code
        if k == ord('q') or k == ord('Q'):  # 'q' or 'Q' to quit
            break
    
    cv2.destroyWindow('MultiTracker')  # Close the ROI selection window
    return bboxes



def process_video(input_video_path, output_video_path, tracker_type='CSRT'):
    """
    Process the input video, apply detection and tracking, and save the output video.
    
    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output video file.
        tracker_type (str): Type of tracker to use.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print('Error while loading the video!')
        return

    ok, frame = cap.read()
    if not ok:
        print('Error reading the video frame!')
        return

    # Perform detection
    # bboxes, scores = detect_persons(frame)
    bboxes = manual_reference_detection(frame)            # *********** manual ***************
    print(f"Detected bounding boxes: {bboxes}")

    # Initialize trackers
    trackers = track_objects(tracker_type, frame, bboxes)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        
        # Update trackers
        bboxes = update_trackers(trackers, frame)
        
        # Draw bounding boxes and IDs
        if bboxes:
            for i, bbox in enumerate(bboxes):
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2)
        else:
            print("No bounding boxes to draw.")
        
        cv2.imshow('MultiTracker', frame)
        if cv2.waitKey(1) & 0XFF == 27: # esc
            break
        
        # Write the processed frame to the output video
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # input_video = 'Videos/walking.avi'
    # output_video = 'output_with_predictions.mp4'
    # process_video(input_video, output_video)

    # yt_link = 'https://www.youtube.com/watch?v=JBoc3w5EKfI'
    # input_vid_1 = 'input_video_1.mp4'
    # download_youtube_video(yt_link, input_vid_1)

    input_video = 'input_video_1.mp4'
    output_video = 'output_with_predictions.mp4'
    process_video(input_video, output_video)


    #*********************  Press Q and then ENTER to exit from selecting *********************