import cv2
def read_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        frames.append(image)
    vidcap.release()
    return frames
