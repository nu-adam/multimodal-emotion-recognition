import cv2


def extract_video_frames(video_path, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate.

    Args:
        video_path (str): Path to the video file.
        frame_rate (int): Number of frames to extract per second.

    Returns:
        list: List of extracted frames as numpy arrays.
    """
    video_frames = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        frame_count += 1

    cap.release()
    return video_frames
