import cv2


def read_video(video_path, sample_rate=1):
    """
    Reads video from the specified path and extracts frames at a given sample rate.

    Args:
    - video_path (str): Path to the input video file.
    - sample_rate (int): Interval in seconds for extracting frames.

    Returns:
    - frames (list of tuples): List of (frame_count, frame) pairs for each extracted frame.
    - fps (float): Frames per second of the input video.
    """
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = []
    
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append((frame_count, frame))
        frame_count += 1
    
    video_capture.release()
    return frames, fps


def save_video(output_path, frames, fps):
    """
    Saves a series of frames to a video file.

    Args:
    - output_path (str): Path to save the resultant video.
    - frames (list of numpy arrays): List of processed frames.
    - fps (float): Frames per second for the output video.
    """
    if not frames:
        raise ValueError("No frames to save.")

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
