import cv2
import torch
from facenet_pytorch import MTCNN
from torchvision import transforms
from PIL import Image
import numpy as np


def extract_frames(video_path, frame_rate=1):
    """
    Extracts frames from a video at a specified frame rate.

    Args:
    - video_path (str): Path to the video file.
    - frame_rate (int): Number of frames to extract per second.

    Returns:
    - list: List of extracted frames as numpy arrays.
    """
    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    video_frames = []

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frames.append(frame)
        frame_count += 1

    video_capture.release()
    return video_frames


def detect_faces(frame):
    """
    Detect faces in a given frame using MTCNN.

    Args:
    - frame (numpy array): The input frame in RGB format.

    Returns:
    - boxes (list): List of bounding boxes for detected faces, each in (x1, y1, x2, y2) format.
    """
    model = MTCNN(keep_all=True)
    boxes, _ = model.detect(frame)
    return boxes


def crop_faces(frame, box):
    """
    Crops faces from the frame based on the provided bounding boxes.

    Args:
    - frame (numpy array): The input frame in RGB format.
    - box (list): List of bounding boxes for detected faces.

    Returns:
    - face_crops (list of numpy arrays): List of cropped face images.
    """
    x1, y1, x2, y2 = map(int, box)
    face_crop = frame[y1:y2, x1:x2]
    return face_crop


def preprocess_face(face_crop, input_size=(224, 224)):
    """
    Preprocesses a single face crop for the emotion recognition model.

    Args:
    - face_crop (numpy.ndarray): The cropped face image in RGB format.

    Returns:
    - torch.Tensor: Preprocessed face tensor ready for model input.
    """
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    face_tensor = preprocess_transform(face_crop)
    return face_tensor


def preprocess_video(video_path, frame_rate=1, input_size=(224, 224)):
    """
    Preprocesses a video to extract and preprocess face tensors for emotion recognition.

    Args:
    - video_path (str): Path to the video file.
    - frame_rate (int): Number of frames to extract per second.
    - input_size (tuple): Desired input size for the model (height, width).

    Returns:
    - torch.Tensor: Batch of preprocessed video tensors.
    """
    video_frames = extract_frames(video_path, frame_rate)

    video_tensors = []
    for frame in video_frames:
        boxes = detect_faces(frame)
        if boxes is None:
            continue
        face_crop = crop_faces(frame, boxes[0]) # TODO: iterate over boxes for multiple faces
        face_tensor = preprocess_face(face_crop, input_size=input_size)
        video_tensors.append(face_tensor)

    if not video_tensors:
        return None  # No faces detected in the video

    video_tensors = torch.stack(video_tensors)
    return video_tensors

def tensor_to_pil(tensor):
    """
    Converts a torch.Tensor to PIL.Image.
    Assumes the tensor is in CHW format and values are normalized between 0 and 1.
    """
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 2, 0).numpy()  # Convert CHW to HWC
    return Image.fromarray((tensor * 255).astype(np.uint8))
