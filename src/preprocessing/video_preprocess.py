import torch
from torchvision import transforms
from transformers import BertTokenizer
from PIL import Image
import cv2

# Constants
FRAME_SIZE = (224, 224)

# Initialize Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def extract_frames(video_path, frame_rate=1):
    """Extract frames from the video at the given frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(int(fps / frame_rate), 1)

    count = 0
    success, frame = cap.read()
    while success:
        if count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        success, frame = cap.read()
        count += 1

    cap.release()
    return frames


def preprocess_video(frames):
    """Preprocess video frames."""
    transform = transforms.Compose([
        transforms.Resize(FRAME_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([transform(frame) for frame in frames])


def extract_faces(frames, face_detector):
    """Extract faces from frames using a face detector (e.g., MTCNN)."""
    face_tensors = []
    for frame in frames:
        face = face_detector.detect_faces(frame)
        if face:
            x, y, w, h = face[0]['box']
            cropped_face = frame.crop((x, y, x + w, y + h))
            face_tensors.append(preprocess_video([cropped_face])[0])
    return torch.stack(face_tensors) if face_tensors else torch.empty((0, *FRAME_SIZE))
