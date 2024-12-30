from facenet_pytorch import MTCNN
from torchvision import transforms


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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    face_tensor = preprocess_transform(face_crop)
    
    return face_tensor
