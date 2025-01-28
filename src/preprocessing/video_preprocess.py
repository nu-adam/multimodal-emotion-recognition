import ray
from insightface.app import FaceAnalysis
import torchvision.transforms as T
import cv2
import torch
import os
from glob import glob


@ray.remote(num_gpus=1)
class FaceDetector:
    """
    Ray Actor for face detection using InsightFace.

    Attributes:
        model (FaceAnalysis): InsightFace model for face detection.
    """
    def __init__(self):
        """
        Initializes the face detector with the InsightFace model.
        """
        self.model = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
        self.model.prepare(ctx_id=0)

    def detect_faces(self, frame):
        """
        Detect faces in a single video frame and return cropped tensors.

        Args:
            frame (np.ndarray): A single video frame in RGB format.

        Returns:
            List[torch.Tensor]: A list of cropped face tensors with shape [3, 224, 224].
        """
        faces = self.model.get(frame)
        tensors = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cropped = frame[y1:y2, x1:x2]
            resized = cv2.resize(cropped, (224, 224))
            tensor = T.ToTensor()(resized)
            tensors.append(tensor)
        return tensors


def process_video(video_path, face_detector_actor):
    """
    Preprocesses a video by extracting frames and detecting faces.

    Args:
        video_path (str): Path to the video file.
        face_detector_actor (ray.actor.ActorHandle): A Ray Actor for face detection.

    Returns:
        torch.Tensor: A batch of face tensors with shape [num_faces, 3, 224, 224].
        If no faces are detected, returns an empty tensor.
    """
    # Step 1: Load video and extract frames (1 frame per second)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % fps == 0:  # Take 1 frame per second
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
    cap.release()

    # Step 2: Perform face detection using the Actor
    batched_faces = []
    for frame in frames:
        detected_tensors = ray.get(face_detector_actor.detect_faces.remote(frame))
        batched_faces.extend(detected_tensors)

    # Step 3: Handle edge case: No faces detected
    if not batched_faces:
        print(f"[WARNING] No faces detected in video: {video_path}")
        return torch.empty(0)  # Return an empty tensor if no faces are detected

    # Step 4: Save batched tensors
    return torch.stack(batched_faces)  # Shape: [num_faces, 3, 224, 224]


@ray.remote
def preprocess_video_task(video_path, face_detector_actor):
    """
    Processes a video file and saves the resulting face tensors.

    Args:
        video_path (str): Path to the video file.
        face_detector_actor (ray.actor.ActorHandle): A Ray Actor for face detection.

    Returns:
        str: Path to the saved tensor file.
    """
    batched_faces = process_video(video_path, face_detector_actor)
    output_path = video_path.replace("raw", "processed").replace(".mp4", ".pth")
    
    # Save only if there are detected faces
    if batched_faces.numel() > 0:  # Check if the tensor is not empty
        torch.save(batched_faces, output_path)
    else:
        print(f"[INFO] No tensors saved for video: {video_path} (No faces detected)")

    return output_path


def get_video_paths(raw_dir):
    """
    Recursively finds all video files in the dataset directory.

    Args:
        raw_dir (str): Path to the raw dataset directory.

    Returns:
        List[str]: A list of paths to video files.
    """
    # return glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
    return glob(os.path.join(raw_dir, "*.mp4"), recursive=True)


def main():
    """
    Main function to preprocess all videos in the dataset using Ray Actors.
    """
    ray.init()  # Initialize Ray

    # Step 1: Start the FaceDetector Actor
    face_detector_actor = FaceDetector.remote()

    # Step 2: Get all video paths
    raw_dir = "data/raw/test/disgust"
    video_paths = get_video_paths(raw_dir)

    # Step 3: Launch preprocessing tasks in parallel
    futures = [preprocess_video_task.remote(video_path, face_detector_actor) for video_path in video_paths]

    # Step 4: Wait for all tasks to complete and get results
    results = ray.get(futures)
    print("All videos processed. Saved to:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
