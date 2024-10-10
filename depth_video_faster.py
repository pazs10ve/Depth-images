
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image
from tqdm import tqdm

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device='cuda')

video_path = 'vid2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for _ in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_pil = Image.fromarray(frame_rgb)

    depth = pipe(frame_pil)["depth"]

    depth_uint8 = (depth / np.max(depth) * 255).astype(np.uint8)

    depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

    output = np.hstack((frame, depth_bgr))

    cv2.imshow('Original and Depth Frame', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
