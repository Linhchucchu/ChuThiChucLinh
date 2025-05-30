import cv2
import numpy as np
import os
from tqdm import tqdm

VIDEO_PATH = 'sokoban_playback.avi'
OUTPUT_DIR = 'processed_data'
FRAME_SIZE = (84, 84)
STACK_SIZE = 4

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, FRAME_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def max_frame(f1, f2):
    return np.maximum(f1, f2)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    print("[INFO] Extracting and preprocessing frames...")
    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read the video.")
        return []

    prev_processed = preprocess_frame(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame)
        combined = max_frame(prev_processed, processed)
        frames.append(combined)
        prev_processed = processed

    cap.release()
    print(f"[INFO] Total preprocessed frames: {len(frames)}")
    return frames

def stack_frames(frames, stack_size=4):
    print("[INFO] Stacking frames...")
    stacked_states = []
    for i in tqdm(range(stack_size - 1, len(frames))):
        stack = np.stack(frames[i-stack_size+1:i+1], axis=0)  # shape: (4, 84, 84)
        stacked_states.append(stack)
    return stacked_states

def save_states(states, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, state in enumerate(states):
        np.save(os.path.join(output_dir, f"state_{i:04d}.npy"), state)
    print(f"[INFO] Saved {len(states)} states to '{output_dir}'")

def main():
    frames = extract_frames(VIDEO_PATH)
    states = stack_frames(frames, STACK_SIZE)
    save_states(states, OUTPUT_DIR)

if __name__ == "__main__":
    main()
