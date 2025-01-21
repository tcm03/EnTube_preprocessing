from datasets import load_dataset
import decord

# Load the dataset in streaming mode
dataset = load_dataset("tcm03/EnTube", split="train", streaming=True)

# Initialize counter and specify the max videos to process
max_videos = 5
video_count = 0

# Stream and process videos
for item in dataset:
    video_reader = item['video']  # Decord VideoReader object
    label = item['label']         # Label for the video

    # Extract frames from the video
    frames = []
    for frame in video_reader:
        frames.append(frame.asnumpy())  # Convert Decord frames to NumPy arrays

    print(f"Processed video {video_count} with label {label}, extracted {len(frames)} frames")

    video_count += 1
    if video_count >= max_videos:
        break
