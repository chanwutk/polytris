import argparse
import cv2
import numpy as np
import os

from scripts.utilities import CACHE_DIR, CLASSIFIERS_TO_TEST, DATA_DIR, format_time, progress_bars

def visualize_first_frame_tiles(video_path: str, output_path: str, tile_size: int):
    """
    Visualizes the tile grid on the first frame of a video.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path where the output image will be saved.
        tile_size (int): The size of the tiles in pixels (e.g., 30, 60, or 120).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    # Get frame dimensions
    height, width, _ = frame.shape

    # Draw horizontal grid lines
    for i in range(0, height, tile_size):
        cv2.line(frame, (0, i), (width, i), (0, 255, 0), 1)

    # Draw vertical grid lines
    for j in range(0, width, tile_size):
        cv2.line(frame, (j, 0), (j, height), (0, 255, 0), 1)

    # Calculate number of tiles in x and y directions
    num_tiles_x = width // tile_size
    num_tiles_y = height // tile_size

    # Add tile numbers to the grid
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the tile number
            tile_number = y * num_tiles_x + x

            # Calculate text position (centered in the tile)
            text_x = int(x * tile_size + tile_size / 2)
            text_y = int(y * tile_size + tile_size / 2)
            
            # Put the text on the frame
            cv2.putText(
                frame,
                str(tile_number),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
    
    # Save the output image
    cv2.imwrite(output_path, frame)
    print(f"Successfully saved tiled frame to {output_path}")

    # Release the video capture object
    cap.release()

def main():
    """
    Parses arguments and calls the visualization function.
    """
    parser = argparse.ArgumentParser(description="Visualize the tile grid on the first frame of a video.")
    parser.add_argument('--tile_size', type=str, choices=['30', '60', '120', 'all'], default='all', help='Tile size to use for classification (or "all" for all tile sizes)')
    parser.add_argument('--dataset', required=False, default='b3d', help='Dataset name')

    args = parser.parse_args()
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)

        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
        output_dir = os.path.join(cache_video_dir, 'tile_grid')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "tile_grid.png")

        visualize_first_frame_tiles(video_file_path, output_path, int(args.tile_size))

if __name__ == "__main__":
    main()