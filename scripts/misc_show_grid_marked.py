import argparse
import cv2
import numpy as np
import os
import torch

from polyis.utilities import CACHE_DIR, DATA_DIR
from polyis.images import padHWC

# hard coded entries below
MANUALLY_INCLUDE = {"jnc00.mp4": [18, 36, 54, 72, 90, 17, 35, 53, 133, 134, 152, 161, 179, 197, 215, 22, 23, 24, 25, 26],
                    "jnc02.mp4": [90, 108, 126, 6, 7, 8, 9, 10, 89, 107, 125, 143, 161, 204, 205, 206, 207, 208, 209],
                    "jnc06.mp4": [18, 36, 54, 72, 90, 108, 126, 7, 8, 9, 10, 11, 12, 53, 71, 89, 107, 125, 143, 161, 203, 204, 205, 206, 207, 208], # good
                    "jnc07.mp4": [109, 110, 111, 112, 72, 90, 108, 6, 7, 8, 9, 10, 11, 12, 107, 125, 210, 204, 205, 206, 207, 208, 209]
                    }

def visualize_first_frame_tiles(video_path: str, output_path: str, tile_size: int):
    """
    Visualizes the tile grid on the first frame of a video, highlighting manually included tiles.

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

    # Pad the frame to be divisible by tile_size, just like in the classification script
    frame_tensor = torch.from_numpy(frame).float()
    padded_frame_tensor = padHWC(frame_tensor, tile_size, tile_size)
    padded_frame = padded_frame_tensor.numpy().astype(np.uint8)

    # Get padded dimensions
    height, width, _ = padded_frame.shape

    # Get the list of tiles to mark for the current video
    video_file = os.path.basename(video_path)
    marked_tiles = MANUALLY_INCLUDE.get(video_file, [])

    # Calculate number of tiles in x and y directions from the padded frame
    num_tiles_y = height // tile_size
    num_tiles_x = width // tile_size

    # Add tile numbers and tint to the grid
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate the tile number
            tile_number = y * num_tiles_x + x

            # Apply red tint if the tile is in the marked list
            if tile_number in marked_tiles:
                y_start, y_end = y * tile_size, (y + 1) * tile_size
                x_start, x_end = x * tile_size, (x + 1) * tile_size
                tile_region = padded_frame[y_start:y_end, x_start:x_end]
                
                # Create a red overlay
                overlay = np.zeros_like(tile_region, dtype=np.uint8)
                overlay[:, :] = [0, 0, 255]  # BGR for red
                
                # Blend the overlay with the tile region
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, tile_region, 1 - alpha, 0, tile_region)
                padded_frame[y_start:y_end, x_start:x_end] = tile_region

            # Calculate text position (centered in the tile)
            text_x = int(x * tile_size + tile_size / 2)
            text_y = int(y * tile_size + tile_size / 2)
            
            # Put the text on the frame
            cv2.putText(
                padded_frame,
                str(tile_number),
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0), # Black text for better visibility on tinted background
                2,
                cv2.LINE_AA,
            )

    # Draw grid lines on top of the tinted tiles
    for i in range(0, height, tile_size):
        cv2.line(padded_frame, (0, i), (width, i), (0, 255, 0), 1)
    for j in range(0, width, tile_size):
        cv2.line(padded_frame, (j, 0), (j, height), (0, 255, 0), 1)
    
    # Save the output image
    cv2.imwrite(output_path, padded_frame)
    print(f"Successfully saved tiled frame to {output_path}")

    # Release the video capture object
    cap.release()

def main():
    """Parses arguments and calls the visualization function."""
    parser = argparse.ArgumentParser(description="Visualize the tile grid on the first frame of a video, highlighting marked tiles.")
    parser.add_argument('--tile_size', type=int, choices=[30, 60, 120], default=60, help='Tile size to use for visualization.')
    parser.add_argument('--dataset', required=False, default='b3d', help='Dataset name')

    args = parser.parse_args()
    dataset_dir = os.path.join(DATA_DIR, args.dataset)
    video_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv')) and f in MANUALLY_INCLUDE]

    for video_file in sorted(video_files):
        video_file_path = os.path.join(dataset_dir, video_file)

        cache_video_dir = os.path.join(CACHE_DIR, args.dataset, video_file)
        output_dir = os.path.join(cache_video_dir, 'tile_grid')
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"tile_grid_marked_ts{args.tile_size}.png"
        output_path = os.path.join(output_dir, output_filename)

        visualize_first_frame_tiles(video_file_path, output_path, args.tile_size)

if __name__ == "__main__":
    main()