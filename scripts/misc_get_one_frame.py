#!/usr/local/bin/python
"""
Extract a single frame from a video file using OpenCV.

This script allows you to extract a specific frame from a video file
and save it as an image. You can specify the frame number or timestamp.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2


def get_frame_from_video(video_path: str, output_path: str, frame_number: int = None, 
                        timestamp: float = None) -> bool:
    """
    Extract a frame from a video file.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the extracted frame
        frame_number: Frame number to extract (0-based index)
        timestamp: Timestamp in seconds to extract frame at
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return False
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    
    # Determine frame to extract
    if timestamp is not None:
        if timestamp < 0 or timestamp > duration:
            print(f"Error: Timestamp {timestamp}s is outside video duration ({duration:.2f}s)")
            cap.release()
            return False
        frame_number = int(timestamp * fps)
        print(f"Extracting frame at timestamp {timestamp}s (frame {frame_number})")
    elif frame_number is None:
        frame_number = 0
        print(f"Extracting first frame (frame {frame_number})")
    else:
        if frame_number < 0 or frame_number >= total_frames:
            print(f"Error: Frame number {frame_number} is outside valid range (0-{total_frames-1})")
            cap.release()
            return False
        print(f"Extracting frame {frame_number}")
    
    # Seek to the specified frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    if not success:
        print(f"Error: Could not save frame to '{output_path}'")
        cap.release()
        return False
    
    print(f"Successfully saved frame to '{output_path}'")
    cap.release()
    return True


def main():
    """Main function to handle command line arguments and execute frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract a single frame from a video file using OpenCV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first frame
  python misc_get_one_frame.py video.mp4 output.jpg
  
  # Extract frame at specific timestamp
  python misc_get_one_frame.py video.mp4 output.jpg --timestamp 5.5
  
  # Extract specific frame number
  python misc_get_one_frame.py video.mp4 output.jpg --frame 150
        """
    )
    
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("output_path", help="Path to save the extracted frame")
    parser.add_argument("--frame", type=int, help="Frame number to extract (0-based index)")
    parser.add_argument("--timestamp", type=float, help="Timestamp in seconds to extract frame at")
    parser.add_argument("--info", action="store_true", help="Show video information and exit")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.frame is not None and args.timestamp is not None:
        print("Error: Cannot specify both --frame and --timestamp")
        sys.exit(1)
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found")
        sys.exit(1)
    
    # Show video info if requested
    if args.info:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{args.video_path}'")
            sys.exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video Information:")
        print(f"  File: {args.video_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        cap.release()
        return
    
    # Extract the frame
    success = get_frame_from_video(
        args.video_path, 
        args.output_path, 
        args.frame, 
        args.timestamp
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()