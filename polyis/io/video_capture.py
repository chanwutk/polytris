"""Video capture using ffmpeg for reading video frames."""

import subprocess
from typing import Iterator, Optional
import numpy as np


class VideoCapture:
    """Video capture that uses ffmpeg to read video frames.

    This class provides an iterable interface for reading video frames
    from a file using ffmpeg. It decodes videos and yields frames as
    numpy arrays in BGR format (OpenCV compatible).

    Usage:
        # Using as an iterable
        for frame in VideoCapture('input.mp4'):
            process(frame)

        # Using as a context manager
        with VideoCapture('input.mp4') as cap:
            for frame in cap:
                process(frame)

        # Manual control
        cap = VideoCapture('input.mp4')
        cap.open()
        frame = cap.read()
        cap.release()
    """

    def __init__(self, input_path: str):
        """Initialize the video capture.

        Args:
            input_path: Path to the input video file.
        """
        self.input_path = input_path
        self._process: Optional[subprocess.Popen] = None
        self._width, self._height, self._fps, self._frame_count = _probe_video(input_path)
        self._frame_size = self._width * self._height * 3

    def open(self):
        """Open the video file and start the ffmpeg process.

        Raises:
            RuntimeError: If the video is already open or if opening fails.
        """
        if self._process is not None:
            raise RuntimeError("VideoCapture is already open")

        if self._width is None or self._height is None:
            raise RuntimeError("Failed to get video dimensions")

        # Build ffmpeg command to read video and output raw frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', self.input_path,  # Input file
            '-f', 'rawvideo',  # Output format: raw video
            '-pix_fmt', 'bgr24',  # Pixel format (OpenCV uses BGR)
            '-',  # Write to stdout
        ]

        # Start ffmpeg process
        self._process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL
        )

        # Ensure stdout is available
        if self._process.stdout is None:
            raise RuntimeError("FFmpeg stdout pipe not available")

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame from the video.

        Returns:
            Frame as a numpy array in BGR format (height, width, 3) with dtype uint8,
            or None if no more frames are available.

        Raises:
            RuntimeError: If the video is not open or if reading fails.
        """
        assert self._is_opened and self._process is not None, \
            "VideoCapture is not open. Call open() first."

        assert self._process.stdout is not None, \
            "FFmpeg stdout pipe not available"

        # Read frame data from ffmpeg stdout
        raw_frame = self._process.stdout.read(self._frame_size)

        # Check if we've reached the end of the video
        if len(raw_frame) != self._frame_size:
            return None

        # Convert raw bytes to numpy array and reshape to (height, width, 3)
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((self._height, self._width, 3))

        return frame

    def release(self):
        """Release the video capture and close the ffmpeg process."""
        if self._process is None:
            return

        try:
            # Close stdout to stop reading
            if self._process.stdout is not None:
                self._process.stdout.close()

            # Terminate the process
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
        finally:
            # Always close all pipes
            for pipe in [self._process.stdin, self._process.stdout, self._process.stderr]:
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass

            self._process = None

    def __enter__(self) -> "VideoCapture":
        """Enter the context manager.

        Returns:
            Self for use in 'with' statement.
        """
        if self._process is None:
            self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Make the VideoCapture iterable.

        Yields:
            Video frames as numpy arrays in BGR format.
        """
        # Open video if not already open
        if self._process is None:
            self.open()

        try:
            # Yield frames until end of video
            while True:
                frame = self.read()
                if frame is None:
                    break
                yield frame
        finally:
            self.release()

    @property
    def width(self) -> Optional[int]:
        """Get the width of the video frames.

        Returns:
            Width in pixels, or None if not yet opened.
        """
        return self._width

    @property
    def height(self) -> Optional[int]:
        """Get the height of the video frames.

        Returns:
            Height in pixels, or None if not yet opened.
        """
        return self._height

    @property
    def fps(self) -> Optional[float]:
        """Get the frame rate of the video.

        Returns:
            Frames per second, or None if not yet opened.
        """
        return self._fps

    @property
    def frame_count(self) -> Optional[int]:
        """Get the total number of frames in the video.

        Returns:
            Total frame count, or None if not available.
        """
        return self._frame_count

    @property
    def is_opened(self) -> bool:
        """Check if the video is currently open.

        Returns:
            True if the video is open, False otherwise.
        """
        return self._process is not None


def _probe_video(input_path: str):
    """Probe video file to get width, height, fps, and frame count using ffprobe."""
    # Get video stream information using ffprobe
    ffprobe_cmd = [
        'ffprobe',
        '-v', 'error',  # Hide verbose output
        '-select_streams', 'v:0',  # Select first video stream
        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',  # Get dimensions, fps, and frame count
        '-of', 'csv=p=0',  # Output as CSV without header
        input_path,
    ]

    # Run ffprobe and capture output
    result = subprocess.run(
        ffprobe_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )

    # Parse output: width,height,fps_num/fps_den,nb_frames
    output = result.stdout.decode('utf-8').strip()
    parts = output.split(',')

    if len(parts) < 3:
        raise RuntimeError(f"Failed to parse ffprobe output: {output}")

    # Extract width and height
    width = int(parts[0])
    height = int(parts[1])

    # Extract and calculate fps from fraction (e.g., "30000/1001" or "30/1")
    fps_str = parts[2]
    if '/' in fps_str:
        num, den = fps_str.split('/')
        fps = float(num) / float(den)
    else:
        fps = float(fps_str)

    # Extract frame count if available
    if len(parts) >= 4 and parts[3] and parts[3] != 'N/A':
        frame_count = int(parts[3])
    else:
        frame_count = None

    return width, height, fps, frame_count
