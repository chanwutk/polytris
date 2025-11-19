"""Video writer using ffmpeg for H.264 encoding."""

import subprocess
import numpy as np


class VideoWriter:
    """Video writer that uses ffmpeg to write H.264 encoded videos.
    
    This class provides a context manager interface for writing video frames
    to a file using ffmpeg. It always encodes videos in H.264 format.
    
    Usage:
        with VideoWriter('output.mp4', width=1920, height=1080, fps=30.0) as writer:
            for frame in frames:
                writer.write(frame)
    """
    
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
    ):
        """Initialize the video writer.
           Starts the ffmpeg process in the background.
        
        Args:
            output_path: Path to the output video file.
            width: Width of the video frames in pixels.
            height: Height of the video frames in pixels.
            fps: Frame rate (frames per second) of the output video.
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps

        # Build ffmpeg command to write H.264 video
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',  # Input format: raw video
            '-vcodec', 'rawvideo',  # Input codec
            '-s', f'{self.width}x{self.height}',  # Frame size
            '-pix_fmt', 'bgr24',  # Pixel format (OpenCV uses BGR)
            '-r', str(self.fps),  # Frame rate
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-c:v', 'libx264',  # Output codec: H.264
            '-preset', 'fast',  # Encoding preset
            '-crf', '28',  # Constant rate factor (quality setting)
            '-pix_fmt', 'yuv420p',  # Output pixel format for compatibility
            self.output_path,
        ]
        
        # Start ffmpeg process
        self._process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        
        # Ensure stdin is available (should always be true when stdin=subprocess.PIPE)
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin pipe not available")
        
    def __enter__(self) -> "VideoWriter":
        """Enter the context manager.
        
        Returns:
            Self for use in 'with' statement.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up the ffmpeg process.
        
        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        if self._process is None:
            return
        
        try:
            # Close stdin to signal end of input
            if self._process.stdin is not None:
                self._process.stdin.close()
            
            # Handle interrupts by terminating immediately
            if exc_type is KeyboardInterrupt and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            else:
                # Normal exit: wait and check return code
                self._process.wait()
                if self._process.returncode != 0:
                    error_msg = "FFmpeg encoding failed"
                    if self._process.stderr is not None:
                        try:
                            stderr = self._process.stderr.read().decode('utf-8')
                            error_msg = f"{error_msg}: {stderr}"
                        except (OSError, ValueError):
                            pass
                    raise RuntimeError(f"{error_msg} (return code: {self._process.returncode})")
        finally:
            # Always close all pipes and ensure process is terminated
            for pipe in [self._process.stdin, self._process.stdout, self._process.stderr]:
                if pipe is not None:
                    pipe.close()
            
            # Kill process if still running
            if self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=0.5)
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        self._process.kill()
                        self._process.wait()
                    except OSError:
                        pass
            
            self._process = None
    
    def release(self):
        """Release the video writer and close the ffmpeg process."""
        self.__exit__(None, None, None)
    
    def write(self, frame: np.ndarray):
        """Write a frame to the video.
        
        Args:
            frame: Frame as a numpy array in BGR format (OpenCV format).
                  Expected shape: (height, width, 3) with dtype uint8.
        
        Raises:
            RuntimeError: If the writer is not in an active context or if
                         writing fails.
        """
        assert self._process is not None, "VideoWriter must be used within a 'with' statement"
        assert self._process.stdin is not None, "FFmpeg stdin pipe not available"
        assert frame.shape == (self.height, self.width, 3), \
            f"Frame shape {frame.shape} does not match " \
            f"expected ({self.height}, {self.width}, 3)"
        assert frame.dtype == np.uint8, \
            f"Frame dtype {frame.dtype} does not match " \
            f"expected np.uint8"
        
        # Write frame to ffmpeg stdin (frames must be in BGR format)
        self._process.stdin.write(frame.tobytes())
    
    def __call__(self, frame: np.ndarray):
        """Write a frame to the video.
        
        Args:
            frame: Frame as a numpy array in BGR format (OpenCV format).
                  Expected shape: (height, width, 3) with dtype uint8.
        """
        self.write(frame)
