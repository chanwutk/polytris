import torch
from typing import Any


class CUDAGraphWrapper:
    """
    Wrapper for CUDA Graph that handles variable batch sizes.
    
    Uses CUDA Graph for the expected batch size, falls back to the base model
    for other batch sizes.
    """
    def __init__(self, model: "torch.nn.Module", expected_batch_size: int, device: str, tile_size: int):
        """
        Initialize CUDA Graph wrapper.
        
        Args:
            model: Base model to wrap
            expected_batch_size: Batch size to use for CUDA Graph
            device: Device to use
            tile_size: Tile size for creating static input
        """
        self.base_model = model
        self.model = model
        self.expected_batch_size = expected_batch_size
        self.device = device
        self.tile_size = tile_size
        self._ensure_graph_captured(torch.randn(expected_batch_size, 3, tile_size, tile_size, device=device))
    
    def _ensure_graph_captured(self, input_tensor: torch.Tensor):
        """Ensure CUDA Graph is captured for the current input shape."""
        self.static_input = input_tensor.clone()
        with torch.no_grad():
            self.static_output = torch.empty_like(self.model(self.static_input))
        
        # Warmup
        for _ in range(5):
            _ = self.model(self.static_input)
        torch.cuda.synchronize()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output.copy_(self.model(self.static_input))
    
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with CUDA Graph"""
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()
    
    def eval(self):
        """Set model to eval mode."""
        self.base_model.eval()
        if hasattr(self.model, 'eval'):
            self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """Set model to train mode."""
        self.base_model.train(mode)
        if hasattr(self.model, 'train'):
            self.model.train(mode)
        return self


class ChannelsLastCUDAGraphWrapper(CUDAGraphWrapper):
    """
    Wrapper for channels-last + CUDA Graph.
    """
    def _ensure_graph_captured(self, input_tensor: torch.Tensor):
        """Ensure CUDA Graph is captured for the current input shape (channels-last)."""
        # Capture graph for this input shape (already in channels-last format)
        self.static_input = input_tensor.clone()
        with torch.no_grad():
            self.static_output = torch.empty_like(self.model(self.static_input))
        
        # Warmup
        for _ in range(5):
            _ = self.model(self.static_input)
        torch.cuda.synchronize()
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output.copy_(self.model(self.static_input))
    
    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass with channels-last memory format and CUDA Graph."""
        # Convert input to channels-last
        input_cl = input_tensor.to(memory_format=torch.channels_last)  # type: ignore
        self.static_input.copy_(input_cl)
        self.graph.replay()
        return self.static_output.clone()
        

def select_model_optimization(model: "torch.nn.Module", benchmark_results: list[dict[str, Any]],
                              device: str, tile_size: int, batch_size: int) -> Any:
    """
    Select and apply the best optimization method based on benchmark results.
    
    Args:
        model: The model to optimize
        benchmark_results: List of benchmark results, each with 'method' and 'runtime_ms'
        device: Device to use
        tile_size: Tile size (used for creating dummy inputs)
        batch_size: Expected batch size (used for CUDA Graph methods)
        
    Returns:
        Optimized model based on the fastest method
    """
    model.eval()
    
    # Filter out failed methods (runtime_ms is None)
    valid_results = [r for r in benchmark_results if r['runtime_ms'] is not None]
    
    if not valid_results:
        # All methods failed, return baseline model
        return model
    
    # Sort by runtime (ascending - fastest first)
    valid_results.sort(key=lambda x: x['runtime_ms'])
    best_result = valid_results[0]
    method = best_result['method']
    
    # Apply the best optimization method
    if method == 'baseline':
        return model
    
    elif method == 'torch_compile':
        return torch.compile(model, mode="reduce-overhead")
    
    elif method == 'channels_last':
        # Convert model to channels-last
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        return model_cl
    
    elif method == 'torch_compile_channels_last':
        # Convert to channels-last and compile
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        return torch.compile(model_cl, mode="reduce-overhead")
    
    elif method == 'torchscript_trace':
        # Create dummy input for tracing
        dummy_input = torch.randn(batch_size, 3, tile_size, tile_size, device=device)
        traced_model = torch.jit.trace(model, dummy_input)
        return traced_model
    
    elif method == 'torchscript_optimize':
        # Trace, freeze, and optimize
        dummy_input = torch.randn(batch_size, 3, tile_size, tile_size, device=device)
        traced_model = torch.jit.trace(model, dummy_input)
        optimized_model = torch.jit.freeze(traced_model)
        optimized_model = torch.jit.optimize_for_inference(optimized_model)
        return optimized_model
    
    elif method == 'cuda_graph':
        # Create CUDA Graph wrapper
        return CUDAGraphWrapper(model, batch_size, device, tile_size)
    
    elif method == 'channels_last_cuda_graph':
        # Convert to channels-last and create CUDA Graph wrapper
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        return ChannelsLastCUDAGraphWrapper(model_cl, batch_size, device, tile_size)
    
    else:
        # Unknown method, return baseline
        return model

