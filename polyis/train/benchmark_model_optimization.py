import torch
import time
import os
import json


def benchmark_model_optimization(model: "torch.nn.Module", device: str, tile_size: int,
                                 batch_size: int, iterations: int = 32):
    """
    Benchmark different acceleration methods and save results to JSONL file.
    
    Tests: baseline, torch.compile, channels-last, torch.compile + channels-last,
           TorchScript Trace, TorchScript + Optimize, CUDA Graph, channels-last + CUDA Graph
    
    Args:
        model: The model to optimize
        device: Device to use
        tile_size: Tile size (used to create dummy input)
        batch_size: Batch size for benchmarking
        iterations: Number of iterations for benchmarking
    """
    model.eval()
    
    # Create dummy input for benchmarking (typical batch size for a frame)
    dummy_input = torch.randn(batch_size, 3, tile_size, tile_size, device=device)
    
    results = []
    
    # 1. Baseline (eager)
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        baseline_time = ((time.time_ns() / 1e6) - start) / iterations
    results.append({
        'method': 'baseline',
        'runtime_ms': baseline_time
    })
    
    # 2. torch.compile
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = compiled_model(dummy_input)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = compiled_model(dummy_input)
            torch.cuda.synchronize()
            compile_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'torch_compile',
            'runtime_ms': compile_time
        })
    except Exception as e:
        results.append({
            'method': 'torch_compile',
            'runtime_ms': None,
            'error': str(e)
        })
    
    # 2a. channels-last (without torch.compile)
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                input_cl = dummy_input.to(memory_format=torch.channels_last)  # type: ignore
                _ = model_cl(input_cl)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                # Include input conversion in timing
                input_cl = dummy_input.to(memory_format=torch.channels_last)  # type: ignore
                _ = model_cl(input_cl)
            torch.cuda.synchronize()
            channels_last_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'channels_last',
            'runtime_ms': channels_last_time
        })
    except Exception as e:
        results.append({
            'method': 'channels_last',
            'runtime_ms': None,
            'error': str(e)
        })
    
    # 2b. torch.compile + channels-last
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        compiled_channels_last_model = torch.compile(model_cl, mode="reduce-overhead")
        dummy_input_cl = dummy_input.to(memory_format=torch.channels_last)  # type: ignore
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = compiled_channels_last_model(dummy_input_cl)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                # Include input conversion in timing
                input_cl = dummy_input.to(memory_format=torch.channels_last)  # type: ignore
                _ = compiled_channels_last_model(input_cl)
            torch.cuda.synchronize()
            compile_cl_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'torch_compile_channels_last',
            'runtime_ms': compile_cl_time
        })
    except Exception as e:
        results.append({
            'method': 'torch_compile_channels_last',
            'runtime_ms': None,
            'error': str(e)
        })
    
    # 3. TorchScript Trace
    traced_model = None
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = traced_model(dummy_input)  # type: ignore
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = traced_model(dummy_input)  # type: ignore
            torch.cuda.synchronize()
            trace_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'torchscript_trace',
            'runtime_ms': trace_time
        })
    except Exception as e:
        results.append({
            'method': 'torchscript_trace',
            'runtime_ms': None,
            'error': str(e)
        })
        traced_model = None
    
    # 4. TorchScript Trace + Optimize
    try:
        if traced_model is None:
            traced_model = torch.jit.trace(model, dummy_input)
        optimized_model = torch.jit.freeze(traced_model)
        optimized_model = torch.jit.optimize_for_inference(optimized_model)
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = optimized_model(dummy_input)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = optimized_model(dummy_input)
            torch.cuda.synchronize()
            optimize_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'torchscript_optimize',
            'runtime_ms': optimize_time
        })
    except Exception as e:
        results.append({
            'method': 'torchscript_optimize',
            'runtime_ms': None,
            'error': str(e)
        })
    
    # 5. CUDA Graph (only if input size is fixed - may not work for variable batch sizes)
    try:
        # CUDA Graph requires fixed input/output sizes
        static_input = dummy_input.clone()
        static_output = torch.empty_like(model(static_input))
        
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(model(static_input))
        
        # Warmup
        for _ in range(5):
            static_input.copy_(dummy_input)
            graph.replay()
        torch.cuda.synchronize()
        
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            static_input.copy_(dummy_input)
            graph.replay()
        torch.cuda.synchronize()
        cuda_graph_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'cuda_graph',
            'runtime_ms': cuda_graph_time
        })
    except Exception as e:
        results.append({
            'method': 'cuda_graph',
            'runtime_ms': None,
            'error': str(e)
        })
    
    # 5b. channels-last + CUDA Graph
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        static_input_cl = dummy_input.to(memory_format=torch.channels_last).clone()  # type: ignore
        static_output_cl = torch.empty_like(model_cl(static_input_cl))
        
        graph_cl = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_cl):
            static_output_cl.copy_(model_cl(static_input_cl))
        
        # Warmup
        for _ in range(5):
            static_input_cl.copy_(dummy_input.to(memory_format=torch.channels_last))  # type: ignore
            graph_cl.replay()
        torch.cuda.synchronize()
        
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            # Include input conversion in timing
            static_input_cl.copy_(dummy_input.to(memory_format=torch.channels_last))  # type: ignore
            graph_cl.replay()
        torch.cuda.synchronize()
        cuda_graph_cl_time = ((time.time_ns() / 1e6) - start) / iterations
        results.append({
            'method': 'channels_last_cuda_graph',
            'runtime_ms': cuda_graph_cl_time
        })
    except Exception as e:
        results.append({
            'method': 'channels_last_cuda_graph',
            'runtime_ms': None,
            'error': str(e)
        })
    
    return results

