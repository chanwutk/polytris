import torch
import time
import os
import json


def benchmark_model_optimization(model: "torch.nn.Module", device: str, tile_size: int,
                                 batch_size: int, iterations: int = 128):
    """
    Benchmark different acceleration methods and save results to JSONL file.
    
    Tests: baseline, torch.compile, channels-last, torch.compile + channels-last,
           TorchScript Trace, TorchScript + Optimize, CUDA Graph, channels-last + CUDA Graph
    
    Args:
        model: The model to optimize (expects forward(image, position))
        device: Device to use
        tile_size: Tile size (used to create dummy input)
        batch_size: Batch size for benchmarking
        iterations: Number of iterations for benchmarking
    """
    model.eval()
    
    # Create dummy inputs for benchmarking (image and position)
    dummy_image = torch.randn(batch_size, 6, tile_size, tile_size, device=device)
    dummy_pos = torch.randn(batch_size, 2, device=device)
    
    results = []
    
    # 1. Baseline (eager)
    print(f"Running baseline (eager) for {iterations} iterations...")
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            _ = model(dummy_image, dummy_pos)
        torch.cuda.synchronize()
        baseline_time = ((time.time_ns() / 1e6) - start) / iterations
    results.append({
        'method': 'baseline',
        'runtime_ms': baseline_time
    })
    
    # 2. torch.compile
    print(f"Running torch.compile for {iterations} iterations...")
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = compiled_model(dummy_image, dummy_pos)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = compiled_model(dummy_image, dummy_pos)
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
    print(f"Running channels-last (without torch.compile) for {iterations} iterations...")
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                input_cl = dummy_image.to(memory_format=torch.channels_last)  # type: ignore
                _ = model_cl(input_cl, dummy_pos)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                # Include input conversion in timing
                input_cl = dummy_image.to(memory_format=torch.channels_last)  # type: ignore
                _ = model_cl(input_cl, dummy_pos)
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
    print(f"Running torch.compile + channels-last for {iterations} iterations...")
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        compiled_channels_last_model = torch.compile(model_cl, mode="reduce-overhead")
        dummy_input_cl = dummy_image.to(memory_format=torch.channels_last)  # type: ignore
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = compiled_channels_last_model(dummy_input_cl, dummy_pos)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                # Include input conversion in timing
                input_cl = dummy_image.to(memory_format=torch.channels_last)  # type: ignore
                _ = compiled_channels_last_model(input_cl, dummy_pos)
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
    print(f"Running TorchScript Trace for {iterations} iterations...")
    traced_model = None
    try:
        traced_model = torch.jit.trace(model, (dummy_image, dummy_pos))
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = traced_model(dummy_image, dummy_pos)  # type: ignore
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = traced_model(dummy_image, dummy_pos)  # type: ignore
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
    print(f"Running TorchScript Trace + Optimize for {iterations} iterations...")
    try:
        if traced_model is None:
            traced_model = torch.jit.trace(model, (dummy_image, dummy_pos))
        optimized_model = torch.jit.freeze(traced_model)
        optimized_model = torch.jit.optimize_for_inference(optimized_model)
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = optimized_model(dummy_image, dummy_pos)
            torch.cuda.synchronize()
            start = time.time_ns() / 1e6
            for _ in range(iterations):
                _ = optimized_model(dummy_image, dummy_pos)
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
    print(f"Running CUDA Graph for {iterations} iterations...")
    try:
        # CUDA Graph requires fixed input/output sizes
        static_image = dummy_image.clone()
        static_pos = dummy_pos.clone()
        static_output = torch.empty_like(model(static_image, static_pos))
        
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output.copy_(model(static_image, static_pos))
        
        # Warmup
        for _ in range(5):
            static_image.copy_(dummy_image)
            static_pos.copy_(dummy_pos)
            graph.replay()
            static_output.clone()
        torch.cuda.synchronize()
        
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            static_image.copy_(dummy_image)
            static_pos.copy_(dummy_pos)
            graph.replay()
            static_output.clone()
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
    print(f"Running channels-last + CUDA Graph for {iterations} iterations...")
    try:
        model_cl = model.to(memory_format=torch.channels_last)  # type: ignore
        static_input_cl = dummy_image.to(memory_format=torch.channels_last).clone()  # type: ignore
        static_pos = dummy_pos.clone()
        static_output_cl = torch.empty_like(model_cl(static_input_cl, static_pos))
        
        graph_cl = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_cl):
            static_output_cl.copy_(model_cl(static_input_cl, static_pos))
        
        # Warmup
        for _ in range(5):
            static_input_cl.copy_(dummy_image.to(memory_format=torch.channels_last))  # type: ignore
            static_pos.copy_(dummy_pos)
            graph_cl.replay()
            static_output_cl.clone()
        torch.cuda.synchronize()
        
        start = time.time_ns() / 1e6
        for _ in range(iterations):
            # Include input conversion in timing
            static_input_cl.copy_(dummy_image.to(memory_format=torch.channels_last))  # type: ignore
            static_pos.copy_(dummy_pos)
            graph_cl.replay()
            static_output_cl.clone()
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
    
    print(f"Benchmarking completed... done")
    return results

