import time
import torch
import psutil
import platform


def measure_inference_time(model, test_loader, device):
    """Measure end-to-end inference speed for a model on the test loader.

    The function performs a short warm-up before timing, then reports total
    elapsed inference time, average latency per image, throughput, and available
    hardware information.

    Args:
        model: PyTorch model to benchmark.
        test_loader: Data loader yielding test batches.
        device: Torch device where input tensors should be evaluated.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            x, _ = next(iter(test_loader))
            x = x.to(device)
            _ = model(x)

    start_time = time.time()
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _ = model(x)
    end_time = time.time()

    total_time = end_time - start_time
    total_images = len(test_loader.dataset)
    avg_time_per_image_ms = (total_time / total_images) * 1000
    throughput = total_images / total_time

    print("=== INFERENCE TIME ===")
    print(f"Total test images:       {total_images}")
    print(f"Total inference time:    {total_time:.3f} sec")
    print(f"Avg. time per image:     {avg_time_per_image_ms:.3f} ms")
    print(f"Throughput:              {throughput:.2f} images/sec")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware:                GPU - {gpu_name}")
    elif torch.backends.mps.is_available():
        print("Hardware:                Apple MPS (Metal Performance Shaders)")
    else:
        cpu_count = psutil.cpu_count(logical=False)
        cpu_name = platform.processor() or "Unknown CPU"
        print(f"Hardware:                CPU - {cpu_name} ({cpu_count} cores)")
