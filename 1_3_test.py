import torch
import torch.nn as nn
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.shape[-1], dtype=torch.float32))
        attn_weights = self.softmax(scores)
        output = torch.matmul(attn_weights, v)
        return output


def get_cpu_memory_usage():
    return psutil.Process().memory_info().rss


def get_flops(model, input_tensor):
    input_size = input_tensor.size(0)
    d_model = input_tensor.size(-1)
    flops = 4 * input_size ** 2 * d_model  # 3 linear layers + attention matmul
    return flops


def profile_self_attention(input_size, d_model=1, device='cpu', use_quantization=False, use_fp16=False):
    torch.cuda.empty_cache()
    input_tensor = torch.randn(input_size, d_model).to(device)
    model = SelfAttention(d_model).to(device)


    if use_quantization and device == 'cpu':
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        ).to(device)


    flops = get_flops(model, input_tensor)


    start_time = time.time()

    if use_fp16 and device == 'cuda':  # Use mixed precision on GPU
        with torch.no_grad():
            with torch.amp.autocast(device_type = 'cuda',enabled = True):
                output = model(input_tensor)
        torch.cuda.synchronize()  # Synchronize for GPU timing
    else:
        output = model(input_tensor)
    torch.cuda.reset_peak_memory_stats()

    elapsed_time = time.time() - start_time


    if device == 'cuda':
        torch.cuda.reset_max_memory_allocated()
        model(input_tensor)  # Forward pass
        memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    else:
        memory_used = get_cpu_memory_usage() / (1024 ** 2)  # Convert to MB

    return flops, memory_used, elapsed_time


def profile_multiple_lengths(lengths, device='cpu', use_quantization=False, use_fp16=False):
    flops_list, memory_list, time_list = [], [], []
    for length in lengths:
        flops, memory, elapsed_time = profile_self_attention(length, device=device, use_quantization=use_quantization, use_fp16=use_fp16)
        flops_list.append(flops)
        memory_list.append(memory)
        time_list.append(elapsed_time)
    return flops_list, memory_list, time_list


def plot_results(lengths, cpu_results, gpu_results, metric_name,filename_prefix="plot"):
    fig, ax = plt.subplots()
    
    cpu_mean = np.mean(cpu_results, axis=0)
    cpu_se = np.std(cpu_results, axis=0) / np.sqrt(len(cpu_results))
    
    ax.errorbar(lengths, cpu_mean, yerr=cpu_se, label='CPU', fmt='-o', capsize=5)

    if len(gpu_results) > 0:  # Check if GPU results exist
        gpu_mean = np.mean(gpu_results, axis=0)
        gpu_se = np.std(gpu_results, axis=0) / np.sqrt(len(gpu_results))

        if len(lengths) == len(gpu_mean):  # Ensure lengths match
            ax.errorbar(lengths, gpu_mean, yerr=gpu_se, label='GPU', fmt='-s', capsize=5)
        else:
            print(f"Warning: Mismatch between input lengths and GPU results size. "
                  f"Lengths: {len(lengths)}, GPU results: {len(gpu_mean)}")

    ax.set_xlabel('Input Length')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs Input Length')
    ax.set_title(f'{metric_name} vs Input Length')
    ax.legend()

  
    fig.savefig(f"{filename_prefix}_{metric_name}.png")


if __name__ == "__main__":
    input_lengths = [10, 100, 1000, 10000, 100000]
    
    
    cpu_flops, cpu_memory, cpu_time = [], [], []
    gpu_flops, gpu_memory, gpu_time = [], [], []
    
    for _ in range(5):  
        cpu_f, cpu_m, cpu_t = profile_multiple_lengths(input_lengths, device='cpu', use_quantization=True)
        cpu_flops.append(cpu_f)
        cpu_memory.append(cpu_m)
        cpu_time.append(cpu_t)

        if torch.cuda.is_available():
            gpu_f, gpu_m, gpu_t = profile_multiple_lengths(input_lengths, device='cuda', use_fp16=True)
            gpu_flops.append(gpu_f)
            gpu_memory.append(gpu_m)
            gpu_time.append(gpu_t)
        else:
            print("GPU not available. Skipping GPU profiling.")
    

    plot_results(input_lengths, cpu_flops, gpu_flops, 'FLOPS', filename_prefix="flops_plot")
    plot_results(input_lengths, cpu_memory, gpu_memory, 'Memory (MB)', filename_prefix="memory_plot")
    plot_results(input_lengths, cpu_time, gpu_time, 'Time (s)', filename_prefix="time_plot")

