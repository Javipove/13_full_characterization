import torch
import ttnn


# Shape constraints for TILE_LAYOUT: last two dimensions should be multiples of 32.
INPUT_SHAPE = (1, 1, 128, 128)   # [B, C, M, K]
W1_SHAPE = (1, 1, 128, 128)       # [B, C, K, N1]
B1_SHAPE = (1, 1, 128, 128)       # [B, C, M, N1]
W2_SHAPE = (1, 1, 128, 128)       # [B, C, N1, N2]
B2_SHAPE = (1, 1, 128, 128)       # [B, C, M, N2]


def run_workload(input_tensor, w1_tensor, b1_tensor, w2_tensor, b2_tensor):
	# Chained workload: matmul -> add -> matmul -> add
	x = ttnn.matmul(input_tensor, w1_tensor)
	x = ttnn.add(x, b1_tensor)
	x = ttnn.matmul(x, w2_tensor)
	x = ttnn.add(x, b2_tensor)
	return x


def main(num_iterations=5):
	print("[Init] Opening Tenstorrent device...")
	device = ttnn.open_device(device_id=0)

	dtype = ttnn.bfloat16
	layout = ttnn.TILE_LAYOUT
	dram_mem_config = ttnn.DRAM_MEMORY_CONFIG
	l1_mem_config = ttnn.L1_MEMORY_CONFIG

	host_input = torch.randn(INPUT_SHAPE, dtype=torch.bfloat16)
	host_w1 = torch.randn(W1_SHAPE, dtype=torch.bfloat16)
	host_b1 = torch.randn(B1_SHAPE, dtype=torch.bfloat16)
	host_w2 = torch.randn(W2_SHAPE, dtype=torch.bfloat16)
	host_b2 = torch.randn(B2_SHAPE, dtype=torch.bfloat16)
	print("[Init] Host tensors created.")

	# Persistent input tensor in DRAM for trace-friendly fixed addressing.
	print("[Init] Allocating persistent DRAM input tensor...")
	input_dram_tensor = ttnn.allocate_tensor_on_device(INPUT_SHAPE, dtype, layout, device, dram_mem_config)

	# Constant model parameters on device.
	print("[Init] Uploading model parameters to device (L1)...")
	w1_tensor = ttnn.from_torch(host_w1, dtype=dtype, layout=layout, device=device, memory_config=l1_mem_config)
	b1_tensor = ttnn.from_torch(host_b1, dtype=dtype, layout=layout, device=device, memory_config=l1_mem_config)
	w2_tensor = ttnn.from_torch(host_w2, dtype=dtype, layout=layout, device=device, memory_config=l1_mem_config)
	b2_tensor = ttnn.from_torch(host_b2, dtype=dtype, layout=layout, device=device, memory_config=l1_mem_config)
	print("[Init] Parameter upload complete.")

	# First run to compile programs.
	print("[Compile] Running first pass to compile kernels/programs...")
	ttnn.copy_host_to_device_tensor(host_input, input_dram_tensor, cq_id=0)
	input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_mem_config)
	output_tensor = run_workload(input_l1_tensor, w1_tensor, b1_tensor, w2_tensor, b2_tensor)
	print("[Compile] First pass completed.")

	# Capture trace with fixed input/output addresses.
	print("[Trace] Starting trace capture...")
	ttnn.copy_host_to_device_tensor(host_input, input_dram_tensor, cq_id=0)
	tid = ttnn.begin_trace_capture(device, cq_id=0)
	input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, l1_mem_config)
	output_tensor = run_workload(input_l1_tensor, w1_tensor, b1_tensor, w2_tensor, b2_tensor)
	ttnn.end_trace_capture(device, tid, cq_id=0)
	print(f"[Trace] Capture completed. Trace ID: {tid}")

	# Replay the captured trace for multiple random inputs.
	outputs = []
	print(f"[Replay] Executing trace for {num_iterations} iterations...")
	for _ in range(num_iterations):
		iter_idx = len(outputs) + 1
		print(f"[Replay] Iteration {iter_idx}/{num_iterations}: write input, execute trace, enqueue readback")
		host_input = torch.randn(INPUT_SHAPE, dtype=torch.bfloat16)
		ttnn.copy_host_to_device_tensor(host_input, input_dram_tensor, cq_id=0)
		ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
		outputs.append(output_tensor.cpu(blocking=False))

	# Final synchronization ensures all non-blocking reads are complete.
	print("[Sync] Waiting for device to finish all queued work...")
	ttnn.synchronize_device(device)
	final_output = ttnn.to_torch(outputs[-1])
	print(f"Trace executed {num_iterations} iterations. Final output shape: {tuple(final_output.shape)}")

	print("[Shutdown] Closing device.")
	ttnn.close_device(device)


if __name__ == "__main__":
	main()