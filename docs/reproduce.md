# Reproduce the H200 benchmark

The recorded run used one NVIDIA H200 SXM with driver 595.91.07. NVIDIA-SMI reported 143,771 MiB of GPU memory. The Runpod image was `runpod/pytorch:2.1.0-cu1320-torch280-ubuntu2404`. The model weights were fetched from Hugging Face during the first server startup. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

1. Start one Linux GPU instance with a compatible driver, Python 3.12, Bash, curl, and enough disk for vLLM dependencies and the model. The recorded Pod used an 80 GB container disk.
2. Clone or copy this repository onto the instance. Run the client and model server on the same machine. Port 8000 can stay private because the runner binds vLLM to `127.0.0.1`.
3. Create and activate a virtual environment, then run the benchmark.

   ```bash
   python -m venv --system-site-packages .venv
   source .venv/bin/activate
   bash run_on_pod.sh 2>&1 | tee benchmark-run.log
   ```

   On the recorded Runpod image, installing directly into system Python failed because pip could not uninstall the distribution-managed PyJWT package. The virtual environment let pip install its own dependencies. In the successful run, pip reused packages downloaded on the first attempt.

4. Verify that each result file contains two JSON lines and that each line contains 51 samples. Inspect the cache setting in each server's startup log.

   ```bash
   wc -l results-off.jsonl results-on.jsonl
   grep 'non-default args' vllm-off.log vllm-on.log
   ```

5. Copy the result files and logs off the GPU instance before terminating it.

The runner pins vLLM to 0.32.0 and holds the model, context cap (8,192), GPU memory utilization fraction (0.5), prompt token usage reporting, and eager execution constant across cache modes. Eager execution avoids compilation and CUDA graph startup for this short run. For longer experiments, measure compiled serving as a separate configuration. [vLLM serve options](https://docs.vllm.ai/en/v0.32.0/cli/serve/)
