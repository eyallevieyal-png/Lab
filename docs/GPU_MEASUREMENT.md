# Running the GPU measurement (free, about 10 minutes)

Goal: replace "computed from specs" with "measured" for the roofline, decode attention and KV transfer. See the header of `experiments/measure_gpu.py` for exactly what is timed and what is not.

## Google Colab (free tier, no payment)

1. Open colab.research.google.com, create a new notebook.
2. Menu: Runtime > Change runtime type > **T4 GPU** > Save. (Do not buy Colab Pro; the free T4 is enough.)
3. Cell 1, a short smoke run of the script itself:

   ```
   !git clone --branch feat/gpu-measurement https://github.com/eyallevieyal-png/Lab.git
   %cd Lab
   !python -m experiments.measure_gpu --quick
   ```

   It should end with `Wrote experiments/results/measured_gpu_..._quick.json`. If any section prints `FAILED`, copy that line back; do not worry about it.
4. Cell 2, the full run (a few minutes):

   ```
   !python -m experiments.measure_gpu
   ```
5. Cell 3, download the result file(s):

   ```
   from google.colab import files
   import glob
   for f in glob.glob("experiments/results/measured_gpu_*.json"):
       files.download(f)
   ```
6. Send the JSON file(s) back. They contain GPU name, timings and clocks only: no account data, no secrets.

Notes: the free GPU can differ between sessions (T4 is typical); the script records which one it got. Run it twice at different times if you can; GPU clocks throttle and one run is one sample.

## What will and will not be concluded from it

- It will show whether the roofline shape holds on a GPU (ratio measured/predicted per batch), where the compute/memory crossover really is, whether decode attention stays memory-bound as the batch grows, and how long a KV fetch takes relative to a decode step.
- It will not validate H100 numbers: a T4 has different bandwidth, tensor cores and PCIe generation. The article will say so.
- MoE with speculative verification and a contended link (the H13 utilisation `u`) are not covered by this run.
