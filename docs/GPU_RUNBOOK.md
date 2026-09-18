# AutoDL GPU runbook

## Fixed stack

ProbeGRPO trains on current `verl`; it does not install RAGEN's pinned training stack. The first
GPU milestone uses:

- `verl` commit `cf14ded3a448107e70a206fd201817cc1cbae348`;
- the commit's frozen `uv.lock` with the `vllm` and `fsdp` extras;
- `Qwen/Qwen3.5-2B`;
- one NVIDIA GPU with at least 45,000 MiB visible memory;
- driver CUDA compatibility 12.8 or newer;
- at least 80 GiB free workspace storage.

RAGEN remains an environment reference: its Sokoban and WebShop behavior will be adapted to verl's
current `AgentLoop` interface after the standard GRPO stack passes. Do not install RAGEN's
`vllm==0.8.2` dependency into this runtime.

## AutoDL bring-up

Choose one 48 GB card such as an RTX A6000, A40, or L40S. Put both the repository and runtime under
AutoDL's persistent data-disk directory rather than a small system disk. In the commands below,
replace both paths with the actual directories shown by the instance.

```bash
cd /path/to/ProbeGRPO
bash scripts/check_gpu_host.sh /path/to/persistent-workspace
bash scripts/bootstrap_verl.sh /path/to/persistent-workspace/probegrpo-runtime
bash scripts/check_verl_stack.sh /path/to/persistent-workspace/probegrpo-runtime/verl
```

The bootstrap is intentionally pinned. It records the ProbeGRPO commit, verl commit, uv version, and
creation time in `runtime_manifest.txt`. If setup fails, keep the full command output; do not upgrade
individual torch, Transformers, vLLM, or verl packages in place.

The bootstrap resolves the lock once and creates `verl/.venv`. Later commands use that environment's
absolute Python path, including for Ray workers, so an accidental ambient Conda environment cannot
silently change the training stack.

If Hugging Face access is slow, configure a trusted mirror explicitly in the shell before running
the scripts. Never write an access token into this repository or a command-line argument saved in
shell history.

## Five-update standard GRPO gate

The first paid GPU run is deliberately not agentic. It checks Qwen3.5 model loading, vLLM rollout,
FSDP2 LoRA updates, GRPO grouping (`K=4`), checkpointing, and the exact software lock before adding
Sokoban or probes.

```bash
cd /path/to/ProbeGRPO
bash scripts/run_verl_grpo_smoke.sh \
  /path/to/persistent-workspace/probegrpo-runtime/verl
```

Defaults:

- two GSM8K prompts per update and four rollouts per prompt;
- LoRA rank 32, alpha 64, `all-linear` target discovery;
- prompt/response limits of 256 tokens;
- optimizer and parameter offload;
- five training updates and a checkpoint at step 5;
- console-only logging and no W&B login.

To verify resume, rerun against the same output directory:

```bash
TOTAL_TRAINING_STEPS=6 bash scripts/run_verl_grpo_smoke.sh \
  /path/to/persistent-workspace/probegrpo-runtime/verl
```

The second command must resume from step 5 and perform exactly one additional update.

## Acceptance checklist

- `check_gpu_host.sh` and `check_verl_stack.sh` pass without overrides.
- Qwen3.5 resolves as model type `qwen3_5`.
- Five updates complete without NaN, Inf, CUDA OOM, Ray worker death, or tokenizer mismatch.
- Each prompt produces exactly four rollout samples.
- A step-5 checkpoint exists and resumes for step 6.
- The log contains reward, response length, actor loss, rollout time, and update time.
- Peak GPU memory and total wall-clock time are recorded in `experiments/environment/` before the
  instance is stopped.

Only after this gate passes should the project implement and run the Sokoban AgentLoop, followed by
the deterministic replay/probe hook.
