# Laptop backup environment (WSL2 Ubuntu-22.04, RTX 3060 Laptop 6 GB)

Running log, 2026-11-23. Not committed.

## Starting state
- Windows repo: main at 392598f, clean.
- WSL clone ~/mmgia: at 3b4f02c (2 commits), clean, remote `windows`. No ~/miniconda3, no ~/jacob.
- WSL: 20 cores, 15 GB RAM, 953 GB free; nvidia-smi: RTX 3060 Laptop, 6144 MiB, driver 610.88.
- Server: Python 3.11.18, torch 2.14.1+cu126, torchvision 0.29.1+cu126, transformers 5.23.0,
  numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, rouge_score 0.1.2 (full freeze copied into
  WSL ~/server_freeze.txt).
- Server third_party: dager @ 5f8e306, invertinggradients @ 1157b61, both clean (= scripts/get_third_party.sh pins).
- WSL has no ssh key for the server: every copy goes Windows scp -> scratchpad staging -> cp into WSL.

## Steps
1. Miniconda: `curl Miniconda3-latest-Linux-x86_64.sh` + `bash -b -p ~/miniconda3` (download ~0.6 MB/s, slow).
2. `mkdir -p ~/jacob && ln -s ~/mmgia ~/jacob/mmgia` done.
3. Clone was clean at 3b4f02c; `git fetch windows && git merge --ff-only windows/main` -> 392598f, clean.
   No CR in scripts/*.sh or src/*.py.
4. Splits: byte md5 differs from the server for pairs.csv and all seed*.json, because the SERVER copies
   have CRLF line endings (pairs.csv 3200 CR = one per line; seed0.json 2369 CR). After stripping CR the
   server md5s equal the laptop md5s exactly, and the parsed JSON / CSV text are equal (checked in Python).
   common.py reads them in text mode (universal newlines), so the code sees identical data. Laptop keeps the
   git-tracked LF versions (not overwritten, to keep the clone clean).
   pretrained/feature_norm_seed0.pt: md5 a223b8e5... identical on both.
5. Copied from server (md5 verified identical): data/cache/images128.npz (8f3bece8...), runs/dry/fl/seed0_fedavg/
   global_round9.pt (9df6739a...) + train_log.json, pretrained/hf/hub (GPT-2, model.safetensors 8b963419...),
   pretrained/torch/hub/checkpoints (resnet18-f37072fd e0b1c919..., alexnet cc6360b2...), third_party/dager
   and third_party/invertinggradients (with .git; only file-mode changes from the Windows copy, restored with
   `git checkout -- .`; HEADs 5f8e306 / 1157b61, clean). No download from GitHub or HF needed.
   Note: laptop has no data/iuxray/extracted images, so images128.npz cannot be rebuilt locally; the copy is required.
6. Env: Miniconda 26.9.4 at ~/miniconda3. `conda create -n mmgia --override-channels -c conda-forge python=3.11.18`
   (conda-forge only, to avoid the Anaconda defaults-channel terms-of-service acceptance; server's Python may be
   an Anaconda-defaults build of the same 3.11.18).
   First tried `pip install -r server_freeze.txt --extra-index-url .../cu126`: ~0.6 MB/s internet, 3 GB of wheels.
   Stopped after ~30 min on the user's instruction (pip PID 987 killed, site-packages emptied).
7. Instead copied the server's env site-packages (7.1 GB, 50511 entries, 7434132046 bytes) over the LAN:
   `ssh server "cd ~/miniconda3/envs/mmgia/lib/python3.11 && nice -n 19 ionice -c3 tar -cf - site-packages"
   | wsl -d Ubuntu-22.04 -- tar -xf - -C ~/miniconda3/envs/mmgia/lib/python3.11` (run from a .cmd file so the pipe is
   binary-safe; PowerShell 5.1 pipes would corrupt binary). 1.1 min, exit 0.
   Verified: 50511 entries / 45512 regular files / 7434132046 bytes of file content on both sides (du -sb of the
   tree differs only by directory sizes). `pip freeze` identical to the server's (88 lines). Python 3.11.18.
   torch 2.14.1+cu126, CUDA 12.6, cuDNN 91002, cuda available, device RTX 3060 Laptop GPU.
   `import common, attack_text, attack_image` from src/ OK (DAGER on path). bin/ was NOT copied, so console
   entry points of server packages (e.g. huggingface-cli, tqdm) are absent from the laptop's bin/; bin/pip is the
   laptop's own (shebang points to the laptop python). Use `python -m ...` / `python src/...`.
8. Test A (text, --parallel 1000): OOM after 34 s in the first victim's sequence search (GPT-2 MLP forward,
   tried to allocate 382 MiB). Cause: attack_text.py default `--gpu-mem 0.5` -> set_per_process_memory_fraction(0.5)
   = 3.0 GiB cap on the 6 GiB card (12 GiB on the 24 GiB 4090). Peak card memory 3191 MiB, 45 C, 60 W.
   Log runs/laptop_test/textA.log. Retried once with --parallel 200 (nothing else changed), as instructed.
9. Test A retry (--parallel 200): RC 0, wall 235 s. Log runs/laptop_test/textA_p200.log, results runs/laptop_test/text/.
   victim 0: ROUGE-L 0.5952, 8/8 recovered, 2.15 min  (server 0.595, 8/8: identical)
   victim 1: ROUGE-L 0.4734, 7/8 recovered, 1.65 min  (server 0.478, 7/8: close, -0.005)
   Peak card memory 2903 MiB (card total incl. Windows), peak 50 C, peak 59.7 W.
   Untested guess for the v1 gap: --parallel 200 vs 1000 changes batch shapes (float rounding near a threshold),
   or Ampere vs Ada numerics.
10. Test B (image, 300 iters): RC 0, wall 34 s; 300 iterations t=0.37 min (summary 0.40 min incl. eval) vs server 0.23
   min alone. Self-check "cosine distance -1.19e-07" (about 0), BN target error 1.9e-06 / 3.0e-07. ssim 0.334, cos 0.432
   (not a result). Peak memory 2755 MiB, 55 C, 58 W. Output runs/laptop_test/image/. attack_image.py has no memory cap.
11. Idle card memory in WSL nvidia-smi reads 0 MiB, so the peaks above are the attack process. Laptop power.limit N/A.
   Estimate for 360 text attacks (90 calls x 4 victims) at --parallel 200 --idle 2: ~1.9 min/victim + ~10 s/call
   startup -> ~11.5-12 h (inferred from 2 seed-0 victims; seeds 1-5 and other protocols may differ).
   Clone still clean (`git status` empty; runs/ is untracked/ignored).

## How to take over the text attack
cd ~/jacob/mmgia && source ~/miniconda3/etc/profile.d/conda.sh && conda activate mmgia
Run the server's text job script unchanged, but --parallel 1000 OOMs under the default --gpu-mem 0.5 (3 GiB cap);
use --parallel 200 (tested) or raise --gpu-mem (untested). The server writes no code here: sync code via git only.
