# About this fork

A downstream fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that
carries a small set of **instrumentation patches**, so this build can be used as a
*reference oracle*: something another inference engine can be diffed against
numerically, rather than compared by eye.

Everything here is additive and inert by default — every hook is behind an environment
variable and costs nothing when unset.

## Branches

| branch | what it is |
|---|---|
| **`nixpt-oracle`** *(default)* | Upstream `master` **plus** our patch series. This is what you want. |
| `master` | A plain mirror of upstream `master`. No local changes. Used only as the rebase target. |

> [!IMPORTANT]
> **`nixpt-oracle` is a rebasing branch and IS FORCE-PUSHED.** When upstream moves, the
> patch series is replayed onto the new base and the branch is force-updated. It is the
> default branch because it is the useful one, not because it is stable history.
>
> After an upstream bump, do **not** `git pull` (it will try to merge a rewritten
> history). Do this instead:
>
> ```sh
> git fetch origin
> git reset --hard origin/nixpt-oracle
> ```
>
> If you have local work on top, rebase it: `git rebase --onto origin/nixpt-oracle <old-base>`.

## Reproducibility: cite a tag, never the branch

Because the branch moves, **any measurement published against this fork must name a
tag**, not `nixpt-oracle`. Tags are cut at the exact tree a result was measured on:

| tag | upstream base |
|---|---|
| `oracle-b9935` | `f2d1c2f` (tag `b9935`) |

Cut a new tag whenever a result is published off a newer base.

## What the patches do

Three independent clusters. See the individual commits for the full reasoning.

**Graph tracing** (`common/debug.cpp`, `examples/eval-callback/eval-callback.cpp`)
- `LLAMA_CB_FILTER` — comma-separated regexes matched against tensor names, so a
  targeted trace of a few named tensors (e.g. one per layer) runs in ONE pass instead
  of dumping every graph tensor. The prohibitive cost of an unfiltered dump is the
  *printing*, not the compute.
- `LLAMA_CB_ELEMS` — how many elements to print from each end of each dimension
  (upstream hardcodes 3). Set it `>= ne[0]/2` to print every element with no elision,
  so a consumer compares complete vectors instead of silently scoring the printed ends.

**Multimodal dumps** (`tools/mtmd/*`)
- `CLIP_DUMP_EMBD=<path>` — writes the vision/audio encoder output as raw f32 behind an
  `[n_embd, n_tokens]` int64 header. Audio encodes once per 30 s mel chunk, so the path
  gets a per-call counter suffix; without it later chunks overwrite earlier ones and the
  result is indistinguishable from a correct single-chunk dump.
- `MTMD_DUMP_TOKENS=1` — prompt token ids as the tokenizer emits them, and each sampled
  id as it is generated.

**Quantize** (`src/llama-quant.cpp`, `tools/quantize/quantize.cpp`)
- Allows `--tensor-type` / `--tensor-type-file` overrides to take effect under
  `--only-copy`, for surgical single-tensor requantisation. Also re-derives the default
  type from the model's real ftype in that mode — it was derived from `LLAMA_FTYPE_COPY`
  (which resolves to f32), so overrides silently produced f32 tensors.
- Exposes `NVFP4` as a `llama-quantize` option.

## Building

For reference traces, build **CPU-only** — a trace never touches the GPU, and a CUDA
build recompiles hundreds of `fattn-mma` template instances for nothing:

```sh
cmake -B build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target llama-eval-callback -j 4
```

> [!WARNING]
> **Rebuild after every branch update.** A stale binary in a long-lived work tree has
> the env var missing from the binary while present in the source — the flag looks
> available and silently does nothing. This has already cost real debugging time once.

## Updating onto a newer upstream

The checkout does not need full history; cherry-pick applies patches without needing a
merge base, so this works from a shallow clone:

```sh
git fetch --depth=1 origin master          # or upstream master, then mirror it
git checkout -b rebase-tmp origin/master
git cherry-pick <the patch series>
# rebuild, re-run your gates, confirm they reproduce
git branch -f nixpt-oracle rebase-tmp
git push --force-with-lease origin nixpt-oracle
```

Tag the outgoing tip **before** force-pushing if any published result was measured on it.

A clean cherry-pick proves nothing about the build or the numbers — rebuild all patched
targets and re-run a known gate before trusting a rebase.
