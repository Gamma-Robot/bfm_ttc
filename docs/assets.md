# Assets and usage conditions

Checkpoints and source datasets come from their original projects.
`artifacts/lock.v1.json` records the expected identity of each file or prepared
directory, and the runtime checks these identities before simulation. In each
lock entry, `source_url` names the upstream project. The runtime downloads a
file automatically only when its entry also has an explicit `uri`.

## G1 robot

The robot model comes from the official
[HEFT source repository](https://github.com/Axellwppr/motion_tracking) at
revision `0d5ba31e33397f3543d350d98b637e26d92f470a`. Fetch its robot assets
with a sparse checkout, then prepare the experiment model:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
  https://github.com/Axellwppr/motion_tracking.git outputs/assets/motion_tracking
git -C outputs/assets/motion_tracking sparse-checkout set sim2real/config/g1/assets
GIT_LFS_SKIP_SMUDGE=1 git -C outputs/assets/motion_tracking checkout \
  0d5ba31e33397f3543d350d98b637e26d92f470a
uv run python tools/prepare_g1.py outputs/assets/motion_tracking \
  artifacts/store/robots/g1-paper-model
```

The tool verifies the official source, then installs the experiment XML, the
35 meshes it references, and the license. The resulting tree
has 37 files and is checked against the lock. The experiment XML itself is
included under `assets/robots/g1/`, and
[SOURCE.md](../assets/robots/g1/SOURCE.md) documents how it differs from the
upstream deployment XML.


## HEFT-PMG

Use the same pinned [official HEFT revision](https://github.com/Axellwppr/motion_tracking/tree/0d5ba31e33397f3543d350d98b637e26d92f470a/sim2real/config/g1/ckpts/G1_PMG).
Following the upstream download instructions, obtain `policy.onnx` and its
external tensor file `policy.onnx.data` from `sim2real/config/g1/ckpts/G1_PMG/`.
Then convert the checkpoint in two steps:

```sh
uv run python -m bfm_adapters.families.heft.source_export \
  /path/to/motion_tracking/sim2real/config/g1/ckpts/G1_PMG/policy.onnx \
  artifacts/store/heft/pmg/policy.onnx
uv run bfm-ttc prepare-bfm heft-pmg-policy.batched
```

The first step embeds the external tensors and exposes the context,
motion-command, target-motion, and proprioception inputs, without changing the
policy computation. The second step makes the graph accept
batched inputs. The identities of both outputs are pinned in the lock. The
applicable MIT notice is in HEFT's upstream repository.

## TWIST2

The official [TWIST2 checkpoint](https://github.com/amazon-far/TWIST2/blob/d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a/assets/ckpts/twist2_1017_20k.onnx)
at revision `d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a` works without
conversion. The runtime downloads this pinned file when it is first needed and
verifies its hash. If you already have an upstream copy, install it instead:

```sh
install -D /path/to/TWIST2/assets/ckpts/twist2_1017_20k.onnx \
  artifacts/store/twist2/1017-20k/policy.onnx
```

This checkpoint already supports batched inference. Its license and usage
conditions are given in the upstream repository.

## SONIC

The source models are published by
[NVIDIA GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) and used by
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
The original release and the low-latency variant are separate checkpoints.
Their usage terms are given in the model card and the upstream repository.

For the original release, obtain `model_encoder.onnx` and
`model_decoder.onnx` from official revision
[`0af0cb7568257b04a8569b9c76f326e707b122fc`](https://huggingface.co/nvidia/GEAR-SONIC/tree/0af0cb7568257b04a8569b9c76f326e707b122fc).
Combine them into a single G1 policy, then prepare it for batched inference:

```sh
uv run python -m bfm_adapters.families.sonic.source_export \
  /path/to/model_encoder.onnx /path/to/model_decoder.onnx \
  artifacts/store/sonic/release/g1/policy.onnx
uv run bfm-ttc prepare-bfm sonic-release-g1-policy.batched
```

For the low-latency variant, obtain `low_latency/model_encoder.onnx` and
`low_latency/model_decoder.onnx` from official revision
[`183ed69145708878ee68cc8791848ee66f233d24`](https://huggingface.co/nvidia/GEAR-SONIC/tree/183ed69145708878ee68cc8791848ee66f233d24/low_latency):

```sh
uv run python -m bfm_adapters.families.sonic.source_export --variant low-latency \
  /path/to/low_latency/model_encoder.onnx /path/to/low_latency/model_decoder.onnx \
  artifacts/store/sonic/low_latency/g1/policy.onnx
uv run bfm-ttc prepare-bfm sonic-low-latency-g1-policy.batched
```

The exporter verifies the hashes of both source files and selects the G1
encoder path. Token quantization and the decoder computation are unchanged. The identities of the native and batched
graphs are pinned in the lock. The paper's tracking experiments use the
original-release checkpoint.

## Motion data

A local motion is an [Any4HDMI v2](https://github.com/EGalahad/any4hdmi/blob/main/docs/dataset_format.md)
dataset with `manifest.json` and the motion files
it references. Set `reference.dataset_root` and `reference.motion_id` to that
dataset as described in the [user guide](user-guide.md). The licenses and
redistribution conditions of the source motions still apply to converted data.

The teaser motions are derived from [AMASS](https://amass.is.tue.mpg.de/) and
[Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
The four converted examples are included under `data/`.
[data/SOURCE.md](../data/SOURCE.md) maps them to their source sequences and
links to their usage conditions. To add more motions, see the
[motion preparation guide](motions.md).
