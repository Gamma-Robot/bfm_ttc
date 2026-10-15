# Motion data

The four G1 reference motions in `data/` work with the example configurations
as they are. [`data/SOURCE.md`](../data/SOURCE.md) lists their original
sequence IDs
## Prepare additional references

Obtain source captures from [AMASS](https://amass.is.tue.mpg.de/) or
[Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
under their usage terms, and retarget them to the 29-joint G1 with your own
pipeline. This repository does not include a retargeting tool for raw human
motion.

BFM-TTC reads datasets in the [Any4HDMI v2 format](https://github.com/EGalahad/any4hdmi/blob/main/docs/dataset_format.md),
defined by [Haoyang Weng's any4hdmi project](https://github.com/EGalahad/any4hdmi).
Each dataset contains `manifest.json` and `motions/<motion_id>.npz`.
Each NPZ contains a single floating-point `qpos` array of shape `(T, 36)`
with at least two frames:

- Columns 0–2: root position in metres, in world coordinates.
- Columns 3–6: normalized root quaternion in **w, x, y, z** order.
- Columns 7–35: G1 joint positions in radians, in the exact order declared by
  [`data/manifest.json`](../data/manifest.json).

If you already have a retargeted `qpos` array in this layout, run the following
from the repository root to create a local dataset:

```python
import json
from pathlib import Path
import numpy as np

root = Path("outputs/local-motions")
(root / "motions").mkdir(parents=True, exist_ok=True)
np.savez_compressed(root / "motions/walking.npz", qpos=qpos)
manifest = json.loads(Path("data/manifest.json").read_text())
manifest.update(dataset_name="local_g1_motions", timestep=0.02, num_motions=1)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
```

Set `timestep` to the actual sampling interval in seconds; the runtime resamples
the motion to the selected BFM's control period. If your retargeter uses a
different joint order or quaternion convention, convert the arrays to the
declared layout before saving them. The loader checks the array shape, finite
values, and quaternion normalization.

Then set `reference.dataset_root` and `reference.motion_id` to the new dataset
and motion, as shown in the [user guide](user-guide.md). Note that the G1
reference is the result of your retargeting and segmentation. The source
sequence ID only identifies the original human capture.
