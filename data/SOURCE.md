# Demonstration motion sources

The four G1 reference motions are sampled at 50 Hz. The three numbered AMASS
examples contain the complete converted `qpos` sequences. `manifest.json`
declares the 36 coordinates and the robot model.

| Local ID | Description of the retargeted motion | Original motion ID | Duration |
| --- | --- | --- | --- |
| `0019` | Left sidestep | `ACCAD/Female1Walking_c3d/B22 - side step left_poses/motion_seg1` | 9.42 s |
| `walk` | Walking | `LAFAN1/walk1_subject1` (retargeted G1 sequence) | 246.62 s; demo executes 10 s |
| `1187` | Backward walking | `KIT/4/WalkingStraightBackwards06_poses/motion_seg1` | 6.78 s |
| `01` | Low, crouched posture | `CMU/22_23_justin/22_03_poses/motion_seg1` | 7.02 s |


Sources: [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset),
[AMASS](https://amass.is.tue.mpg.de/),
[ACCAD motion capture data](https://accad.osu.edu/research/motion-lab/mocap-system-and-data),
[KIT motion database](https://motion-database.humanoids.kit.edu/), and
[CMU motion capture database](https://mocap.cs.cmu.edu/).
