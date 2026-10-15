# G1 model source and modifications

The robot platform and its original geometry are from the Unitree G1.

This model uses the G1 assets published in
[Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking/tree/0d5ba31e33397f3543d350d98b637e26d92f470a/sim2real/config/g1/assets)
at revision `0d5ba31e33397f3543d350d98b637e26d92f470a`.
All 35 mesh files the model references are byte-for-byte
copies from that source, and their identities are pinned in the preparation
manifest.

`g1-mode_13_15.xml` is the BFM-TTC experiment model, derived from the same G1
description. We ship it because using the upstream deployment XML instead
would change the simulation. Compared with the pinned upstream
`sim2real/config/g1/assets/g1.xml`, it:

- Uses the six upstream 5010 wrist meshes and the experiment's wrist inertias,
  joint offsets, and wrist/hand collision geometry.
- Restores the experiment's shin collision geometry, foot sites, body set,
  and self-collision exclusions.
- Removes deployment-only mimic bodies, floor, lighting, textures, armatures,
  and motor declarations; retains the experiment's tracking camera and sensors.
- Leaves simulation timing and joint dynamics to the runtime robot
  configuration, which sets the experiment's timestep, armatures, friction,
  and torque limits. The runtime applies joint torques directly, so the model
  must not declare actuators.

The XML keeps the experiment's 29-joint order, masses, inertias, collision
geometry, and sensor definitions. Its SHA-256 is
`29a7ad71803d37d09f564bb1c9ae15e348a8c82b815c5d1ccbdde3f2f0521513`.
Before installing this XML, the preparation tool verifies the pinned upstream
XML and the identity of every source mesh. It installs no unreferenced meshes
and no other robot XML.
