"""Repository-wide test defaults.

MuJoCo's default GLFW backend requires an X display; EGL renders offscreen
on the GPU. Set the default before any test imports mujoco so rendering
tests pass on headless machines. An explicit non-empty MUJOCO_GL always
wins; mujoco treats an empty value as GLFW, so empty counts as unset here.
"""

import os

if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"
