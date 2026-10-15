# Third-party notices

The original BFM-TTC code is licensed under [Apache-2.0](LICENSE).
The notices below cover the upstream code and assets that BFM-TTC adapts or uses.

## hydrax

The CEM, predictive sampling, MPPI, MPPI-CMA, CBO, and DIAL modules in
`src/sampling_optimizers/sampling_optimizers/algs/` adapt
[hydrax](https://github.com/vincekurtz/hydrax) to BFM-TTC's NumPy
candidate-and-cost interface. DIAL uses hydrax's implementation of the
DIAL-MPC sampling method. The upstream MIT notice follows.

```text
MIT License

Copyright (c) 2024 Vince Kurtz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SPIDER

`src/sampling_optimizers/sampling_optimizers/algs/spider.py` adapts the
[SPIDER structured sampler](https://github.com/facebookresearch/spider/tree/44717007de41cbef7565dff7ff9f4453557a2d3d)
to NumPy and lower-is-better costs. This file keeps its upstream
**CC BY-NC 4.0** license. The upstream notice follows; see the
[full terms](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

```text
CC-by-NC License
Copyright (c) Meta Platforms, Inc. and affiliates.

You are free to:

Share — copy and share the work in any way

Adapt — change or build upon the work

But you must:

Give credit to the original creator

Not use the work for commercial purposes

You cannot:

Use the work for anything that is primarily intended for commercial advantage or monetary compensation

For full details, visit: https://creativecommons.org/licenses/by-nc/4.0/
```

## HEFT and G1 model

The HEFT interface and G1 experiment model use the
[official HEFT source](https://github.com/Axellwppr/motion_tracking/tree/0d5ba31e33397f3543d350d98b637e26d92f470a).
The upstream MIT notice is preserved in
[assets/robots/g1/LICENSE](assets/robots/g1/LICENSE). The model's source and
modifications are documented in [SOURCE.md](assets/robots/g1/SOURCE.md).
The robot preparation tool installs the notice next to the prepared assets.

## TWIST2

The TWIST2 observation and action interface follows the
[official implementation](https://github.com/amazon-far/TWIST2/tree/d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a).
Its upstream MIT notice is retained below.

```text
MIT License

Copyright (c) 2025 Yanjie Ze

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## SONIC

The SONIC observation and action interface follows
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
Its upstream source-code notice is retained below. The Apache-2.0 license
text is provided in the root [LICENSE](LICENSE).

```text
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
