# Third-party Reference Code

This directory contains vendored reference implementations used for local
research and adaptation. They are committed as ordinary source trees, not git
submodules, so cloud training machines can pull a single repository.

## MaskGIT

- Path: `third_party/maskgit`
- Upstream: https://github.com/google-research/maskgit
- Snapshot: `1db23594e1bd328ee78eadcd148a19281cd0f5b8`
- License: Apache-2.0
- Purpose: official JAX implementation of MaskGIT for algorithm reference.

## MaskGIT PyTorch

- Path: `third_party/maskgit-pytorch`
- Upstream: https://github.com/xyfJASON/maskgit-pytorch
- Snapshot: `ca5c7c39cf55d49fb034d7348595c9c3abc2c4ed`
- License: MIT
- Purpose: PyTorch implementation to guide the mask token prior implementation.
