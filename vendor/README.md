# vendor

`tensor-gpu.bend` is a byte-exact copy of bend-ml's `tensor-gpu/main.bend` (bend-ml-tensor-gpu 0.1.0.0, bend-ml commit
7d8ac2c), which is not on BendHub yet. Once it is published, `torch/gpu.bend` imports
`bend-ml-tensor-gpu@0.1.0.0/main.bend` instead and this folder goes away. `make laws` checks that the copy is unchanged
when bend-ml sits next to this repository (`../bend2-ml`).
