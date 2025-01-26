### CPP Autograd
CPP autograd is a simple autograd engine written in C++.
Its syntax aims to mirrow LibTorch.

For some examples please see `grad/demo/` or the Tutorial section below.

## Tutorial


## Speedup
two areas of speedup:
1) do the matrix multiplication on multiple cores
2) distributed training

WIP: paralelization on multiple cores, adding a cuda kernel

## Usage

## design choices
make no diff when broadcasting
tensors having ids
should to backward with topological sorting i think