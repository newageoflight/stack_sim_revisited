"""

Redux of the stack article.

This time the code will be optimised for GPUs using CuPy and Numba, taking advantage of np.array ops
https://carpentries-incubator.github.io/gpu-speedups/01_CuPy_and_Numba_on_the_GPU/index.html

Broad outline of things to be done:
- The numba anneal function can be changed to use ufuncs
- cuda.jit the parts involving math ops on np.arrays (no need for large numbers, so a uint8 (or u1 for short) will be fine)

"""