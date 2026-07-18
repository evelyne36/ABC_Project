# Environment migration status

`source-requirements.txt` is the original root requirements snapshot. Per-runner requirements are preserved under `experiments/`.

Before rerunning experiments, create a resolved lock and record Python, OS, CPU/RAM, compiler, CMake, OpenMP runtime, BLAS runtime, thread variables, build flags, and the loaded `libxgboost` SHA256.

