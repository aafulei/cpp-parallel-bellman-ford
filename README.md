## Parallelizing Bellman-Ford Algorithm with Multiple Programming Models

[Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) is useful in identifying arbitrage opportunities in foreign exchange markets. With an aim for speed, I parallelized the shortest-path algorithm using distributed-memory programming (MPI), shared-memory programming (OpenMP), and GPU programming (CUDA).

- [`bf-mpi.cpp`](bf-mpi.cpp) Bellman-Ford in MPI
- [`bf-omp.cpp`](bf-omp.cpp) Bellman-Ford in OpenMP
- [`bf-cuda.cu`](bf-cuda.cu) Bellman-Ford in CUDA
- [`genmat.cpp`](genmat.cpp) A helper program to generate test cases
