## Parallelizing Bellman-Ford Algorithm with Multiple Programming Models

[Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm) is useful in identifying arbitrage opportunities in foreign exchange markets. With an aim for speed, I parallelized the shortest-path algorithm using distributed-memory programming (MPI), shared-memory programming (OpenMP), and GPU programming (CUDA).

- Bellman-Ford in MPI: [bf-mpi.cpp](bf-mpi.cpp)
- Bellman-Ford in OpenMP: [bf-omp.cpp](bf-omp.cpp)
- Bellman-Ford in CUDA: [bf-cuda.cu](bf-cuda.cu)
- A helper program to generate test cases: [genmat.cpp](genmat.cpp)
