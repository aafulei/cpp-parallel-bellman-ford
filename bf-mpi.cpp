// 17/11/01 = Wed

// MPI Bellman-Ford

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "mpi.h"

using std::cout;
using std::endl;
using std::string;

#define INF 1000000

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
    int N; 				//number of vertices
    int *mat;			// the adjacency matrix

    void abort_with_error_message(string msg) {
        std::cerr << msg << endl;
        abort();
    }

    //translate 2-dimension coordinate to 1-dimension
    int convert_dimension_2D_1D(int x, int y, int n) {
        return x * n + y;
    }

    int read_file(string filename) {
        std::ifstream inputf(filename, std::ifstream::in);
        if (!inputf.good()) {
            abort_with_error_message("ERROR OCCURRED WHILE READING INPUT FILE");
        }
        inputf >> N;
        //input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-processors)
        assert(N < (1024 * 1024 * 20));
        mat = (int *) malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                inputf >> mat[convert_dimension_2D_1D(i, j, N)];
            }
        return 0;
    }

    int print_result(bool has_negative_cycle, int *dist) {
        std::ofstream outputf("output.txt", std::ofstream::out);
        if (!has_negative_cycle) {
            for (int i = 0; i < N; i++) {
                if (dist[i] > INF)
                    dist[i] = INF;
                outputf << dist[i] << '\n';
            }
            outputf.flush();
        } else {
            outputf << "FOUND NEGATIVE CYCLE!" << endl;
        }
        outputf.close();
        return 0;
    }
}

void bellman_ford(int my_rank, int p, MPI_Comm comm, int n, int *mat, int *dist, bool *has_negative_cycle)
{
    MPI_Bcast(&n, 1, MPI_INT, 0, comm);
    int *my_dist;

    int q = n / p, r = n % p;				//q for quotient, r for remainder
    int load[p], begin[p];
    load[0] = q;
    for (int i = 1; i < p; ++i)
        load[i] = q + ((i <= r) ? 1 : 0);
    begin[0] = 0;
    for (int i = 1; i < p; ++i)
        begin[i] = begin[i - 1] + load[i - 1];

    int my_load = load[my_rank];
    int my_begin = begin[my_rank];
    int my_end = my_begin + my_load;
    int *my_work = new int[my_load];

    my_dist = (int *) malloc(n * sizeof(int));

    if (my_rank != 0)
        mat = new int[n*n];
    MPI_Bcast(mat, n*n, MPI_INT, 0, comm);

    for (int i = 0; i < n; i++)
        my_dist[i] = INF;

    my_dist[0] = 0;
    MPI_Barrier(comm);

    bool my_has_change;
    int my_iter_num = 0;
    for (int i = 0; i < n - 1; i++) {
        my_has_change = false;
        my_iter_num++;
        for (int u = 0; u < n; u++) {
            if (my_dist[u] == INF)
                continue;
            for (int v = my_begin; v < my_end; v++)
            {
                int weight = mat[u * n + v];
                if (weight < INF) {
                    if (my_dist[u] + weight < my_dist[v]) {
                        my_dist[v] = my_dist[u] + weight;
                        // CANNOT : my_work[v - my_begin] = my_dist[u] + weight;
                        my_has_change = true;
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &my_has_change, 1, MPI_C_BOOL, MPI_LOR, comm);
        if (!my_has_change)
            break;
        MPI_Allreduce(MPI_IN_PLACE, my_dist, n, MPI_INT, MPI_MIN, comm);
        // CANNOT : MPI_Allgatherv(my_work, my_load, MPI_INT, my_dist, load, begin, MPI_INT, comm); -- this requires synchronization;
        // NEED NOT : MPI_Barrier(comm);
    }

    if (my_iter_num == n - 1) {
        my_has_change = false;
        for (int u = 0; u < n; u++) {
            for (int v = my_begin; v < my_end; v++) {
                int weight = mat[u * n + v];
                if (weight < INF && my_dist[u] + weight < my_dist[v]) {
                    my_has_change = true;
                    break;
                }
            }
            if (my_has_change)
                break;
        }
        MPI_Allreduce(&my_has_change, has_negative_cycle, 1, MPI_C_BOOL, MPI_LOR, comm);
    }

    if(my_rank == 0)
        memcpy(dist, my_dist, n * sizeof(int));

    if (my_rank != 0)
        delete[] mat;
    delete[] my_dist;

    //------end of your code------
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    string filename = argv[1];

    int *dist;
    bool has_negative_cycle = false;

    //MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm comm;

    int p; 						//number of processors
    int my_rank; 				//my global rank
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    //only rank 0 process do the I/O
    if (my_rank == 0) {
        assert(utils::read_file(filename) == 0);
        dist = (int *) malloc(sizeof(int) * utils::N);
    }

    //time counter
    double t1, t2;
    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    //bellman-ford algorithm
    bellman_ford(my_rank, p, comm, utils::N, utils::mat, dist, &has_negative_cycle);
    MPI_Barrier(comm);

    //end timer
    t2 = MPI_Wtime();

    if (my_rank == 0) {
        std::cerr.setf(std::ios::fixed);
        std::cerr << std::setprecision(6) << "Time(s): " << (t2 - t1) << endl;
        utils::print_result(has_negative_cycle, dist);
        free(dist);
        free(utils::mat);
    }
    MPI_Finalize();
    return 0;
}