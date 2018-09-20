/*
 * This is a openmp version of bellman_ford algorithm
 * Compile: g++ -std=c++11 -fopenmp -o openmp_bellman_ford openmp_bellman_ford.cpp
 * Run: ./openmp_bellman_ford <input file> <number of threads>, you will find the output file 'output.txt'
 * */

#include <string>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <sys/time.h>

#include "omp.h"

using std::string;
using std::cout;
using std::endl;

#define INF 1000000

/**
 * utils is a namespace for utility functions
 * including I/O (read input file and print results) and matrix dimension convert(2D->1D) function
 */
namespace utils {
    int N; //number of vertices
    int *mat; // the adjacency matrix

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
        //input matrix should be smaller than 20MB * 20MB (400MB, we don't have too much memory for multi-threads)
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
}//namespace utils

// you may add some helper functions here.

/**
 * Bellman-Ford algorithm. Find the shortest path from vertex 0 to other vertices.
 * @param p number of threads
 * @param n input size
 * @param *mat input adjacency matrix
 * @param *dist distance array
 * @param *has_negative_cycle a bool variable to recode if there are negative cycles
*/

void bellman_ford(int p, int n, int *mat, int *dist, bool *has_negative_cycle) {
    //------your code starts from here------

    // task allocation
    int q = n / p, r = n % p;
    int load[p], begin[p];
    load[0] = q;
    for (int i = 1; i < p; ++i)
        load[i] = q + ((i <= r) ? 1 : 0);
    begin[0] = 0;
    for (int i = 1; i < p; ++i)
        begin[i] = begin[i - 1] + load[i - 1];

    // initialization
    dist[0] = 0;
    for (int i = 1; i < n; ++i)
        dist[i] = INF;
    bool has_change = false;
    *has_negative_cycle = false;

    bool *relaxed_last_round = new bool[n];
    std::fill_n(relaxed_last_round, n, false);
    relaxed_last_round[0] = true;

    bool *relaxed_this_round = new bool[n];
    std::fill_n(relaxed_this_round, n, false);

    int *relaxed_times = new int[n];
    std::fill_n(relaxed_times, n, 0);
        
    #pragma omp parallel num_threads(p)
    {
        int my_rank = omp_get_thread_num();
        int my_load = load[my_rank];
        int my_begin = begin[my_rank];
        int my_end = my_begin + my_load;
        bool my_has_change = false;

        #pragma omp barrier

        for (size_t i = 0; ; ++i) {
            has_change = my_has_change = false;
            for (int u = 0; u < n; u++) {
                if (relaxed_last_round[u]) {
                    for (int v = my_begin; v < my_end; ++v) {
                        int weight = mat[u * n + v];
                        if (weight < INF)
                            if (dist[u] + weight < dist[v]) {
                                #pragma omp critical
                                dist[v] = dist[u] + weight;
                                #pragma omp critical
                                relaxed_times[v] += 1;
                                relaxed_this_round[v] = true;
                                my_has_change = true;
                                if (v == 0 && dist[v] < 0) {
                                    *has_negative_cycle = true;
                                }
                                if (relaxed_times[v] == n) {
                                    *has_negative_cycle = true;
                                }
                            }
                    }
                }
            }
            #pragma omp barrier         
            #pragma omp critical
            has_change = has_change || my_has_change;            
            #pragma omp barrier
            if (!has_change) {
                goto END;
            }
            if (*has_negative_cycle) {
                goto END;
            }
            #pragma omp barrier
            if (my_rank == 0) {
                std::swap(relaxed_last_round, relaxed_this_round);
                std::fill_n(relaxed_this_round, n, false);
            }
            #pragma omp barrier
        }
        END : {}
    }

    delete[] relaxed_last_round;
    delete[] relaxed_this_round;
    delete[] relaxed_times;

    //------end of your code------
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        utils::abort_with_error_message("INPUT FILE WAS NOT FOUND!");
    }
    if (argc <= 2) {
        utils::abort_with_error_message("NUMBER OF THREADS WAS NOT FOUND!");
    }
    string filename = argv[1];
    int p = atoi(argv[2]);

    int *dist;
    bool has_negative_cycle = false;

    assert(utils::read_file(filename) == 0);
    dist = (int *) malloc(sizeof(int) * utils::N);

    //time counter
    timeval start_wall_time_t, end_wall_time_t;
    float ms_wall;

    //start timer
    gettimeofday(&start_wall_time_t, nullptr);

    //bellman-ford algorithm
    bellman_ford(p, utils::N, utils::mat, dist, &has_negative_cycle);

    //end timer
    gettimeofday(&end_wall_time_t, nullptr);
    ms_wall = ((end_wall_time_t.tv_sec - start_wall_time_t.tv_sec) * 1000 * 1000
               + end_wall_time_t.tv_usec - start_wall_time_t.tv_usec) / 1000.0;

    std::cerr.setf(std::ios::fixed);
    std::cerr << std::setprecision(6) << "Time(s): " << (ms_wall/1000.0) << endl;
    utils::print_result(has_negative_cycle, dist);
    free(dist);
    free(utils::mat);

    return 0;
}