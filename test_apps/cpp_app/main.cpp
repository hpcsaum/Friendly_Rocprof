/* Small MPI-decomposed 1D stencil, C++. See ../README.md for the matrix
 * this suite covers and how to build/run it under each compiler and MPI
 * distro.
 *
 * Call structure (kept deliberately real and multi-level, not a flat
 * main->kernel hop, so the resulting call tree is worth profiling):
 *   main -> run_simulation -> step -> exchange_halo   (MPI point-to-point)
 *                                   -> launch_backend_kernel -> launch_hip_kernel / launch_omp_kernel / launch_stdpar_kernel
 *                                   -> reduce_residual (MPI collective)
 */
#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdlib>

void launch_hip_kernel(double *array, int n);
void launch_omp_kernel(double *array, int n);
void launch_stdpar_kernel(double *array, int n);

static int rank = 0, nranks = 1;
static int left_neighbor = MPI_PROC_NULL, right_neighbor = MPI_PROC_NULL;

static void exchange_halo(std::vector<double> &local, int local_n) {
    MPI_Request reqs[4];
    MPI_Isend(&local[1],       1, MPI_DOUBLE, left_neighbor,  0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&local[local_n], 1, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(&local[0],           1, MPI_DOUBLE, left_neighbor,  1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(&local[local_n + 1], 1, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

static double reduce_residual(const std::vector<double> &local, int local_n) {
    double local_sum = 0.0;
    for (int i = 1; i <= local_n; ++i) {
        local_sum += std::fabs(local[i]);
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

static void launch_backend_kernel(const std::string &backend, std::vector<double> &local, int local_n) {
    if (backend == "hip") {
        launch_hip_kernel(local.data(), local_n + 2);
    } else if (backend == "omp") {
        launch_omp_kernel(local.data(), local_n + 2);
    } else if (backend == "stdpar") {
        launch_stdpar_kernel(local.data(), local_n + 2);
    } else {
        std::cerr << "unknown backend '" << backend << "'\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void step(const std::string &backend, std::vector<double> &local, int local_n) {
    exchange_halo(local, local_n);
    launch_backend_kernel(backend, local, local_n);
    double residual = reduce_residual(local, local_n);
    if (rank == 0) {
        std::cout << "[" << backend << "] residual=" << residual << "\n";
    }
}

static void run_simulation(const std::string &backend, int local_n, int niter) {
    std::vector<double> local(local_n + 2, 0.0);
    for (int i = 1; i <= local_n; ++i) {
        local[i] = 1.0 / (1.0 + i + rank * local_n);
    }
    for (int it = 0; it < niter; ++it) {
        step(backend, local, local_n);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    left_neighbor  = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    right_neighbor = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;

    std::string requested = (argc > 1) ? argv[1] : "all";
    int local_n = (argc > 2) ? std::atoi(argv[2]) : 1024;
    int niter   = (argc > 3) ? std::atoi(argv[3]) : 5;

    const std::vector<std::string> all_backends = {"hip", "omp", "stdpar"};

    if (requested == "all") {
        for (const auto &backend : all_backends) {
            run_simulation(backend, local_n, niter);
        }
    } else {
        run_simulation(requested, local_n, niter);
    }

    MPI_Finalize();
    return 0;
}
