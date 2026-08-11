/* Small MPI-decomposed 1D stencil, C. See ../README.md for the matrix this
 * suite covers and how to build/run it under each compiler and MPI distro.
 *
 * Call structure (kept deliberately real and multi-level, not a flat
 * main->kernel hop, so the resulting call tree is worth profiling):
 *   main -> run_simulation -> step -> exchange_halo   (MPI point-to-point)
 *                                   -> launch_backend_kernel -> launch_hip_kernel / launch_omp_kernel
 *                                   -> reduce_residual (MPI collective)
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern void launch_omp_kernel(double *array, int n);
extern void launch_hip_kernel(double *array, int n);

static int rank, nranks;
static int left_neighbor, right_neighbor;

static void exchange_halo(double *local, int local_n) {
    MPI_Request reqs[4];
    MPI_Isend(&local[1],       1, MPI_DOUBLE, left_neighbor,  0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&local[local_n], 1, MPI_DOUBLE, right_neighbor, 1, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(&local[0],           1, MPI_DOUBLE, left_neighbor,  1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(&local[local_n + 1], 1, MPI_DOUBLE, right_neighbor, 0, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

static double reduce_residual(const double *local, int local_n) {
    double local_sum = 0.0;
    for (int i = 1; i <= local_n; ++i) {
        local_sum += fabs(local[i]);
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

static void launch_backend_kernel(const char *backend, double *local, int local_n) {
    if (strcmp(backend, "hip") == 0) {
        launch_hip_kernel(local, local_n + 2);
    } else if (strcmp(backend, "omp") == 0) {
        launch_omp_kernel(local, local_n + 2);
    } else {
        fprintf(stderr, "unknown backend '%s'\n", backend);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void step(const char *backend, double *local, int local_n) {
    exchange_halo(local, local_n);
    launch_backend_kernel(backend, local, local_n);
    double residual = reduce_residual(local, local_n);
    if (rank == 0) {
        printf("[%s] residual=%g\n", backend, residual);
    }
}

static void run_simulation(const char *backend, int local_n, int niter) {
    double *local = calloc((size_t)local_n + 2, sizeof(double));
    for (int i = 1; i <= local_n; ++i) {
        local[i] = 1.0 / (1.0 + i + rank * local_n);
    }
    for (int it = 0; it < niter; ++it) {
        step(backend, local, local_n);
    }
    free(local);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    left_neighbor  = (rank == 0)          ? MPI_PROC_NULL : rank - 1;
    right_neighbor = (rank == nranks - 1) ? MPI_PROC_NULL : rank + 1;

    const char *requested = (argc > 1) ? argv[1] : "all";
    int local_n = (argc > 2) ? atoi(argv[2]) : 1024;
    int niter   = (argc > 3) ? atoi(argv[3]) : 5;

    const char *all_backends[] = {"hip", "omp"};
    int n_all = 2;

    if (strcmp(requested, "all") == 0) {
        for (int b = 0; b < n_all; ++b) {
            run_simulation(all_backends[b], local_n, niter);
        }
    } else {
        run_simulation(requested, local_n, niter);
    }

    MPI_Finalize();
    return 0;
}
