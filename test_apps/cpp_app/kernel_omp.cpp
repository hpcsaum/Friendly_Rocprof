/* OpenMP target-offload backend: a plain compilation unit, no MPI calls, so
 * it can be built with the bare compiler regardless of which MPI wrapper
 * the rest of the app uses. */

void launch_omp_kernel(double *array, int n) {
    #pragma omp target teams distribute parallel for map(tofrom: array[0:n])
    for (int i = 1; i < n - 1; ++i) {
        array[i] = 0.5 * (array[i - 1] + array[i + 1]);
    }
}
