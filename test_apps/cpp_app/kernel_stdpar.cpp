/* Standard-parallelism ("stdpar") backend: std::for_each with an execution
 * policy, offloaded to the GPU by amdclang++'s --hipstdpar flag
 * (COMPILER=amd in the Makefile). Under COMPILER=gnu this still compiles
 * and runs -- on the CPU, via libstdc++'s parallel STL -- since GNU has no
 * GPU stdpar offload today; that's still useful for exercising the
 * call-tree shape and this backend's CPU-side runtime symbols. Cray's
 * compiler has no stdpar-offload path either, so it also runs CPU-only
 * under COMPILER=cray. */
#include <algorithm>
#include <execution>
#include <numeric>
#include <vector>

void launch_stdpar_kernel(double *array, int n) {
    std::vector<double> snapshot(array, array + n);
    std::vector<int> indices(n - 2);
    std::iota(indices.begin(), indices.end(), 1);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                  [array, snapshot = snapshot.data()](int i) {
                      array[i] = 0.5 * (snapshot[i - 1] + snapshot[i + 1]);
                  });
}
