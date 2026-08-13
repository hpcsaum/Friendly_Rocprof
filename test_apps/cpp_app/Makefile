# Builds the same source under three compiler toolchains, selected via
# COMPILER=gnu|cray|amd (default gnu). See ../README.md for the MPI
# distribution axis (build-time/launch-time only, no source changes).
#
# Override any variable on the command line for your site, e.g.:
#   make COMPILER=amd OFFLOAD_ARCH=gfx942

COMPILER ?= gnu
OFFLOAD_ARCH ?= gfx90a

ifeq ($(COMPILER),cray)
  MPICXX ?= CC
  OMP_OFFLOAD_FLAGS ?= -fopenmp
  STDPAR_CXXFLAGS ?=
  STDPAR_LDFLAGS ?=
else ifeq ($(COMPILER),amd)
  MPICXX  ?= mpicxx
  CXX_amd ?= amdclang++
  OMP_OFFLOAD_FLAGS ?= -fopenmp --offload-arch=$(OFFLOAD_ARCH)
  STDPAR_CXXFLAGS ?= --hipstdpar --offload-arch=$(OFFLOAD_ARCH)
  # --hipstdpar is needed to COMPILE kernel_stdpar.cpp (above), but must NOT
  # also be passed at final link alongside -fopenmp/--offload-arch: on
  # amdclang++/ROCm, combining both offload kinds on one link line breaks
  # OpenMP's own device-code linking step (undefined symbol:
  # __tgt_target_kernel), even though the already-compiled kernel_stdpar.o
  # links in and runs correctly on the GPU without it -- confirmed via a
  # real HPC build+run, kernel launch visible in rocprofv3's kernel-trace
  # output (a "kernel" dispatch alongside HIP's "stencil_kernel" and OMP's
  # "__omp_offloading_..." dispatch, one per iteration).
  STDPAR_LDFLAGS ?=
else ifeq ($(COMPILER),gnu)
  MPICXX ?= mpicxx
  # GCC's AMDGPU OpenMP-offload support depends on the toolchain being built
  # with amdgcn offload enabled -- if this errors on your system, override
  # OMP_OFFLOAD_FLAGS to drop the -foffload bits; the kernel still runs
  # correctly on the CPU (still useful for exercising the call-tree shape).
  OMP_OFFLOAD_FLAGS ?= -fopenmp -foffload=amdgcn-amdhsa -foffload-options=-march=$(OFFLOAD_ARCH)
  # GNU has no GPU stdpar offload today -- this runs on the CPU instead, via
  # libstdc++'s parallel STL backend (needs TBB linked in).
  STDPAR_CXXFLAGS ?=
  STDPAR_LDFLAGS ?= -ltbb
else
  $(error unknown COMPILER '$(COMPILER)' -- expected gnu, cray, or amd)
endif

HIPCC ?= hipcc
EXTRA_LDFLAGS ?= -lamdhip64

TARGET := cpp_app
OBJS := main.o kernel_omp.o kernel_hip.o kernel_stdpar.o
CXXFLAGS ?= -O2 -std=c++17

.PHONY: all clean check-tools

all: check-tools $(TARGET)

check-tools:
	@command -v $(MPICXX) >/dev/null 2>&1 || { echo "error: MPI C++ compiler '$(MPICXX)' not found on PATH -- load the right compiler/MPI module for COMPILER=$(COMPILER)" >&2; exit 1; }
	@command -v $(HIPCC) >/dev/null 2>&1 || { echo "error: '$(HIPCC)' not found on PATH -- load a ROCm module" >&2; exit 1; }
ifeq ($(COMPILER),amd)
	@command -v $(CXX_amd) >/dev/null 2>&1 || { echo "error: '$(CXX_amd)' not found on PATH -- load a ROCm/amd compiler module" >&2; exit 1; }
endif

main.o: main.cpp
	$(MPICXX) $(CXXFLAGS) -c $< -o $@

kernel_hip.o: kernel_hip.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

ifeq ($(COMPILER),amd)
kernel_omp.o: kernel_omp.cpp
	$(CXX_amd) $(CXXFLAGS) $(OMP_OFFLOAD_FLAGS) -c $< -o $@

kernel_stdpar.o: kernel_stdpar.cpp
	$(CXX_amd) $(CXXFLAGS) $(STDPAR_CXXFLAGS) -c $< -o $@
else
kernel_omp.o: kernel_omp.cpp
	$(MPICXX) $(CXXFLAGS) $(OMP_OFFLOAD_FLAGS) -c $< -o $@

kernel_stdpar.o: kernel_stdpar.cpp
	$(MPICXX) $(CXXFLAGS) $(STDPAR_CXXFLAGS) -c $< -o $@
endif

# $(OMP_OFFLOAD_FLAGS) must be repeated at final link, not just at
# kernel_omp.o's compile step -- Clang/amdclang++ only run the device-code
# linking step (which pulls in the offload runtime, e.g. libomptarget's
# __tgt_target_kernel) when -fopenmp/--offload-arch are present on the link
# command itself. Omitting them here fails with "undefined symbol:
# __tgt_target_kernel" even though the object file compiled cleanly.
$(TARGET): $(OBJS)
	$(MPICXX) $(OBJS) -o $@ $(OMP_OFFLOAD_FLAGS) $(EXTRA_LDFLAGS) $(STDPAR_LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)
