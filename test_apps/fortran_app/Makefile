# Builds the same source under three compiler toolchains, selected via
# COMPILER=gnu|cray|amd (default gnu). See ../README.md for the MPI
# distribution axis (build-time/launch-time only, no source changes).
#
# Override any variable on the command line for your site, e.g.:
#   make COMPILER=amd OFFLOAD_ARCH=gfx942

COMPILER ?= gnu
OFFLOAD_ARCH ?= gfx90a

ifeq ($(COMPILER),cray)
  # FC has a built-in GNU Make default ('f77') -- plain '=' here (not '?=')
  # is required to actually override it; command-line `make FC=...` still
  # wins over this regardless, since command-line variables always beat
  # in-makefile assignments.
  FC    = ftn
  MPIFC ?= ftn
  OMP_OFFLOAD_FLAGS ?= -fopenmp
else ifeq ($(COMPILER),amd)
  FC    = amdflang
  MPIFC ?= mpif90
  OMP_OFFLOAD_FLAGS ?= -fopenmp --offload-arch=$(OFFLOAD_ARCH)
else ifeq ($(COMPILER),gnu)
  FC    = gfortran
  MPIFC ?= mpif90
  # GCC's AMDGPU OpenMP-offload support depends on the toolchain being built
  # with amdgcn offload enabled -- if this errors on your system, override
  # OMP_OFFLOAD_FLAGS to drop the -foffload bits; the kernel still runs
  # correctly on the CPU (still useful for exercising the call-tree shape).
  OMP_OFFLOAD_FLAGS ?= -fopenmp -foffload=amdgcn-amdhsa -foffload-options=-march=$(OFFLOAD_ARCH)
else
  $(error unknown COMPILER '$(COMPILER)' -- expected gnu, cray, or amd)
endif

HIPCC ?= hipcc
# Final link uses the Fortran MPI wrapper, so the HIP runtime and the C++
# runtime the hipcc-compiled kernel object depends on both need to be named
# explicitly (a C++ link driver would pull libstdc++ in automatically; a
# Fortran one doesn't).
EXTRA_LDFLAGS ?= -lamdhip64 -lstdc++

TARGET := fortran_app
OBJS := kernel_hip.o kernel_omp.o kernel_hip_binding.o sim_mod.o main.o
FFLAGS ?= -O2

.PHONY: all clean check-tools

all: check-tools $(TARGET)

check-tools:
	@command -v $(MPIFC) >/dev/null 2>&1 || { echo "error: MPI Fortran compiler '$(MPIFC)' not found on PATH -- load the right compiler/MPI module for COMPILER=$(COMPILER)" >&2; exit 1; }
	@command -v $(FC) >/dev/null 2>&1 || { echo "error: Fortran compiler '$(FC)' not found on PATH -- load the right compiler module for COMPILER=$(COMPILER)" >&2; exit 1; }
	@command -v $(HIPCC) >/dev/null 2>&1 || { echo "error: '$(HIPCC)' not found on PATH -- load a ROCm module" >&2; exit 1; }

kernel_hip.o: kernel_hip.cpp
	$(HIPCC) -O2 -c $< -o $@

kernel_omp.o: kernel_omp.f90
	$(FC) $(FFLAGS) $(OMP_OFFLOAD_FLAGS) -c $< -o $@

kernel_hip_binding.o: kernel_hip_binding.f90
	$(MPIFC) $(FFLAGS) -c $< -o $@

sim_mod.o: sim_mod.f90 kernel_hip_binding.o kernel_omp.o
	$(MPIFC) $(FFLAGS) -c $< -o $@

main.o: main.f90 sim_mod.o
	$(MPIFC) $(FFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(MPIFC) $(OBJS) -o $@ $(EXTRA_LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS) *.mod
