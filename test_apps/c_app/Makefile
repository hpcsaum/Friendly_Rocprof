# Builds the same source under three compiler toolchains, selected via
# COMPILER=gnu|cray|amd (default gnu). MPI distribution (OpenMPI vs
# CrayMPI/MPICH) is NOT a build-time choice -- it's whichever MPI module is
# loaded when the wrapper compilers below resolve, and which launch command
# (mpirun/srun) is used at run time -- see ../README.md.
#
# Override any variable on the command line if your site's names/paths
# differ from the defaults, e.g.:
#   make COMPILER=amd OFFLOAD_ARCH=gfx942

COMPILER ?= gnu
OFFLOAD_ARCH ?= gfx90a

ifeq ($(COMPILER),cray)
  # CC has a built-in GNU Make default ('cc') -- plain '=' here (not '?=')
  # is required to actually override it; command-line `make CC=...` still
  # wins over this regardless, since command-line variables always beat
  # in-makefile assignments.
  CC     = cc
  MPICC  ?= cc
  MPICXX ?= CC
  OMP_OFFLOAD_FLAGS ?= -fopenmp
else ifeq ($(COMPILER),amd)
  CC     = amdclang
  MPICC  ?= mpicc
  MPICXX ?= mpicxx
  OMP_OFFLOAD_FLAGS ?= -fopenmp --offload-arch=$(OFFLOAD_ARCH)
else ifeq ($(COMPILER),gnu)
  CC     = gcc
  MPICC  ?= mpicc
  MPICXX ?= mpicxx
  # GCC's AMDGPU OpenMP-offload support depends on the toolchain being built
  # with amdgcn offload enabled -- if this errors on your system, override
  # OMP_OFFLOAD_FLAGS to drop the -foffload bits; the kernel still runs
  # correctly on the CPU (still useful for exercising the call-tree shape).
  OMP_OFFLOAD_FLAGS ?= -fopenmp -foffload=amdgcn-amdhsa -foffload-options=-march=$(OFFLOAD_ARCH)
else
  $(error unknown COMPILER '$(COMPILER)' -- expected gnu, cray, or amd)
endif

HIPCC ?= hipcc
EXTRA_LDFLAGS ?= -lamdhip64

TARGET := c_app
OBJS := main.o kernel_omp.o kernel_hip.o
CFLAGS ?= -O2

.PHONY: all clean check-tools

all: check-tools $(TARGET)

check-tools:
	@command -v $(MPICC) >/dev/null 2>&1 || { echo "error: MPI C compiler '$(MPICC)' not found on PATH -- load the right compiler/MPI module for COMPILER=$(COMPILER)" >&2; exit 1; }
	@command -v $(CC) >/dev/null 2>&1 || { echo "error: C compiler '$(CC)' not found on PATH -- load the right compiler module for COMPILER=$(COMPILER)" >&2; exit 1; }
	@command -v $(HIPCC) >/dev/null 2>&1 || { echo "error: '$(HIPCC)' not found on PATH -- load a ROCm module" >&2; exit 1; }

main.o: main.c
	$(MPICC) $(CFLAGS) -c $< -o $@

kernel_omp.o: kernel_omp.c
	$(CC) $(CFLAGS) $(OMP_OFFLOAD_FLAGS) -c $< -o $@

kernel_hip.o: kernel_hip.cpp
	$(HIPCC) $(CFLAGS) -c $< -o $@

# $(OMP_OFFLOAD_FLAGS) must be repeated at final link, not just at
# kernel_omp.o's compile step -- Clang/amdclang only run the device-code
# linking step (which pulls in the offload runtime, e.g. libomptarget's
# __tgt_target_kernel) when -fopenmp/--offload-arch are present on the link
# command itself. Omitting them here fails with "undefined symbol:
# __tgt_target_kernel" even though the object file compiled cleanly.
$(TARGET): $(OBJS)
	$(MPICXX) $(OBJS) -o $@ $(OMP_OFFLOAD_FLAGS) $(EXTRA_LDFLAGS)

clean:
	rm -f $(TARGET) $(OBJS)
