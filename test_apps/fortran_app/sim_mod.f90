! Simulation driver shared by both backends. Call structure (kept
! deliberately real and multi-level, not a flat main->kernel hop, so the
! resulting call tree is worth profiling):
!   run_simulation -> step -> exchange_halo         (MPI point-to-point)
!                           -> launch_backend_kernel -> launch_hip_kernel / launch_omp_kernel
!                           -> reduce_residual        (MPI collective)
module sim_mod
  use mpi
  use kernel_hip_binding, only: launch_hip_kernel
  use kernel_omp_mod,     only: launch_omp_kernel
  implicit none

  integer :: rank, nranks, left_neighbor, right_neighbor

contains

  subroutine exchange_halo(local, local_n)
    integer, intent(in) :: local_n
    real(kind=8), intent(inout) :: local(0:local_n+1)
    integer :: ierr
    integer :: reqs(4)
    integer :: statuses(MPI_STATUS_SIZE, 4)

    call MPI_Isend(local(1),         1, MPI_DOUBLE_PRECISION, left_neighbor,  0, MPI_COMM_WORLD, reqs(1), ierr)
    call MPI_Isend(local(local_n),   1, MPI_DOUBLE_PRECISION, right_neighbor, 1, MPI_COMM_WORLD, reqs(2), ierr)
    call MPI_Irecv(local(0),         1, MPI_DOUBLE_PRECISION, left_neighbor,  1, MPI_COMM_WORLD, reqs(3), ierr)
    call MPI_Irecv(local(local_n+1), 1, MPI_DOUBLE_PRECISION, right_neighbor, 0, MPI_COMM_WORLD, reqs(4), ierr)
    call MPI_Waitall(4, reqs, statuses, ierr)
  end subroutine exchange_halo

  function reduce_residual(local, local_n) result(global_sum)
    integer, intent(in) :: local_n
    real(kind=8), intent(in) :: local(0:local_n+1)
    real(kind=8) :: local_sum, global_sum
    integer :: ierr, i

    local_sum = 0.0d0
    do i = 1, local_n
      local_sum = local_sum + abs(local(i))
    end do
    call MPI_Allreduce(local_sum, global_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
  end function reduce_residual

  subroutine launch_backend_kernel(backend, local, local_n)
    character(len=*), intent(in) :: backend
    integer, intent(in) :: local_n
    real(kind=8), intent(inout) :: local(0:local_n+1)
    integer :: ierr

    select case (trim(backend))
    case ("hip")
      call launch_hip_kernel(local, local_n + 2)
    case ("omp")
      call launch_omp_kernel(local, local_n + 2)
    case default
      if (rank == 0) print *, "unknown backend: ", trim(backend)
      call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    end select
  end subroutine launch_backend_kernel

  subroutine step(backend, local, local_n)
    character(len=*), intent(in) :: backend
    integer, intent(in) :: local_n
    real(kind=8), intent(inout) :: local(0:local_n+1)
    real(kind=8) :: residual

    call exchange_halo(local, local_n)
    call launch_backend_kernel(backend, local, local_n)
    residual = reduce_residual(local, local_n)
    if (rank == 0) print *, "[", trim(backend), "] residual=", residual
  end subroutine step

  subroutine run_simulation(backend, local_n, niter)
    character(len=*), intent(in) :: backend
    integer, intent(in) :: local_n, niter
    real(kind=8), allocatable :: local(:)
    integer :: it, i

    allocate(local(0:local_n+1))
    local = 0.0d0
    do i = 1, local_n
      local(i) = 1.0d0 / (1.0d0 + real(i + rank * local_n, kind=8))
    end do

    do it = 1, niter
      call step(backend, local, local_n)
    end do

    deallocate(local)
  end subroutine run_simulation

end module sim_mod
