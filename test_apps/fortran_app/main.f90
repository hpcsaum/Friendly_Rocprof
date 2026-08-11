! Small MPI-decomposed 1D stencil, Fortran. See ../README.md for the matrix
! this suite covers and how to build/run it under each compiler and MPI
! distro.
program main
  use mpi
  use sim_mod
  implicit none

  integer :: ierr
  character(len=16) :: requested
  integer :: local_n, niter
  character(len=16) :: all_backends(2)
  integer :: b

  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)

  if (rank == 0) then
    left_neighbor = MPI_PROC_NULL
  else
    left_neighbor = rank - 1
  end if
  if (rank == nranks - 1) then
    right_neighbor = MPI_PROC_NULL
  else
    right_neighbor = rank + 1
  end if

  call parse_args(requested, local_n, niter)

  all_backends(1) = "hip"
  all_backends(2) = "omp"

  if (trim(requested) == "all") then
    do b = 1, 2
      call run_simulation(trim(all_backends(b)), local_n, niter)
    end do
  else
    call run_simulation(trim(requested), local_n, niter)
  end if

  call MPI_Finalize(ierr)

contains

  subroutine parse_args(backend, n, iters)
    character(len=*), intent(out) :: backend
    integer, intent(out) :: n, iters
    character(len=32) :: arg
    integer :: nargs

    backend = "all"
    n = 1024
    iters = 5
    nargs = command_argument_count()
    if (nargs >= 1) then
      call get_command_argument(1, arg)
      backend = trim(arg)
    end if
    if (nargs >= 2) then
      call get_command_argument(2, arg)
      read(arg, *) n
    end if
    if (nargs >= 3) then
      call get_command_argument(3, arg)
      read(arg, *) iters
    end if
  end subroutine parse_args

end program main
