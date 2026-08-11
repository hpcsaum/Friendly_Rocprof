! OpenMP target-offload backend: a plain compilation unit, no MPI calls, so
! it can be built with the bare compiler ($(FC) in the Makefile) regardless
! of which MPI wrapper the rest of the app uses.
module kernel_omp_mod
  implicit none
contains
  subroutine launch_omp_kernel(array, n)
    integer, intent(in) :: n
    real(kind=8), intent(inout) :: array(0:n-1)
    integer :: i

    !$omp target teams distribute parallel do map(tofrom: array)
    do i = 1, n - 2
      array(i) = 0.5d0 * (array(i - 1) + array(i + 1))
    end do
    !$omp end target teams distribute parallel do
  end subroutine launch_omp_kernel
end module kernel_omp_mod
