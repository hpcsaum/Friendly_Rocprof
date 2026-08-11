! Fortran interface to kernel_hip.cpp's C-linkage launch entry point,
! via iso_c_binding -- this is the same interop pattern real Cray-Fortran
! codes use to call HIP kernels from Fortran drivers.
module kernel_hip_binding
  use iso_c_binding, only: c_double, c_int
  implicit none

  interface
    subroutine launch_hip_kernel_c(array, n) bind(C, name="launch_hip_kernel")
      import :: c_double, c_int
      real(c_double), intent(inout) :: array(*)
      integer(c_int), value :: n
    end subroutine launch_hip_kernel_c
  end interface

contains

  subroutine launch_hip_kernel(array, n)
    integer(c_int), intent(in) :: n
    real(kind=8), intent(inout) :: array(0:n-1)
    call launch_hip_kernel_c(array, n)
  end subroutine launch_hip_kernel

end module kernel_hip_binding
