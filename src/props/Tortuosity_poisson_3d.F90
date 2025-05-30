module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real ! Use AMReX's real kind
  use iso_c_binding, only : c_int         ! For C integer type if needed for bind(c) robustness

  implicit none

  ! Parameters defining directions (consistent with C++ enum Direction {X=0, Y=1, Z=2})
  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  ! Parameters defining cell types
  integer, parameter :: cell_type_blocked = 0  ! Blocked cell (e.g., solid)
  integer, parameter :: cell_type_free    = 1  ! Free cell (e.g., conductive phase)

  ! Parameters defining components in the solution MultiFab (sol/p/n)
  integer, parameter :: comp_phi = 1  ! Component index for potential/concentration field (maps to C++ index 0)
  integer, parameter :: comp_ct  = 2  ! Component index for cell type field (maps to C++ index 1)

  private ! Default visibility
  public :: tortuosity_poisson_flux, tortuosity_poisson_update, tortuosity_poisson_fio ! Public routines

contains

!-----------------------------------------------------------------------
! Calculate face fluxes using finite difference on the solution field.
!-----------------------------------------------------------------------
  subroutine tortuosity_poisson_flux (lo, hi, fx, fxlo, fxhi, fy, fylo, fyhi, &
                                      fz, fzlo, fzhi, sol, slo, shi, dxinv) &
                                      bind(c, name='tortuosity_poisson_flux')

    ! Arguments:
    integer, dimension(3), intent(in) :: lo, hi      ! Valid cell index range for this box
    integer, dimension(3), intent(in) :: fxlo, fxhi  ! Bounds of fx array
    integer, dimension(3), intent(in) :: fylo, fyhi  ! Bounds of fy array
    integer, dimension(3), intent(in) :: fzlo, fzhi  ! Bounds of fz array
    integer, dimension(3), intent(in) :: slo, shi    ! Bounds of sol array (including ghost cells)
    real(amrex_real), dimension(fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3)), intent(out) :: fx ! X-face flux out
    real(amrex_real), dimension(fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3)), intent(out) :: fy ! Y-face flux out
    real(amrex_real), dimension(fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3)), intent(out) :: fz ! Z-face flux out
    ! Declare sol as 4D up to comp_ct
    real(amrex_real), dimension(slo(1): shi(1), slo(2): shi(2), slo(3): shi(3), comp_ct), intent(in) :: sol ! Solution array (phi, ct)
    real(amrex_real), dimension(3), intent(in) :: dxinv       ! Inverse cell sizes (1/dx, 1/dy, 1/dz)

    ! Local variables
    integer :: i, j, k
    real(amrex_real) :: dhx, dhy, dhz ! Inverse cell size aliases

    dhx = dxinv(1)
    dhy = dxinv(2)
    dhz = dxinv(3)

    ! Calculate X-fluxes on faces i=lo(1) to hi(1)+1
    ! FIX: Replace DO CONCURRENT with nested DO loops
    do k = lo(3), hi(3)
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)+1
          if ( (nint(sol(i,   j, k, comp_ct)) == cell_type_blocked) .or. &
               (nint(sol(i-1, j, k, comp_ct)) == cell_type_blocked) ) then
            fx(i, j, k) = 0.0_amrex_real
          else
            fx(i, j, k) = dhx * (sol(i, j, k, comp_phi) - sol(i-1, j, k, comp_phi))
          end if
        end do ! i
      end do ! j
    end do ! k

    ! Calculate Y-fluxes on faces j=lo(2) to hi(2)+1
    ! FIX: Replace DO CONCURRENT with nested DO loops
    do k = lo(3), hi(3)
      do j = lo(2), hi(2)+1
        do i = lo(1), hi(1)
          if ( (nint(sol(i, j,   k, comp_ct)) == cell_type_blocked) .or. &
               (nint(sol(i, j-1, k, comp_ct)) == cell_type_blocked) ) then
            fy(i, j, k) = 0.0_amrex_real
          else
            fy(i, j, k) = dhy * (sol(i, j, k, comp_phi) - sol(i, j-1, k, comp_phi))
          end if
        end do ! i
      end do ! j
    end do ! k

    ! Calculate Z-fluxes on faces k=lo(3) to hi(3)+1
    ! FIX: Replace DO CONCURRENT with nested DO loops
    do k = lo(3), hi(3)+1
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          if ( (nint(sol(i, j, k,   comp_ct)) == cell_type_blocked) .or. &
               (nint(sol(i, j, k-1, comp_ct)) == cell_type_blocked) ) then
            fz(i, j, k) = 0.0_amrex_real
          else
            fz(i, j, k) = dhz * (sol(i, j, k, comp_phi) - sol(i, j, k-1, comp_phi))
          end if
        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_poisson_flux

!-----------------------------------------------------------------------
! Update the solution field using an explicit time step (Forward Euler)
! based on the divergence of fluxes. Solves d(phi)/dt = Div(Flux).
!-----------------------------------------------------------------------
  subroutine tortuosity_poisson_update(lo, hi, p, plo, phi, n, nlo, nhi, &
                                       fx, fxlo, fxhi, fy, fylo, fyhi, fz, fzlo, fzhi, &
                                       ncomp, dxinv, dt) & ! <<< FIX: Added ncomp argument >>>
                                       bind(c, name='tortuosity_poisson_update')

    ! Arguments
    integer, dimension(3), intent(in) :: lo, hi      ! Valid cell index range to update
    integer, dimension(3), intent(in) :: plo, phi    ! Bounds of p array (old solution)
    integer, dimension(3), intent(in) :: nlo, nhi    ! Bounds of n array (new solution)
    integer, dimension(3), intent(in) :: fxlo, fxhi  ! Bounds of fx array
    integer, dimension(3), intent(in) :: fylo, fyhi  ! Bounds of fy array
    integer, dimension(3), intent(in) :: fzlo, fzhi  ! Bounds of fz array
    integer,              intent(in) :: ncomp       ! <<< FIX: Added ncomp argument >>>
    ! FIX: Declare p and n as 4D using ncomp
    real(amrex_real), dimension(plo(1):phi(1), plo(2):phi(2), plo(3):phi(3), ncomp), intent(in)  :: p ! Old solution (phi) @ comp_phi
    real(amrex_real), dimension(nlo(1):nhi(1), nlo(2):nhi(2), nlo(3):nhi(3), ncomp), intent(out) :: n ! New solution (phi) @ comp_phi
    real(amrex_real), dimension(fxlo(1):fxhi(1), fxlo(2):fxhi(2), fxlo(3):fxhi(3)), intent(in) :: fx ! X-face flux (in)
    real(amrex_real), dimension(fylo(1):fyhi(1), fylo(2):fyhi(2), fylo(3):fyhi(3)), intent(in) :: fy ! Y-face flux (in)
    real(amrex_real), dimension(fzlo(1):fzhi(1), fzlo(2):fzhi(2), fzlo(3):fzhi(3)), intent(in) :: fz ! Z-face flux (in)
    real(amrex_real), dimension(3), intent(in) :: dxinv       ! Inverse cell sizes (1/dx, 1/dy, 1/dz)
    real(amrex_real),               intent(in) :: dt          ! Timestep (or relaxation factor)

    ! Local variables
    real(amrex_real) :: dtdx(3)
    integer          :: i, j, k

    ! Input check
    if (ncomp < comp_phi) error stop "tortuosity_poisson_update: ncomp too small"

    dtdx(:) = dt * dxinv(:) ! Combine dt * (1/dx) factors

    ! Update loop: n = p + dt * Div(Flux)
    ! Div(Flux) = d(fx)/dx + d(fy)/dy + d(fz)/dz
    ! Approximated by [fx(i+1)-fx(i)]/dx + [fy(j+1)-fy(j)]/dy + [fz(k+1)-fz(k)]/dz
    ! FIX: Replace DO CONCURRENT with nested DO loops
    do k = lo(3), hi(3)
      do j = lo(2), hi(2)
        do i = lo(1), hi(1)
          ! FIX: Indexing now correct with 4D declaration
          n(i, j, k, comp_phi) = p(i, j, k, comp_phi) &
                 + dtdx(1) * (fx(i+1, j,   k  ) - fx(i, j, k)) & ! dt * d(fx)/dx
                 + dtdx(2) * (fy(i,   j+1, k  ) - fy(i, j, k)) & ! dt * d(fy)/dy
                 + dtdx(3) * (fz(i,   j,   k+1) - fz(i, j, k))     ! dt * d(fz)/dz
        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_poisson_update

!-----------------------------------------------------------------------
! Calculate total flux entering ('flux_in') and exiting ('flux_out')
! the domain boundaries owned by this box ('lo':'hi') in a given direction.
!-----------------------------------------------------------------------
  subroutine tortuosity_poisson_fio(lo, hi, fx, fxlo, fxhi, fy, fylo, fyhi, fz, fzlo, fzhi, &
                                   dir, flux_in, flux_out) &
                                   bind(c, name='tortuosity_poisson_fio')

    ! Arguments
    integer, dimension(3), intent(in) :: lo, hi      ! Valid cell index range for this box
    integer, dimension(3), intent(in) :: fxlo, fxhi  ! Bounds of fx array
    integer, dimension(3), intent(in) :: fylo, fyhi  ! Bounds of fy array
    integer, dimension(3), intent(in) :: fzlo, fzhi  ! Bounds of fz array
    real(amrex_real), dimension(fxlo(1):fxhi(1), fxlo(2):fxhi(2), fxlo(3):fxhi(3)), intent(in) :: fx ! X-face flux (in)
    real(amrex_real), dimension(fylo(1):fyhi(1), fylo(2):fyhi(2), fylo(3):fyhi(3)), intent(in) :: fy ! Y-face flux (in)
    real(amrex_real), dimension(fzlo(1):fzhi(1), fzlo(2):fzhi(2), fzlo(3):fzhi(3)), intent(in) :: fz ! Z-face flux (in)
    integer,               intent(in) :: dir         ! Direction (0, 1, or 2)
    real(amrex_real),      intent(out):: flux_in     ! Output: Sum of flux on the low face
    real(amrex_real),      intent(out):: flux_out    ! Output: Sum of flux on the high face

    ! Local variables
    integer :: i, j, k ! Loop indices

    ! Initialize output fluxes for this box
    flux_in  = 0.0_amrex_real
    flux_out = 0.0_amrex_real

    ! Sum fluxes on the relevant faces if this box owns the boundary data
    select case (dir)
    case (direction_x)
      ! Check if this box owns the data for the low face (index lo(1))
      if (fxlo(1) <= lo(1)) then
        do j = lo(2), hi(2) ! Iterate over Y indices
          do k = lo(3), hi(3) ! Iterate over Z indices
            flux_in = flux_in + fx(lo(1), j, k)
          end do
        end do
      end if
      ! Check if this box owns the data for the high face (index hi(1)+1)
      if (fxhi(1) >= hi(1)+1) then
         do j = lo(2), hi(2) ! Iterate over Y indices
           do k = lo(3), hi(3) ! Iterate over Z indices
             flux_out = flux_out + fx(hi(1)+1, j, k)
           end do
         end do
      end if

    case (direction_y)
      if (fylo(2) <= lo(2)) then
         do k = lo(3), hi(3) ! Iterate over Z indices
           do i = lo(1), hi(1) ! Iterate over X indices
             flux_in = flux_in + fy(i, lo(2), k)
           end do
         end do
      end if
      if (fyhi(2) >= hi(2)+1) then
         do k = lo(3), hi(3) ! Iterate over Z indices
           do i = lo(1), hi(1) ! Iterate over X indices
             flux_out = flux_out + fy(i, hi(2)+1, k)
           end do
         end do
      end if

    case (direction_z)
      if (fzlo(3) <= lo(3)) then
        do j = lo(2), hi(2) ! Iterate over Y indices
          do i = lo(1), hi(1) ! Iterate over X indices
            flux_in = flux_in + fz(i, j, lo(3))
          end do
        end do
      end if
      if (fzhi(3) >= hi(3)+1) then
        do j = lo(2), hi(2) ! Iterate over Y indices
          do i = lo(1), hi(1) ! Iterate over X indices
            flux_out = flux_out + fz(i, j, hi(3)+1)
          end do
        end do
      end if

    case default
      ! Add error handling for invalid direction (Requires F2008+)
      error stop "tortuosity_poisson_fio: Invalid direction specified."

    end select

  end subroutine tortuosity_poisson_fio

end module tortuosity_poisson_3d_module
