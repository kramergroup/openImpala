module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real, amrex_abort

  implicit none

  private ! Default module visibility to private

  public :: tortuosity_fillmtx ! Make only the subroutine public

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  ! Component index for phase data in MultiFab 'p'
  integer, parameter :: comp_phase = 1

  ! Stencil indices (assuming typical HYPRE Struct convention: C, W, E, S, N, B, T)
  ! VERIFY THIS CONVENTION WITH YOUR HYPRE USAGE!
  integer, parameter :: istn_c  = 1 ! Center
  integer, parameter :: istn_mx = 2 ! -X (West)
  integer, parameter :: istn_px = 3 ! +X (East)
  integer, parameter :: istn_my = 4 ! -Y (South)
  integer, parameter :: istn_py = 5 ! +Y (North)
  integer, parameter :: istn_mz = 6 ! -Z (Bottom)
  integer, parameter :: istn_pz = 7 ! +Z (Top)
  integer, parameter :: nstencil = 7

contains

  ! ::: -----------------------------------------------------------
  ! ::: Fills HYPRE matrix coefficients for a Poisson equation within a box.
  ! ::: Handles internal phase boundaries (zero Neumann) and domain boundaries
  ! ::: perpendicular to flow (Dirichlet). Assumes uniform conductivity within the phase.
  ! :::
  ! ::: INPUTS:
  ! ::: nval        => number of points in the box (size of rhs, xinit)
  ! ::: p           => phase field data (4D array including component dim)
  ! ::: p_lo, p_hi  => index bounds of the phase field 'p' array
  ! ::: bxlo, bxhi  => index bounds of the current box to process
  ! ::: domlo,domhi => index bounds of the entire problem domain
  ! ::: dxinv       => array(3) of inverse grid spacings squared (1/dx^2, 1/dy^2, 1/dz^2)
  ! ::: vlo, vhi    => Dirichlet boundary values at lo/hi ends perpendicular to flow
  ! ::: phase       => index of the conductive phase
  ! ::: dir         => direction of flow (direction_x, direction_y, or direction_z)
  ! :::
  ! ::: OUTPUTS:
  ! ::: a           <= array(nval*nstencil) of matrix coefficients (flattened)
  ! ::: rhs         <= array(nval) of right-hand side values
  ! ::: xinit       <= array(nval) of initial guess values
  ! ::: -----------------------------------------------------------
  subroutine tortuosity_fillmtx(a, rhs, xinit, nval, p, p_lo, p_hi, &
                                bxlo, bxhi, domlo, domhi, dxinv, vlo, vhi, phase, dir) bind(c)

    ! Argument declarations
    integer,          intent(in)  :: nval
    real(amrex_real), intent(out) :: a(nval*nstencil), rhs(nval), xinit(nval)
    integer,          intent(in)  :: p_lo(3), p_hi(3), bxlo(3), bxhi(3), domlo(3), domhi(3)
    integer,          intent(in)  :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), *)
    real(amrex_real), intent(in)  :: dxinv(3) ! [1/dx^2, 1/dy^2, 1/dz^2]
    integer,          intent(in)  :: phase, dir
    real(amrex_real), intent(in)  :: vlo, vhi

    ! Local variables
    integer :: i, j, k, local_m
    integer :: len_x, len_y, len_z, expected_nval
    integer :: local_idx(nstencil)
    real(amrex_real) :: coeff_c, coeff_x, coeff_y, coeff_z
    real(amrex_real) :: domain_extent, factor

    ! Calculate box dimensions
    len_x = bxhi(1) - bxlo(1) + 1
    len_y = bxhi(2) - bxlo(2) + 1
    len_z = bxhi(3) - bxlo(3) + 1
    expected_nval = len_x * len_y * len_z

    ! Pre-calculate default stencil coefficients based on grid spacing
    coeff_x = dxinv(1)
    coeff_y = dxinv(2)
    coeff_z = dxinv(3)
    coeff_c = 2.0_amrex_real * (coeff_x + coeff_y + coeff_z) ! Center coeff for -Laplacian

    ! Check consistency between expected nval and passed nval (only if box not empty)
    if (expected_nval > 0 .and. expected_nval /= nval) then
        call amrex_abort("tortuosity_fillmtx: nval mismatch. Expected " // &
                         achar(int(log10(real(expected_nval)))+49) // ", got " // &
                         achar(int(log10(real(nval)))+49) )
    end if
    if (nval <= 0) return ! Nothing to do for empty box

    ! Use DO CONCURRENT (Fortran 2008) for potential parallelization
    ! Iterates over all points (i,j,k) within the specified box (bxlo:bxhi)
    DO CONCURRENT (k = bxlo(3), bxhi(3), j = bxlo(2), bxhi(2), i = bxlo(1), bxhi(1))

        ! Calculate the 1D index 'local_m' for the current point (i,j,k)
        ! Assumes Fortran ordering (i varies fastest)
        local_m = (i - bxlo(1)) + (j - bxlo(2)) * len_x + (k - bxlo(3)) * len_x * len_y + 1

        ! Calculate the starting index in the flat 'a' array for this point
        local_idx = (/ (nstencil*(local_m-1) + u, u=1,nstencil) /)

        ! --- Determine stencil based on phase ---

        if ( p(i,j,k,comp_phase) .ne. phase ) then

            ! Blocked cell (not the specified conductive phase)
            ! Force solution to zero: phi = 0  =>  1*phi = 0
            a(local_idx) = 0.0_amrex_real
            a(local_idx(istn_c)) = 1.0_amrex_real
            rhs(local_m) = 0.0_amrex_real

        else

            ! Fluid cell (conductive phase)
            ! Start with default stencil for -Laplacian(phi) = 0
            a(local_idx(istn_c))  =  coeff_c
            a(local_idx(istn_mx)) = -coeff_x
            a(local_idx(istn_px)) = -coeff_x
            a(local_idx(istn_my)) = -coeff_y
            a(local_idx(istn_py)) = -coeff_y
            a(local_idx(istn_mz)) = -coeff_z
            a(local_idx(istn_pz)) = -coeff_z
            rhs(local_m) = 0.0_amrex_real

            ! --- Modify stencil for internal phase boundaries (Zero Neumann) ---
            ! Check neighbors: if neighbor is blocked, modify stencil for zero flux across that face.

            ! -X face
            if ( p(i-1,j,k,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_x ! Center term absorbs flux
                a(local_idx(istn_mx)) = 0.0_amrex_real                   ! No connection to neighbor
            end if
            ! +X face
            if ( p(i+1,j,k,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_x
                a(local_idx(istn_px)) = 0.0_amrex_real
            end if
            ! -Y face
            if ( p(i,j-1,k,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_y
                a(local_idx(istn_my)) = 0.0_amrex_real
            end if
            ! +Y face
            if ( p(i,j+1,k,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_y
                a(local_idx(istn_py)) = 0.0_amrex_real
            end if
            ! -Z face
            if ( p(i,j,k-1,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_z
                a(local_idx(istn_mz)) = 0.0_amrex_real
            end if
             ! +Z face
            if ( p(i,j,k+1,comp_phase) .ne. phase ) then
                a(local_idx(istn_c))  = a(local_idx(istn_c))  - coeff_z
                a(local_idx(istn_pz)) = 0.0_amrex_real
            end if

        end if ! End of fluid/blocked check

        ! --- Overwrite stencil for Domain Boundaries Perpendicular to Flow (Dirichlet) ---
        ! This takes precedence over internal BC modifications if a cell is both.

        ! X-Direction Flow
        if ( dir == direction_x ) then
            if ( i == domlo(1) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vlo
            else if ( i == domhi(1) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vhi
            end if
        ! Y-Direction Flow
        else if ( dir == direction_y ) then
            if ( j == domlo(2) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vlo
            else if ( j == domhi(2) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vhi
            end if
        ! Z-Direction Flow
        else if ( dir == direction_z ) then
            if ( k == domlo(3) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vlo
            else if ( k == domhi(3) ) then
                a(local_idx) = 0.0_amrex_real
                a(local_idx(istn_c)) = 1.0_amrex_real
                rhs(local_m) = vhi
            end if
        end if ! End of Dirichlet BC overwrite

        ! --- Calculate Initial Guess ---
        ! Linear ramp between vlo and vhi along the flow direction

        if ( dir == direction_x ) then
            domain_extent = domhi(1) - domlo(1)
            if (abs(domain_extent) < 1.0e-12_amrex_real) then
                 factor = 0.0_amrex_real
            else
                 factor = 1.0_amrex_real / domain_extent
            end if
            xinit(local_m) = vlo + (vhi - vlo) * (i - domlo(1)) * factor
        else if ( dir == direction_y ) then
            domain_extent = domhi(2) - domlo(2)
             if (abs(domain_extent) < 1.0e-12_amrex_real) then
                 factor = 0.0_amrex_real
             else
                 factor = 1.0_amrex_real / domain_extent
             end if
             xinit(local_m) = vlo + (vhi - vlo) * (j - domlo(2)) * factor
        else if ( dir == direction_z ) then
             domain_extent = domhi(3) - domlo(3)
             if (abs(domain_extent) < 1.0e-12_amrex_real) then
                 factor = 0.0_amrex_real
             else
                 factor = 1.0_amrex_real / domain_extent
             end if
             xinit(local_m) = vlo + (vhi - vlo) * (k - domlo(3)) * factor
        else ! Should not happen if dir is 0, 1, or 2
             xinit(local_m) = 0.5_amrex_real * (vlo + vhi)
        end if

    END DO ! End DO CONCURRENT loop

  end subroutine tortuosity_fillmtx

end module tortuosity_poisson_3d_module
