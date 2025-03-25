module tortuosity_filcc_module

  ! Use AMReX modules for types and constants
  use amrex_fort_module, only : amrex_real, amrex_spacedim
  use amrex_bc_types_module, only : amrex_bc_ext_dir ! For boundary condition types
  use iso_c_binding, only : c_int ! Optional, can use default integer kind

  implicit none

  ! Parameters defining directions (consistent with C++ enum class Direction {X=0, Y=1, Z=2})
  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  ! Parameters defining cell types
  ! NOTE: Simplified to only blocked/free based on observed usage in kernels.
  !       If boundary types or bitflags are needed later, revise these definitions
  !       and update checking logic in relevant routines.
  integer, parameter :: cell_type_blocked = 0  ! Blocked cell (e.g., solid)
  integer, parameter :: cell_type_free    = 1  ! Free cell (e.g., conductive phase)

  ! Parameters defining components in MultiFabs (Fortran 1-based indices)
  ! WARNING: C++ calling code uses 0-based indices. Ensure correct mapping!
  integer, parameter :: comp_phase = 1  ! Component index for phase ID (maps to C++ index 0 in phase MultiFab)
  integer, parameter :: comp_phi   = 1  ! Component index for potential/concentration field (maps to C++ index 0 in solution MultiFab)
  integer, parameter :: comp_ct    = 2  ! Component index for cell type field (maps to C++ index 1 in solution MultiFab)

  ! Parameter for marking neighbors outside the physical domain in remspot
  integer, parameter :: neighbor_outside = -1

  private ! Default visibility
  ! Declare public the routines intended to be called from C++ via bind(c)
  public :: tortuosity_filct, tortuosity_remspot, tortuosity_filbc, tortuosity_filic

contains

!-----------------------------------------------------------------------
! Fills the cell type component based on phase information.
! Sets q(comp_ct) = cell_type_free if p(comp_phase) == phase, else cell_type_blocked.
!-----------------------------------------------------------------------
  subroutine tortuosity_filct(q, q_lo, q_hi, q_ncomp, p, p_lo, p_hi, p_ncomp, &
                              domlo, domhi, phase) &
                              bind(c, name='tortuosity_filct')

    implicit none

    ! Argument Declarations
    integer,          intent(in)    :: q_lo(3), q_hi(3)             ! Bounds of output array q
    integer,          intent(in)    :: p_lo(3), p_hi(3)             ! Bounds of input array p
    integer,          intent(in)    :: domlo(amrex_spacedim)        ! Domain lower corner
    integer,          intent(in)    :: domhi(amrex_spacedim)        ! Domain upper corner
    integer,          intent(in)    :: q_ncomp                      ! Number of components in q
    integer,          intent(in)    :: p_ncomp                      ! Number of components in p
    real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1), q_lo(2):q_hi(2), q_lo(3):q_hi(3), q_ncomp) ! Output array (cell types written to comp_ct)
    integer,          intent(in)    :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), p_ncomp) ! Input phase data (read from comp_phase)
    integer,          intent(in)    :: phase                        ! ID of the phase considered 'free'

    ! Local variables
    integer :: k, j, i
    integer :: ilo, ihi, jlo, jhi, klo, khi ! Domain index limits

    ! Input validation (Fortran 2008+)
    if (q_ncomp < comp_ct) error stop "tortuosity_filct: Output array q must have at least comp_ct components."
    if (p_ncomp < comp_phase) error stop "tortuosity_filct: Input array p must have at least comp_phase components."

    ! Extract domain limits for clarity
    ilo = domlo(1); ihi = domhi(1)
    jlo = domlo(2); jhi = domhi(2)
    klo = domlo(3); khi = domhi(3)

    ! Iterate over domain cells and fill cell type in component comp_ct of q
    ! Using DO CONCURRENT (F2008+) as iterations are independent
    DO CONCURRENT (k = klo, khi, j = jlo, jhi, i = ilo, ihi)
      if (p(i, j, k, comp_phase) == phase) then
        q(i, j, k, comp_ct) = cell_type_free
      else
        q(i, j, k, comp_ct) = cell_type_blocked
      end if
    END DO

  end subroutine tortuosity_filct

!-----------------------------------------------------------------------
! Removes disconnected single points (islands) from the phase MultiFab.
! Checks the 6 neighbors of a cell; if none match the cell's phase,
! the cell's phase is flipped (0->1 or 1->0). Operates in-place on q.
!-----------------------------------------------------------------------
  subroutine tortuosity_remspot(q, q_lo, q_hi, ncomp, bxlo, bxhi, domlo, domhi) &
                                bind(c, name='tortuosity_remspot')

    implicit none

    ! Argument Declarations
    integer, intent(in)    :: q_lo(3), q_hi(3)      ! Bounds of array q (incl. ghost cells)
    integer, intent(in)    :: domlo(3), domhi(3)    ! Domain index limits
    integer, intent(in)    :: bxlo(3), bxhi(3)     ! Valid box index limits to iterate over
    integer, intent(in)    :: ncomp               ! Number of components in q
    integer, intent(inout) :: q(q_lo(1):q_hi(1), q_lo(2):q_hi(2), q_lo(3):q_hi(3), ncomp) ! Phase data array (modified in-place at comp_phase)

    ! Local variables
    integer :: i, j, k             ! Running indices for the valid box
    integer :: neighbor_idx        ! Index for neighbor loop
    integer :: p_stencil(7)        ! Local array for 7-point stencil phase values
    logical :: is_connected        ! Flag to track if cell is connected

    ! Input validation (Fortran 2008+)
    if (ncomp < comp_phase) error stop "tortuosity_remspot: Input array q must have at least comp_phase components."

    ! Loop over the valid box defined by bxlo, bxhi
    ! Standard DO used here; neighbor reads make DO CONCURRENT complex to verify safely.
    do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          ! Get phase of center cell
          p_stencil(1) = q(i, j, k, comp_phase)

          ! Get phase of 6 neighbors, checking domain boundaries
          ! -X direction
          if ( i == domlo(1) ) then
            p_stencil(2) = neighbor_outside
          else
            p_stencil(2) = q(i-1, j, k, comp_phase)
          end if
          ! +X direction
          if ( i == domhi(1) ) then
            p_stencil(3) = neighbor_outside
          else
            p_stencil(3) = q(i+1, j, k, comp_phase)
          end if
          ! -Y direction
          if ( j == domlo(2) ) then
            p_stencil(4) = neighbor_outside
          else
            p_stencil(4) = q(i, j-1, k, comp_phase)
          end if
          ! +Y direction
          if ( j == domhi(2) ) then
            p_stencil(5) = neighbor_outside
          else
            p_stencil(5) = q(i, j+1, k, comp_phase)
          end if
          ! -Z direction
          if ( k == domlo(3) ) then
            p_stencil(6) = neighbor_outside
          else
            p_stencil(6) = q(i, j, k-1, comp_phase)
          end if
          ! +Z direction
          if ( k == domhi(3) ) then
            p_stencil(7) = neighbor_outside
          else
            p_stencil(7) = q(i, j, k+1, comp_phase)
          end if

          ! Check if the center cell phase matches any neighbor's phase
          is_connected = .false.
          do neighbor_idx = 2, 7
            if (p_stencil(1) == p_stencil(neighbor_idx)) then
              is_connected = .true.
              exit ! Found a connection, no need to check further
            end if
          end do

          ! If not connected to any neighbor, flip the phase (0->1 or 1->0)
          if (.not. is_connected) then
            if (q(i, j, k, comp_phase) == 0) then
              q(i, j, k, comp_phase) = 1
            else
              q(i, j, k, comp_phase) = 0
            end if
          end if

        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_remspot

!-----------------------------------------------------------------------
! Fills Dirichlet boundary condition values in ghost cells.
! Iterates through components and applies vlo/vhi based on bc flags.
!-----------------------------------------------------------------------
  subroutine tortuosity_filbc(q, q_lo, q_hi, ncomp, & ! Removed p, p_lo/hi, p_ncomp as unused
                              domlo, domhi, vlo, vhi, bc) &
                              bind(c, name='tortuosity_filbc')

    implicit none

    ! Argument Declarations
    integer,          intent(in)    :: q_lo(3), q_hi(3)      ! Bounds of array q (incl. ghost cells)
    integer,          intent(in)    :: domlo(amrex_spacedim) ! Domain lower corner
    integer,          intent(in)    :: domhi(amrex_spacedim) ! Domain upper corner
    integer,          intent(in)    :: ncomp               ! Number of components in q
    real(amrex_real), intent(in)    :: vlo                 ! Value at low boundary
    real(amrex_real), intent(in)    :: vhi                 ! Value at high boundary
    real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1), q_lo(2):q_hi(2), q_lo(3):q_hi(3), ncomp) ! Array to fill ghost cells
    integer,          intent(in)    :: bc(amrex_spacedim, 2) ! BC flags (bc(dim, side), side=1 for low, 2 for high)

    ! Local variables
    integer :: ilo, ihi, jlo, jhi, klo, khi ! Domain index limits
    integer :: i, j, k, n                  ! Loop indices
    integer :: imin, imax, jmin, jmax, kmin, kmax ! Ghost cell loop bounds

    ! Extract domain limits
    ilo = domlo(1); ihi = domhi(1)
    jlo = domlo(2); jhi = domhi(2)
    klo = domlo(3); khi = domhi(3)

    ! Loop over all components in the array q
    do n = 1, ncomp

      ! --- X direction boundaries ---
      if (q_lo(1) < ilo) then ! Fill low-X ghost cells
        imin = q_lo(1)
        imax = ilo - 1
        if (bc(1, 1) == amrex_bc_ext_dir) then ! Check if BC type is Dirichlet
          DO CONCURRENT (k = q_lo(3), q_hi(3), j = q_lo(2), q_hi(2), i = imin, imax)
            q(i, j, k, n) = vlo
          END DO
        end if
      end if
      if (q_hi(1) > ihi) then ! Fill high-X ghost cells
        imin = ihi + 1
        imax = q_hi(1)
        if (bc(1, 2) == amrex_bc_ext_dir) then
          DO CONCURRENT (k = q_lo(3), q_hi(3), j = q_lo(2), q_hi(2), i = imin, imax)
            q(i, j, k, n) = vhi
          END DO
        end if
      end if

#if AMREX_SPACEDIM >= 2
      ! --- Y direction boundaries ---
      if (q_lo(2) < jlo) then ! Fill low-Y ghost cells
        jmin = q_lo(2)
        jmax = jlo - 1
        if (bc(2, 1) == amrex_bc_ext_dir) then
          DO CONCURRENT (k = q_lo(3), q_hi(3), j = jmin, jmax, i = q_lo(1), q_hi(1))
             q(i, j, k, n) = vlo
          END DO
        end if
      end if
      if (q_hi(2) > jhi) then ! Fill high-Y ghost cells
        jmin = jhi + 1
        jmax = q_hi(2)
        if (bc(2, 2) == amrex_bc_ext_dir) then
          DO CONCURRENT (k = q_lo(3), q_hi(3), j = jmin, jmax, i = q_lo(1), q_hi(1))
             q(i, j, k, n) = vhi
          END DO
        end if
      end if
#endif

#if AMREX_SPACEDIM == 3
      ! --- Z direction boundaries ---
      if (q_lo(3) < klo) then ! Fill low-Z ghost cells
        kmin = q_lo(3)
        kmax = klo - 1
        if (bc(3, 1) == amrex_bc_ext_dir) then
          DO CONCURRENT (k = kmin, kmax, j = q_lo(2), q_hi(2), i = q_lo(1), q_hi(1))
             q(i, j, k, n) = vlo
          END DO
        end if
      end if
      if (q_hi(3) > khi) then ! Fill high-Z ghost cells
        kmin = khi + 1
        kmax = q_hi(3)
        if (bc(3, 2) == amrex_bc_ext_dir) then
          DO CONCURRENT (k = kmin, kmax, j = q_lo(2), q_hi(2), i = q_lo(1), q_hi(1))
             q(i, j, k, n) = vhi
          END DO
        end if
      end if
#endif

    end do ! n components

  end subroutine tortuosity_filbc

!-----------------------------------------------------------------------
! Fills the initial condition for the potential field (q).
! Applies a linear gradient between vlo and vhi along the specified direction (dir),
! but only within the cells belonging to the specified phase. Other cells are set to 0.
!-----------------------------------------------------------------------
  subroutine tortuosity_filic(q, q_lo, q_hi, ncomp, p, p_lo, p_hi, p_ncomp, &
                              lo, hi, domlo, domhi, vlo, vhi, phase, dir) &
                              bind(c, name='tortuosity_filic')

    implicit none

    ! Argument Declarations
    integer,          intent(in)    :: q_lo(3), q_hi(3)      ! Bounds of output array q
    integer,          intent(in)    :: p_lo(3), p_hi(3)      ! Bounds of input phase array p
    integer,          intent(in)    :: lo(3), hi(3)          ! Valid region loop bounds
    integer,          intent(in)    :: domlo(amrex_spacedim) ! Domain lower corner
    integer,          intent(in)    :: domhi(amrex_spacedim) ! Domain upper corner
    integer,          intent(in)    :: ncomp               ! Number of components in q
    integer,          intent(in)    :: p_ncomp               ! Number of components in p
    real(amrex_real), intent(in)    :: vlo                 ! Value at low boundary
    real(amrex_real), intent(in)    :: vhi                 ! Value at high boundary
    real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1), q_lo(2):q_hi(2), q_lo(3):q_hi(3), ncomp) ! Array to fill with initial condition
    integer,          intent(in)    :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), p_ncomp) ! Input phase data (read from comp_phase)
    integer,          intent(in)    :: phase               ! ID of the phase to initialize
    integer,          intent(in)    :: dir                 ! Direction for linear gradient (0, 1, or 2)

    ! Local variables
    integer :: i, j, k, n
    real(amrex_real) :: extent, factor, coord
    integer :: dom_lo_d, dom_hi_d ! Domain bounds in specified direction

    ! Input validation (Fortran 2008+)
    if (ncomp == 0) return ! Nothing to do if q has no components
    if (p_ncomp < comp_phase) error stop "tortuosity_filic: Input array p must have at least comp_phase components."
    if (dir < direction_x .or. dir > direction_z) error stop "tortuosity_filic: Invalid direction specified."

    ! Pre-calculate interpolation factor for the specified direction
    dom_lo_d = domlo(dir+1) ! +1 for Fortran 1-based dimension index
    dom_hi_d = domhi(dir+1)
    extent = real(dom_hi_d - dom_lo_d, amrex_real)

    if (abs(extent) < 1e-12_amrex_real) then
      factor = 0.0_amrex_real ! Avoid division by zero if domain is 1 cell wide
    else
      factor = 1.0_amrex_real / extent
    end if

    ! Loop over components and valid region (lo:hi)
    do n = 1, ncomp
      ! Using DO CONCURRENT (F2008+) as iterations are independent
      DO CONCURRENT (k = lo(3), hi(3), j = lo(2), hi(2), i = lo(1), hi(1))
        if (p(i, j, k, comp_phase) == phase) then
          ! Get coordinate along the specified direction
          select case (dir)
          case (direction_x); coord = real(i - dom_lo_d, amrex_real)
          case (direction_y); coord = real(j - dom_lo_d, amrex_real)
          case (direction_z); coord = real(k - dom_lo_d, amrex_real)
          end select
          ! Linear interpolation: vlo + (coordinate / extent) * (vhi - vlo)
          q(i, j, k, n) = vlo + coord * factor * (vhi - vlo)
        else
          q(i, j, k, n) = 0.0_amrex_real ! Set non-phase cells to zero
        end if
      END DO ! Concurrent i,j,k loop
    end do ! n component loop

  end subroutine tortuosity_filic

end module tortuosity_filcc_module
