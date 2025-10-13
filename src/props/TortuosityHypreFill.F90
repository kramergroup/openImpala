module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  use amrex_error_module, only : amrex_abort

  implicit none

  private ! Default module visibility to private

  public :: tortuosity_fillmtx ! Make only the subroutine public

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  integer, parameter :: comp_phase = 1
  integer, parameter :: comp_mask = 1

  ! Stencil indices (0-based, matching C++ HYPRE_StructStencilSetElement order)
  integer, parameter :: istn_c  = 0 ! Center
  integer, parameter :: istn_mx = 1 ! -X (West)
  integer, parameter :: istn_px = 2 ! +X (East)
  integer, parameter :: istn_my = 3 ! -Y (South)
  integer, parameter :: istn_py = 4 ! +Y (North)
  integer, parameter :: istn_mz = 5 ! -Z (Bottom)
  integer, parameter :: istn_pz = 6 ! +Z (Top)
  integer, parameter :: nstencil = 7

  ! Activity mask values (assuming C++ convention)
  integer, parameter :: cell_inactive = 0
  integer, parameter :: cell_active = 1

  ! Tolerance for floating point comparisons
  real(amrex_real), parameter :: small_real = 1.0e-15_amrex_real

contains

  ! ::: -----------------------------------------------------------
  ! ::: Fills HYPRE matrix coefficients for a Poisson equation.
  ! ::: Constructs the matrix for cells that belong to the specified 'phase'
  ! ::: AND are marked as active in the 'active_mask' (i.e., part of a
  ! ::: percolating path).
  ! ::: -----------------------------------------------------------
  subroutine tortuosity_fillmtx(a, rhs, xinit, nval, &
                                p, p_lo, p_hi, &
                                active_mask, mask_lo, mask_hi, &
                                bxlo, bxhi, domlo, domhi, dxinv, vlo, vhi, phase, dir, &
                                debug_print_level) bind(c)

    ! Argument declarations
    integer,            intent(in)  :: nval
    real(amrex_real),   intent(out) :: a(0:nval*nstencil-1), rhs(nval), xinit(nval)
    integer,            intent(in)  :: p_lo(3), p_hi(3)
    integer,            intent(in)  :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), *)
    integer,            intent(in)  :: mask_lo(3), mask_hi(3)
    integer,            intent(in)  :: active_mask(mask_lo(1):mask_hi(1), &
                                                     mask_lo(2):mask_hi(2), &
                                                     mask_lo(3):mask_hi(3))
    integer,            intent(in)  :: bxlo(3), bxhi(3)
    integer,            intent(in)  :: domlo(3), domhi(3)
    real(amrex_real),   intent(in)  :: dxinv(3) ! [1/dx^2, 1/dy^2, 1/dz^2]
    real(amrex_real),   intent(in)  :: vlo, vhi
    integer,            intent(in)  :: phase, dir ! The phase ID to solve for
    integer,            intent(in)  :: debug_print_level

    ! Local variables
    integer :: i, j, k, m_idx, stencil_idx_start
    integer :: len_x, len_y, len_z, expected_nval
    real(amrex_real) :: diag_val, coeff_x, coeff_y, coeff_z
    real(amrex_real) :: domain_extent, factor
    logical :: on_dirichlet_boundary
    logical :: has_inactive_neighbor
    
    ! (Other debug variables can remain)
    ...

    ! Pre-calculate stencil coefficients
    coeff_x = dxinv(1)
    coeff_y = dxinv(2)
    coeff_z = dxinv(3)
    
    ...

    ! Loop over the valid box defined by bxlo/bxhi
    do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          ! Calculate indices
          m_idx = (i - bxlo(1)) + (j - bxlo(2)) * len_x + (k - bxlo(3)) * len_x * len_y + 1
          stencil_idx_start = nstencil * (m_idx - 1)
          
          has_inactive_neighbor = .false.

          ! --- Check if the cell is part of the simulation ---
          ! It must both be in the correct phase AND part of the percolating mask.
          ! The mask check should be sufficient if generated correctly, but checking
          ! the phase here makes the kernel more robust.
          if ( p(i,j,k,comp_phase) /= phase .or. active_mask(i,j,k) == cell_inactive ) then
              ! --- Apply Explicit Decoupling for INACTIVE cells ---
              a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
              a(stencil_idx_start + istn_c) = 1.0_amrex_real
              rhs(m_idx)  = 0.0_amrex_real
              xinit(m_idx) = 0.0_amrex_real
              cycle ! Skip to next (i,j,k)
          end if

          ! --- If we reach here, cell (i,j,k) is ACTIVE and in the correct PHASE ---
          diag_val = 0.0_amrex_real
          a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real

          ! --- Assemble stencil based on ACTIVE neighbors in the same PHASE ---
          ! -X face
          if ( p(i-1,j,k,comp_phase) == phase .and. active_mask(i-1,j,k) == cell_active ) then
             a(stencil_idx_start + istn_mx) = -coeff_x
             diag_val = diag_val + coeff_x
          else
             has_inactive_neighbor = .true.
          end if
          ! +X face
          if ( p(i+1,j,k,comp_phase) == phase .and. active_mask(i+1,j,k) == cell_active ) then
             a(stencil_idx_start + istn_px) = -coeff_x
             diag_val = diag_val + coeff_x
          else
             has_inactive_neighbor = .true.
          end if
          ! -Y face
          if ( p(i,j-1,k,comp_phase) == phase .and. active_mask(i,j-1,k) == cell_active ) then
             a(stencil_idx_start + istn_my) = -coeff_y
             diag_val = diag_val + coeff_y
          else
             has_inactive_neighbor = .true.
          end if
          ! +Y face
          if ( p(i,j+1,k,comp_phase) == phase .and. active_mask(i,j+1,k) == cell_active ) then
             a(stencil_idx_start + istn_py) = -coeff_y
             diag_val = diag_val + coeff_y
          else
             has_inactive_neighbor = .true.
          end if
          ! -Z face
          if ( p(i,j,k-1,comp_phase) == phase .and. active_mask(i,j,k-1) == cell_active ) then
             a(stencil_idx_start + istn_mz) = -coeff_z
             diag_val = diag_val + coeff_z
          else
             has_inactive_neighbor = .true.
          end if
           ! +Z face
          if ( p(i,j,k+1,comp_phase) == phase .and. active_mask(i,j,k+1) == cell_active ) then
             a(stencil_idx_start + istn_pz) = -coeff_z
             diag_val = diag_val + coeff_z
          else
             has_inactive_neighbor = .true.
          end if

          ! (The rest of the subroutine for setting the diagonal, handling Dirichlet BCs,
          !  and calculating xinit remains the same as your original)
          ...

        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_fillmtx

end module tortuosity_poisson_3d_module
