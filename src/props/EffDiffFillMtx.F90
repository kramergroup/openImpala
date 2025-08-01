module effdiff_fillmtx_module

  use amrex_fort_module, only : amrex_real, amrex_spacedim
  use amrex_error_module, only : amrex_abort
  implicit none

  private ! Default module visibility to private
  public :: effdiff_fillmtx_mp ! Make only the new multi-phase subroutine public

  ! --- Constants ---
  integer, parameter :: DIR_X = 0, DIR_Y = 1, DIR_Z = 2
  integer, parameter :: CELL_INACTIVE = 0, CELL_ACTIVE = 1
  integer, parameter :: STN_C  = 0, STN_MX = 1, STN_PX = 2, STN_MY = 3, STN_PY = 4, STN_MZ = 5, STN_PZ = 6
  integer, parameter :: NSTENCIL = 7
  real(amrex_real), parameter :: SMALL_REAL = 1.0e-15_amrex_real
  real(amrex_real), parameter :: ONE = 1.0_amrex_real, ZERO = 0.0_amrex_real, TWO = 2.0_amrex_real

contains

!-----------------------------------------------------------------------
! Fills HYPRE matrix (A) and RHS (b) for the multi-phase cell problem:
! ∇_ξ ⋅ (D(x) ∇_ξ χ_k) = -∇_ξ ⋅ (D(x) ê_k)
! D(x) is the spatially varying coefficient field.
! ê_k is the unit vector in direction dir_k.
!-----------------------------------------------------------------------
  subroutine effdiff_fillmtx_mp(a_out, rhs_out, xinit_out, &
                                npts_valid, &
                                coeff_d, d_lo, d_hi, &
                                active_mask, mask_lo, mask_hi, &
                                valid_bx_lo, valid_bx_hi, &
                                domain_lo, domain_hi, &
                                cell_sizes_in, &
                                dir_k_in, &
                                verbose_level_in) bind(c, name='effdiff_fillmtx_mp')

    ! --- Argument Declarations ---
    integer, intent(in) :: npts_valid
    real(amrex_real), intent(out) :: a_out(0:npts_valid*NSTENCIL-1)
    real(amrex_real), intent(out) :: rhs_out(npts_valid)
    real(amrex_real), intent(out) :: xinit_out(npts_valid)

    integer, intent(in) :: d_lo(3), d_hi(3)
    real(amrex_real), intent(in) :: coeff_d(d_lo(1):d_hi(1), d_lo(2):d_hi(2), d_lo(3):d_hi(3))
    integer, intent(in) :: mask_lo(3), mask_hi(3)
    integer, intent(in) :: active_mask(mask_lo(1):mask_hi(1), mask_lo(2):mask_hi(2), mask_lo(3):mask_hi(3))

    integer, intent(in) :: valid_bx_lo(3), valid_bx_hi(3), domain_lo(3), domain_hi(3)
    real(amrex_real), intent(in) :: cell_sizes_in(3)
    integer, intent(in) :: dir_k_in, verbose_level_in

    ! --- Local Variables ---
    integer :: i, j, k, m_idx, stencil_idx_start
    real(amrex_real) :: dx, dy, dz
    real(amrex_real) :: inv_dx2, inv_dy2, inv_dz2
    real(amrex_real) :: inv_2dx, inv_2dy, inv_2dz
    real(amrex_real) :: diag_val, off_diag_val
    real(amrex_real) :: D_curr, D_im1, D_ip1, D_jm1, D_jp1, D_km1, D_kp1
    real(amrex_real) :: D_face
    real(amrex_real) :: rhs_term_div_De

    if (npts_valid <= 0) return

    dx = cell_sizes_in(1); dy = cell_sizes_in(2); dz = cell_sizes_in(3)
    inv_dx2 = ONE / (dx*dx); inv_dy2 = ONE / (dy*dy); inv_dz2 = ONE / (dz*dz)
    inv_2dx = ONE / (TWO*dx); inv_2dy = ONE / (TWO*dy); inv_2dz = ONE / (TWO*dz)

    m_idx = 0

    ! Internal helper function for harmonic mean
    contains
      function get_face_coeff(d1, d2) result(d_face_res)
        real(amrex_real), intent(in) :: d1, d2
        real(amrex_real) :: d_face_res
        if (d1 + d2 > SMALL_REAL) then
          d_face_res = TWO * d1 * d2 / (d1 + d2)
        else
          d_face_res = ZERO
        end if
      end function get_face_coeff

    ! Loop over the cells in the valid box
    do k = valid_bx_lo(3), valid_bx_hi(3)
      do j = valid_bx_lo(2), valid_bx_hi(2)
        do i = valid_bx_lo(1), valid_bx_hi(1)
          m_idx = m_idx + 1
          stencil_idx_start = NSTENCIL * (m_idx - 1)

          ! Initialize
          a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
          rhs_out(m_idx) = ZERO
          xinit_out(m_idx) = ZERO

          if (active_mask(i, j, k) == CELL_INACTIVE) then
              a_out(stencil_idx_start + STN_C) = ONE
              cycle
          end if

          ! --- Cell is ACTIVE ---
          diag_val = ZERO
          D_curr = coeff_d(i, j, k)
          D_im1 = coeff_d(i-1, j, k); D_ip1 = coeff_d(i+1, j, k)
          D_jm1 = coeff_d(i, j-1, k); D_jp1 = coeff_d(i, j+1, k)
          D_km1 = coeff_d(i, j, k-1); D_kp1 = coeff_d(i, j, k+1)

          ! === LHS: ∇ ⋅ (D ∇χ_k) using harmonic mean for face D ===
          ! -X face
          D_face = get_face_coeff(D_curr, D_im1)
          off_diag_val = -D_face * inv_dx2
          a_out(stencil_idx_start + STN_MX) = off_diag_val
          diag_val = diag_val - off_diag_val
          ! +X face
          D_face = get_face_coeff(D_curr, D_ip1)
          off_diag_val = -D_face * inv_dx2
          a_out(stencil_idx_start + STN_PX) = off_diag_val
          diag_val = diag_val - off_diag_val
          ! -Y face
          D_face = get_face_coeff(D_curr, D_jm1)
          off_diag_val = -D_face * inv_dy2
          a_out(stencil_idx_start + STN_MY) = off_diag_val
          diag_val = diag_val - off_diag_val
          ! +Y face
          D_face = get_face_coeff(D_curr, D_jp1)
          off_diag_val = -D_face * inv_dy2
          a_out(stencil_idx_start + STN_PY) = off_diag_val
          diag_val = diag_val - off_diag_val
          ! -Z face
          if (AMREX_SPACEDIM == 3) then
            D_face = get_face_coeff(D_curr, D_km1)
            off_diag_val = -D_face * inv_dz2
            a_out(stencil_idx_start + STN_MZ) = off_diag_val
            diag_val = diag_val - off_diag_val
            ! +Z face
            D_face = get_face_coeff(D_curr, D_kp1)
            off_diag_val = -D_face * inv_dz2
            a_out(stencil_idx_start + STN_PZ) = off_diag_val
            diag_val = diag_val - off_diag_val
          end if
          a_out(stencil_idx_start + STN_C) = diag_val

          ! === RHS: -∇ ⋅ (D ê_k) using central difference ===
          rhs_term_div_De = ZERO
          if (dir_k_in == DIR_X) then
            rhs_term_div_De = -(D_ip1 - D_im1) * inv_2dx
          else if (dir_k_in == DIR_Y) then
            rhs_term_div_De = -(D_jp1 - D_jm1) * inv_2dy
          else if (dir_k_in == DIR_Z .and. AMREX_SPACEDIM == 3) then
            rhs_term_div_De = -(D_kp1 - D_km1) * inv_2dz
          end if
          rhs_out(m_idx) = rhs_term_div_De

          ! Safety check for isolated active cells
          if (abs(diag_val) < SMALL_REAL) then
              a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
              a_out(stencil_idx_start + STN_C) = ONE
              rhs_out(m_idx) = ZERO
          end if

        end do
      end do
    end do

  end subroutine effdiff_fillmtx_mp

end module effdiff_fillmtx_module
