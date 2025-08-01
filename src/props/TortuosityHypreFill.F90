module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  use amrex_error_module, only : amrex_abort

  implicit none

  private ! Default module visibility to private
  public :: tortuosity_fillmtx_mp

  ! ... (Parameters remain the same) ...
  integer, parameter :: direction_x = 0, direction_y = 1, direction_z = 2
  integer, parameter :: comp_phase = 1, comp_mask = 1
  integer, parameter :: istn_c  = 0, istn_mx = 1, istn_px = 2, istn_my = 3, istn_py = 4, istn_mz = 5, istn_pz = 6
  integer, parameter :: nstencil = 7
  integer, parameter :: cell_inactive = 0, cell_active = 1
  real(amrex_real), parameter :: small_real = 1.0e-15_amrex_real

contains

!-----------------------------------------------------------------------
! NEW multi-phase routine.
! Discretizes Div(D(x) Grad(phi)) = 0 using a harmonic mean for face conductivity.
! This is a direct update of the original tortuosity_fillmtx, preserving its structure.
!-----------------------------------------------------------------------
  subroutine tortuosity_fillmtx_mp(a, rhs, xinit, nval, &
                                 coeff_d, d_lo, d_hi, &
                                 active_mask, mask_lo, mask_hi, &
                                 bxlo, bxhi, domlo, domhi, dx, vlo, vhi, dir, &
                                 verbose) bind(c, name='tortuosity_fillmtx_mp')

    ! Argument declarations (updated for multi-phase)
    integer,          intent(in)  :: nval
    real(amrex_real), intent(out) :: a(0:nval*nstencil-1), rhs(nval), xinit(nval)
    integer,          intent(in)  :: d_lo(3), d_hi(3)
    real(amrex_real), intent(in)  :: coeff_d(d_lo(1):d_hi(1), d_lo(2):d_hi(2), d_lo(3):d_hi(3))
    integer,          intent(in)  :: mask_lo(3), mask_hi(3)
    integer,          intent(in)  :: active_mask(mask_lo(1):mask_hi(1), mask_lo(2):mask_hi(2), mask_lo(3):mask_hi(3))
    integer,          intent(in)  :: bxlo(3), bxhi(3), domlo(3), domhi(3)
    real(amrex_real), intent(in)  :: dx(3)
    real(amrex_real), intent(in)  :: vlo, vhi
    integer,          intent(in)  :: dir, verbose

    ! Local variables (preserved from original)
    integer :: i, j, k, m_idx, stencil_idx_start, s_idx
    integer :: len_x, len_y, len_z, expected_nval
    real(amrex_real) :: diag_val
    real(amrex_real) :: domain_extent, factor
    logical :: on_dirichlet_boundary
    integer :: neighbor_mask_val
    ! Debugging variables (preserved from original)
    logical :: print_this_cell, near_boundary, has_inactive_neighbor
    real(amrex_real) :: off_diag_sum, diag_ratio
    ! NEW variables for multi-phase
    real(amrex_real) :: d_center, d_neighbor, d_face, off_diag_val, dxinv_sq(3)

    ! Calculate box dimensions based on bxlo/bxhi (the valid box)
    len_x = bxhi(1) - bxlo(1) + 1
    len_y = bxhi(2) - bxlo(2) + 1
    len_z = bxhi(3) - bxlo(3) + 1
    expected_nval = len_x * len_y * len_z

    ! CHANGE: Calculate inverse grid spacing squared from dx
    dxinv_sq(1) = 1.0_amrex_real / (dx(1) * dx(1))
    dxinv_sq(2) = 1.0_amrex_real / (dx(2) * dx(2))
    dxinv_sq(3) = 1.0_amrex_real / (dx(3) * dx(3))

    if (expected_nval > 0 .and. expected_nval /= nval) then
       call amrex_abort("tortuosity_fillmtx_mp: nval mismatch.")
    end if
    if (nval <= 0) return

    ! Internal helper function for harmonic mean
    contains
      function get_face_coeff(d1, d2) result(d_face_res)
        real(amrex_real), intent(in) :: d1, d2
        real(amrex_real) :: d_face_res
        if (d1 + d2 > small_real) then
          d_face_res = 2.0_amrex_real * d1 * d2 / (d1 + d2)
        else
          d_face_res = 0.0_amrex_real
        end if
      end function get_face_coeff

    ! Loop over the valid box defined by bxlo/bxhi
    do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          m_idx = (i - bxlo(1)) + (j - bxlo(2)) * len_x + (k - bxlo(3)) * len_x * len_y + 1
          stencil_idx_start = nstencil * (m_idx - 1)
          print_this_cell = .false.
          near_boundary = .false.
          has_inactive_neighbor = .false.

          if ( active_mask(i,j,k) == cell_inactive ) then
              a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
              a(stencil_idx_start + istn_c) = 1.0_amrex_real
              rhs(m_idx)   = 0.0_amrex_real
              xinit(m_idx) = 0.0_amrex_real
              cycle
          end if

          diag_val = 0.0_amrex_real
          a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
          d_center = coeff_d(i, j, k)

          ! -X face
          neighbor_mask_val = active_mask(i-1, j, k)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i-1, j, k)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(1)
              a(stencil_idx_start + istn_mx) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if
          ! +X face
          neighbor_mask_val = active_mask(i+1, j, k)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i+1, j, k)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(1)
              a(stencil_idx_start + istn_px) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if
          ! -Y face
          neighbor_mask_val = active_mask(i, j-1, k)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i, j-1, k)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(2)
              a(stencil_idx_start + istn_my) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if
          ! +Y face
          neighbor_mask_val = active_mask(i, j+1, k)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i, j+1, k)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(2)
              a(stencil_idx_start + istn_py) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if
          ! -Z face
          neighbor_mask_val = active_mask(i, j, k-1)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i, j, k-1)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(3)
              a(stencil_idx_start + istn_mz) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if
          ! +Z face
          neighbor_mask_val = active_mask(i, j, k+1)
          if ( neighbor_mask_val == cell_active ) then
              d_neighbor = coeff_d(i, j, k+1)
              d_face = get_face_coeff(d_center, d_neighbor)
              off_diag_val = -d_face * dxinv_sq(3)
              a(stencil_idx_start + istn_pz) = off_diag_val
              diag_val = diag_val - off_diag_val
          else
              has_inactive_neighbor = .true.
          end if

          a(stencil_idx_start + istn_c) = diag_val
          rhs(m_idx) = 0.0_amrex_real

          if ( abs(diag_val) < small_real ) then
              write(*,'(A,3I5)') "WARNING: Zero diagonal in ACTIVE cell at (i,j,k)=", i,j,k
              a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
              a(stencil_idx_start + istn_c) = 1.0_amrex_real
              rhs(m_idx)   = 0.0_amrex_real
              xinit(m_idx) = 0.0_amrex_real
              cycle
          end if

          ! --- Overwrite stencil for Domain Boundaries (Unchanged) ---
          on_dirichlet_boundary = .false.
          if ( dir == direction_x ) then
              if ( i == domlo(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( i == domhi(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          else if ( dir == direction_y ) then
              if ( j == domlo(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( j == domhi(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          else if ( dir == direction_z ) then
              if ( k == domlo(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
                  on_dirichlet_boundary = .true.
              else if ( k == domhi(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
                  on_dirichlet_boundary = .true.
              end if
          end if

          ! --- Calculate Initial Guess (Unchanged) ---
          if ( abs(a(stencil_idx_start + istn_c) - 1.0_amrex_real) > small_real .or. on_dirichlet_boundary ) then
              if ( dir == direction_x ) then
                  domain_extent = domhi(1) - domlo(1)
                  if (abs(domain_extent) < small_real) then; factor = 0.0_amrex_real; else; factor = 1.0_amrex_real / domain_extent; end if
                  xinit(m_idx) = vlo + (vhi - vlo) * (i - domlo(1)) * factor
              else if ( dir == direction_y ) then
                  domain_extent = domhi(2) - domlo(2)
                  if (abs(domain_extent) < small_real) then; factor = 0.0_amrex_real; else; factor = 1.0_amrex_real / domain_extent; end if
                  xinit(m_idx) = vlo + (vhi - vlo) * (j - domlo(2)) * factor
              else if ( dir == direction_z ) then
                  domain_extent = domhi(3) - domlo(3)
                  if (abs(domain_extent) < small_real) then; factor = 0.0_amrex_real; else; factor = 1.0_amrex_real / domain_extent; end if
                  xinit(m_idx) = vlo + (vhi - vlo) * (k - domlo(3)) * factor
              end if
          end if

          ! --- Debug Printing Section (Unchanged, uses 'verbose' now) ---
          if (verbose >= 3) then
              near_boundary = (i <= domlo(1)+1 .or. i >= domhi(1)-1 .or. &
                               j <= domlo(2)+1 .or. j >= domhi(2)-1 .or. &
                               k <= domlo(3)+1 .or. k >= domhi(3)-1)
              print_this_cell = near_boundary .or. has_inactive_neighbor
              if (print_this_cell) then
                  off_diag_sum = 0.0_amrex_real
                  do s_idx = 1, nstencil-1
                      off_diag_sum = off_diag_sum + abs(a(stencil_idx_start + s_idx))
                  end do
                  diag_val = a(stencil_idx_start + istn_c)
                  if (abs(off_diag_sum) < small_real) then
                    if (abs(diag_val) < small_real) then; diag_ratio = 1.0_amrex_real; else; diag_ratio = 1.0e+30_amrex_real; end if
                  else
                      diag_ratio = abs(diag_val) / off_diag_sum
                  end if
                  write(*,'(A,3I5,A,L1,A,L1,A,L1)') "DEBUG Stencil at (", i, j, k, ")", &
                       " Active=", .true., " Dirichlet=", on_dirichlet_boundary, " Interface=", has_inactive_neighbor
                  write(*,'(A,ES12.4)') "  RHS =", rhs(m_idx)
                  write(*,'(A,7(ES12.4,1X))') "  Stencil (C,-X,+X,-Y,+Y,-Z,+Z) =" , (a(stencil_idx_start + s_idx), s_idx=0, nstencil-1)
                  write(*,'(A,ES12.4, A,ES12.4)') "  Diag Dominance Ratio (|Aii|/Sum|Aij|) =", diag_ratio, &
                       " (OffDiagSum =", off_diag_sum, ")"
              end if
          end if

        end do
      end do
    end do

  end subroutine tortuosity_fillmtx_mp

end module tortuosity_poisson_3d_module
