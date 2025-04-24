module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  use amrex_error_module, only : amrex_abort

  implicit none

  private ! Default module visibility to private

  public :: tortuosity_fillmtx ! Make only the subroutine public

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  ! Component index for phase data in MultiFab 'p' (Fortran 1-based)
  integer, parameter :: comp_phase = 1

  ! Stencil indices (0-based, matching C++ HYPRE_StructStencilSetElement order)
  ! Assuming C++ order: {Center, -x, +x, -y, +y, -z, +z} maps to indices 0..6
  integer, parameter :: istn_c  = 0 ! Center
  integer, parameter :: istn_mx = 1 ! -X (West)
  integer, parameter :: istn_px = 2 ! +X (East)
  integer, parameter :: istn_my = 3 ! -Y (South)
  integer, parameter :: istn_py = 4 ! +Y (North)
  integer, parameter :: istn_mz = 5 ! -Z (Bottom)
  integer, parameter :: istn_pz = 6 ! +Z (Top)
  integer, parameter :: nstencil = 7

contains

  ! ::: -----------------------------------------------------------
  ! ::: Fills HYPRE matrix coefficients for a Poisson equation within a box.
  ! ::: Uses penalty method for blocked cells to improve conditioning.
  ! ::: Includes debug prints to check array indices and diagonal entries.
  ! ::: -----------------------------------------------------------
  subroutine tortuosity_fillmtx(a, rhs, xinit, nval, p, p_lo, p_hi, &
                                bxlo, bxhi, domlo, domhi, dxinv, vlo, vhi, phase, dir) bind(c)

    ! Argument declarations
    integer,            intent(in)  :: nval
    ! Fortran declaration uses 0-based indexing matching C++ std::vector
    ! a: matrix coefficients (size nval * nstencil)
    ! rhs: right-hand side vector (size nval)
    ! xinit: initial guess vector (size nval)
    real(amrex_real), intent(out) :: a(0:nval*nstencil-1), rhs(nval), xinit(nval)
    integer,            intent(in)  :: p_lo(3), p_hi(3) ! Bounds of the phase FAB (incl. ghost cells)
    integer,            intent(in)  :: bxlo(3), bxhi(3) ! Bounds of the current valid box (tilebox)
    integer,            intent(in)  :: domlo(3), domhi(3) ! Bounds of the overall problem domain
    ! p: phase data FAB (indexed using p_lo/p_hi)
    integer,            intent(in)  :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2), p_lo(3):p_hi(3), *)
    real(amrex_real), intent(in)  :: dxinv(3) ! [1/dx^2, 1/dy^2, 1/dz^2]
    integer,            intent(in)  :: phase, dir ! Phase ID to treat as conductive, direction of flow
    real(amrex_real), intent(in)  :: vlo, vhi ! Potential values at low/high boundaries

    ! Local variables
    integer :: i, j, k, m_idx, stencil_idx_start
    integer :: len_x, len_y, len_z, expected_nval
    real(amrex_real) :: coeff_c, coeff_x, coeff_y, coeff_z
    real(amrex_real) :: domain_extent, factor
    logical :: print_debug_info
    ! Penalty factor for blocked cells
    real(amrex_real), parameter :: penalty_factor = 1.0e20_amrex_real

    ! Calculate box dimensions based on bxlo/bxhi (the valid box)
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
       call amrex_abort("tortuosity_fillmtx: nval mismatch.")
    end if
    if (nval <= 0) return ! Nothing to do for empty box

    ! Loop over the valid box defined by bxlo/bxhi
    do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          ! Calculate indices
          m_idx = (i - bxlo(1)) + (j - bxlo(2)) * len_x + (k - bxlo(3)) * len_x * len_y + 1
          stencil_idx_start = nstencil * (m_idx - 1)

          ! --- DEBUG PRINT: Check indices --- (Optional)
          ! if (m_idx == 1 .or. m_idx == nval) then ... end if

          ! --- Set Base Stencil (Laplacian) for ALL points initially ---
          a(stencil_idx_start + istn_c)  =  coeff_c
          a(stencil_idx_start + istn_mx) = -coeff_x
          a(stencil_idx_start + istn_px) = -coeff_x
          a(stencil_idx_start + istn_my) = -coeff_y
          a(stencil_idx_start + istn_py) = -coeff_y
          a(stencil_idx_start + istn_mz) = -coeff_z
          a(stencil_idx_start + istn_pz) = -coeff_z
          rhs(m_idx) = 0.0_amrex_real ! Default RHS

          ! --- Apply Modifications Based on Phase ---
          if ( p(i,j,k,comp_phase) == phase ) then
              ! Fluid cell (conductive phase) - Apply Neumann at phase boundaries

              ! -X face
              if ( p(i-1,j,k,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_x ! Absorb flux
                  a(stencil_idx_start + istn_mx) = 0.0_amrex_real                        ! Zero connection
              end if
              ! +X face
              if ( p(i+1,j,k,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_x
                  a(stencil_idx_start + istn_px) = 0.0_amrex_real
              end if
              ! -Y face
              if ( p(i,j-1,k,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_y
                  a(stencil_idx_start + istn_my) = 0.0_amrex_real
              end if
              ! +Y face
              if ( p(i,j+1,k,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_y
                  a(stencil_idx_start + istn_py) = 0.0_amrex_real
              end if
              ! -Z face
              if ( p(i,j,k-1,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_z
                  a(stencil_idx_start + istn_mz) = 0.0_amrex_real
              end if
               ! +Z face
              if ( p(i,j,k+1,comp_phase) .ne. phase ) then
                  a(stencil_idx_start + istn_c)  = a(stencil_idx_start + istn_c)  - coeff_z
                  a(stencil_idx_start + istn_pz) = 0.0_amrex_real
              end if
              ! RHS remains 0.0 for internal fluid cells

          else
              ! Blocked cell (not the specified conductive phase)
              ! --- Apply Penalty Method ---
              ! Keep the base Laplacian stencil (already set)
              ! Add large penalty factor to the diagonal coefficient
              a(stencil_idx_start + istn_c) = a(stencil_idx_start + istn_c) + penalty_factor
              ! RHS becomes penalty * desired_value (which is 0 here)
              rhs(m_idx) = penalty_factor * 0.0_amrex_real ! Still zero

          end if ! End of fluid/blocked check


          ! --- Overwrite stencil for Domain Boundaries Perpendicular to Flow (Dirichlet) ---
          ! This applies to BOTH fluid and blocked cells if they fall on the boundary.
          ! X-Direction Flow
          if ( dir == direction_x ) then
              if ( i == domlo(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
              else if ( i == domhi(1) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
              end if
          ! Y-Direction Flow
          else if ( dir == direction_y ) then
              if ( j == domlo(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
              else if ( j == domhi(2) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
              end if
          ! Z-Direction Flow
          else if ( dir == direction_z ) then
              if ( k == domlo(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vlo
              else if ( k == domhi(3) ) then
                  a(stencil_idx_start : stencil_idx_start + nstencil - 1) = 0.0_amrex_real
                  a(stencil_idx_start + istn_c) = 1.0_amrex_real
                  rhs(m_idx) = vhi
              end if
          end if ! End of Dirichlet BC overwrite

          ! --- Calculate Initial Guess ---
          ! (This logic remains the same)
          if ( dir == direction_x ) then
              domain_extent = domhi(1) - domlo(1)
              if (abs(domain_extent) < 1.0e-12_amrex_real) then
                  factor = 0.0_amrex_real
              else
                  factor = 1.0_amrex_real / domain_extent
              end if
              xinit(m_idx) = vlo + (vhi - vlo) * (i - domlo(1)) * factor
          else if ( dir == direction_y ) then
              domain_extent = domhi(2) - domlo(2)
              if (abs(domain_extent) < 1.0e-12_amrex_real) then
                  factor = 0.0_amrex_real
              else
                  factor = 1.0_amrex_real / domain_extent
              end if
              xinit(m_idx) = vlo + (vhi - vlo) * (j - domlo(2)) * factor
          else if ( dir == direction_z ) then
              domain_extent = domhi(3) - domlo(3)
              if (abs(domain_extent) < 1.0e-12_amrex_real) then
                  factor = 0.0_amrex_real
              else
                  factor = 1.0_amrex_real / domain_extent
              end if
              xinit(m_idx) = vlo + (vhi - vlo) * (k - domlo(3)) * factor
          else ! Should not happen if dir is 0, 1, or 2
              xinit(m_idx) = 0.5_amrex_real * (vlo + vhi)
          end if

          ! --- Debug check for near-zero diagonals ---
          ! (Still useful to keep)
          if ( p(i,j,k,comp_phase) == phase ) then
             if ( abs(a(stencil_idx_start + istn_c)) < 1.0e-15_amrex_real ) then
                 write(*,'(A,3I5,A,ES10.3)') &
                      "WARNING: Near-zero diagonal at (i,j,k)=", i,j,k, &
                      " value=", a(stencil_idx_start + istn_c)
             end if
          end if

        end do ! i
      end do ! j
    end do ! k

  end subroutine tortuosity_fillmtx

end module tortuosity_poisson_3d_module
