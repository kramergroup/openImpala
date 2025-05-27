module effdiff_fillmtx_module

  use amrex_fort_module, only : amrex_real, amrex_spacedim
  use amrex_error_module, only : amrex_fi_abort
  implicit none

  private ! Default module visibility to private
  public :: effdiff_fillmtx ! Make only the subroutine public

  ! --- Constants ---
  integer, parameter :: DIR_X = 0
  integer, parameter :: DIR_Y = 1
  integer, parameter :: DIR_Z = 2

  ! active_mask values (consistent with C++)
  integer, parameter :: CELL_INACTIVE = 0 ! Solid, D=0
  integer, parameter :: CELL_ACTIVE   = 1 ! Pore, D=D_material (assumed D_material=1 for simplicity here)

  ! Stencil entry indices (0-based, matching C++ HYPRE convention passed to HYPRE_StructMatrixSetBoxValues)
  integer, parameter :: STN_C  = 0 ! Center
  integer, parameter :: STN_MX = 1 ! -X (West)
  integer, parameter :: STN_PX = 2 ! +X (East)
  integer, parameter :: STN_MY = 3 ! -Y (South)
  integer, parameter :: STN_PY = 4 ! +Y (North)
  integer, parameter :: STN_MZ = 5 ! -Z (Bottom)
  integer, parameter :: STN_PZ = 6 ! +Z (Top)
  integer, parameter :: NSTENCIL = 7

  real(amrex_real), parameter :: SMALL_REAL = 1.0e-15_amrex_real
  real(amrex_real), parameter :: ONE = 1.0_amrex_real
  real(amrex_real), parameter :: ZERO = 0.0_amrex_real
  real(amrex_real), parameter :: HALF = 0.5_amrex_real
  real(amrex_real), parameter :: TWO = 2.0_amrex_real

contains

  ! Subroutine to fill HYPRE matrix (A) and RHS (b) for the cell problem:
  ! ∇_ξ ⋅ (D ∇_ξ χ_k) = -∇_ξ ⋅ (D ê_k)
  ! where D = 1 in active phase (pores), D = 0 in inactive phase (solids).
  ! ê_k is the unit vector in direction dir_k.
  subroutine effdiff_fillmtx(a_out, rhs_out, xinit_out, &
                             npts_valid, &
                             active_mask_ptr, mask_lo, mask_hi, &
                             valid_bx_lo, valid_bx_hi, &
                             domain_lo, domain_hi, &
                             cell_sizes_in, & ! dx, dy, dz
                             dir_k_in, &
                             verbose_level_in) bind(c)

    ! --- Argument Declarations ---
    integer, intent(in) :: npts_valid
    real(amrex_real), intent(out) :: a_out(0:npts_valid*NSTENCIL-1)
    real(amrex_real), intent(out) :: rhs_out(npts_valid)
    real(amrex_real), intent(out) :: xinit_out(npts_valid)

    integer, intent(in) :: mask_lo(3), mask_hi(3)
    integer, intent(in) :: active_mask_ptr(mask_lo(1):mask_hi(1), mask_lo(2):mask_hi(2), mask_lo(3):mask_hi(3))

    integer, intent(in) :: valid_bx_lo(3), valid_bx_hi(3)
    integer, intent(in) :: domain_lo(3), domain_hi(3) ! For checking physical domain boundaries if needed (not for periodic)

    real(amrex_real), intent(in) :: cell_sizes_in(3) ! dx, dy, dz
    integer, intent(in) :: dir_k_in            ! 0 for X, 1 for Y, 2 for Z
    integer, intent(in) :: verbose_level_in

    ! --- Local Variables ---
    integer :: i, j, k, m_idx, stencil_idx_start, s_idx
    integer :: len_x_valid, len_y_valid, len_z_valid ! Dimensions of valid_bx
    real(amrex_real) :: dx, dy, dz
    real(amrex_real) :: inv_dx2, inv_dy2, inv_dz2 ! 1/dx^2, etc.
    real(amrex_real) :: inv_2dx, inv_2dy, inv_2dz ! 1/(2*dx), etc.

    real(amrex_real) :: diag_val
    real(amrex_real) :: D_curr, D_im1, D_ip1, D_jm1, D_jp1, D_km1, D_kp1 ! D at cell centers
    real(amrex_real) :: rhs_term_div_De     ! Contribution from -∇⋅(Dê_k)
    real(amrex_real) :: flux_bc_contrib_rhs ! Contribution to RHS from internal Neumann BC

    integer :: current_cell_activity
    integer :: neighbor_activity_mx, neighbor_activity_px
    integer :: neighbor_activity_my, neighbor_activity_py
    integer :: neighbor_activity_mz, neighbor_activity_pz

    ! --- Initialization ---
    if (npts_valid <= 0) return ! Nothing to do for empty box

    dx = cell_sizes_in(1)
    dy = cell_sizes_in(2)
    dz = cell_sizes_in(3)

    if (dx <= SMALL_REAL .or. dy <= SMALL_REAL .or. dz <= SMALL_REAL) then
      call amrex_fi_abort("effdiff_fillmtx: cell_sizes (dx, dy, dz) must be positive.")
    end if

    inv_dx2 = ONE / (dx * dx)
    inv_dy2 = ONE / (dy * dy)
    inv_dz2 = ONE / (dz * dz)
    inv_2dx = ONE / (TWO * dx)
    inv_2dy = ONE / (TWO * dy)
    inv_2dz = ONE / (TWO * dz)

    len_x_valid = valid_bx_hi(1) - valid_bx_lo(1) + 1
    len_y_valid = valid_bx_hi(2) - valid_bx_lo(2) + 1
    ! len_z_valid = valid_bx_hi(3) - valid_bx_lo(3) + 1 ! Not strictly needed for indexing m_idx

    m_idx = 0 ! Fortran linear index for output arrays (1 to npts_valid)

    ! Loop over the cells in valid_bx (the region this MPI rank is responsible for)
    do k = valid_bx_lo(3), valid_bx_hi(3)
      do j = valid_bx_lo(2), valid_bx_hi(2)
        do i = valid_bx_lo(1), valid_bx_hi(1)
          m_idx = m_idx + 1
          stencil_idx_start = NSTENCIL * (m_idx - 1) ! 0-based for a_out

          current_cell_activity = active_mask_ptr(i, j, k)

          ! Initialize stencil row to zero for safety
          a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
          rhs_out(m_idx)  = ZERO
          xinit_out(m_idx) = ZERO ! Initial guess for chi_k

          if (current_cell_activity == CELL_INACTIVE) then
            ! --- Solid cell (D=0): Decouple the equation ---
            a_out(stencil_idx_start + STN_C) = ONE
            ! rhs_out and xinit_out already 0
            cycle ! Next cell
          end if

          ! --- If we reach here, cell (i,j,k) is ACTIVE (Pore, D=1) ---
          diag_val = ZERO
          rhs_term_div_De = ZERO
          flux_bc_contrib_rhs = ZERO

          ! Get D at current and neighbor cell centers (D=1 if active, 0 if inactive)
          ! Assumes active_mask_ptr has filled ghost cells for neighbors.
          D_curr = ONE ! Since current_cell_activity is CELL_ACTIVE
          D_im1 = real(active_mask_ptr(i-1, j, k), kind=amrex_real)
          D_ip1 = real(active_mask_ptr(i+1, j, k), kind=amrex_real)
          D_jm1 = real(active_mask_ptr(i, j-1, k), kind=amrex_real)
          D_jp1 = real(active_mask_ptr(i, j+1, k), kind=amrex_real)
          D_km1 = real(active_mask_ptr(i, j, k-1), kind=amrex_real)
          D_kp1 = real(active_mask_ptr(i, j, k+1), kind=amrex_real)

          ! Get neighbor activity for clarity in BC logic
          neighbor_activity_mx = active_mask_ptr(i-1, j, k)
          neighbor_activity_px = active_mask_ptr(i+1, j, k)
          neighbor_activity_my = active_mask_ptr(i, j-1, k)
          neighbor_activity_py = active_mask_ptr(i, j+1, k)
          neighbor_activity_mz = active_mask_ptr(i, j, k-1)
          neighbor_activity_pz = active_mask_ptr(i, j, k+1)

          ! === LHS: ∇_ξ ⋅ (D ∇_ξ χ_k) where D=1 in pores ===
          ! Standard 7-point Laplacian for χ_k, modified by internal Neumann BCs.

          ! -X face (West)
          if (neighbor_activity_mx == CELL_ACTIVE) then
            a_out(stencil_idx_start + STN_MX) = -inv_dx2
            diag_val = diag_val + inv_dx2
          else ! Interface with solid: n̂=(-1,0,0). BC: -dχ/dx = -(-1)*(ê_k)_x => dχ/dx = (ê_k)_x
            diag_val = diag_val + inv_dx2 ! Contribution from (χ_i - χ_{i-1,ghost})/dx^2 where χ_{i-1,ghost} = χ_i - dx*(ê_k)_x
                                          ! So -1/dx^2 * (χ_i - dx*(ê_k)_x) contributes +1/dx to RHS
            if (dir_k_in == DIR_X) then   ! (ê_k)_x = 1
              flux_bc_contrib_rhs = flux_bc_contrib_rhs + (ONE/dx)
            end if
            ! No (ê_k)_y or (ê_k)_z component for this face's normal in x-dir
          end if

          ! +X face (East)
          if (neighbor_activity_px == CELL_ACTIVE) then
            a_out(stencil_idx_start + STN_PX) = -inv_dx2
            diag_val = diag_val + inv_dx2
          else ! Interface with solid: n̂=(1,0,0). BC: dχ/dx = -(1)*(ê_k)_x => dχ/dx = -(ê_k)_x
            diag_val = diag_val + inv_dx2 ! χ_{i+1,ghost} = χ_i + dx*(-(ê_k)_x)
            if (dir_k_in == DIR_X) then   ! (ê_k)_x = 1
              flux_bc_contrib_rhs = flux_bc_contrib_rhs - (ONE/dx) ! term is -D * grad_normal = -1 * (-(ê_k)_x)
            end if
          end if

          ! -Y face (South)
          if (neighbor_activity_my == CELL_ACTIVE) then
            a_out(stencil_idx_start + STN_MY) = -inv_dy2
            diag_val = diag_val + inv_dy2
          else ! Interface with solid: n̂=(0,-1,0). BC: -dχ/dy = -(-1)*(ê_k)_y => dχ/dy = (ê_k)_y
            diag_val = diag_val + inv_dy2
            if (dir_k_in == DIR_Y) then   ! (ê_k)_y = 1
              flux_bc_contrib_rhs = flux_bc_contrib_rhs + (ONE/dy)
            end if
          end if

          ! +Y face (North)
          if (neighbor_activity_py == CELL_ACTIVE) then
            a_out(stencil_idx_start + STN_PY) = -inv_dy2
            diag_val = diag_val + inv_dy2
          else ! Interface with solid: n̂=(0,1,0). BC: dχ/dy = -(1)*(ê_k)_y => dχ/dy = -(ê_k)_y
            diag_val = diag_val + inv_dy2
            if (dir_k_in == DIR_Y) then   ! (ê_k)_y = 1
              flux_bc_contrib_rhs = flux_bc_contrib_rhs - (ONE/dy)
            end if
          end if

          ! -Z face (Bottom)
          if (AMREX_SPACEDIM == 3) then
            if (neighbor_activity_mz == CELL_ACTIVE) then
              a_out(stencil_idx_start + STN_MZ) = -inv_dz2
              diag_val = diag_val + inv_dz2
            else ! Interface with solid: n̂=(0,0,-1). BC: -dχ/dz = -(-1)*(ê_k)_z => dχ/dz = (ê_k)_z
              diag_val = diag_val + inv_dz2
              if (dir_k_in == DIR_Z) then   ! (ê_k)_z = 1
                flux_bc_contrib_rhs = flux_bc_contrib_rhs + (ONE/dz)
              end if
            end if

            ! +Z face (Top)
            if (neighbor_activity_pz == CELL_ACTIVE) then
              a_out(stencil_idx_start + STN_PZ) = -inv_dz2
              diag_val = diag_val + inv_dz2
            else ! Interface with solid: n̂=(0,0,1). BC: dχ/dz = -(1)*(ê_k)_z => dχ/dz = -(ê_k)_z
              diag_val = diag_val + inv_dz2
              if (dir_k_in == DIR_Z) then   ! (ê_k)_z = 1
                flux_bc_contrib_rhs = flux_bc_contrib_rhs - (ONE/dz)
              end if
            end if
          end if ! AMREX_SPACEDIM == 3

          a_out(stencil_idx_start + STN_C) = diag_val

          ! === RHS: -∇_ξ ⋅ (D ê_k) ===
          ! This is -∂(D)/∂x for ê_x, -∂(D)/∂y for ê_y, -∂(D)/∂z for ê_z
          ! Using central difference for ∂(D)/∂x: (D_{i+1} - D_{i-1}) / (2dx)
          ! (where D_i is D at cell i center, which is 0 or 1)
          if (dir_k_in == DIR_X) then
            rhs_term_div_De = -(D_ip1 - D_im1) * inv_2dx
          else if (dir_k_in == DIR_Y) then
            rhs_term_div_De = -(D_jp1 - D_jm1) * inv_2dy
          else if (dir_k_in == DIR_Z .and. AMREX_SPACEDIM == 3) then
            rhs_term_div_De = -(D_kp1 - D_km1) * inv_2dz
          end if

          rhs_out(m_idx) = rhs_term_div_De + flux_bc_contrib_rhs

          ! Check for nearly zero diagonal in an active cell (can happen if isolated pore)
          ! If so, decouple it to prevent solver issues.
          if (abs(diag_val) < SMALL_REAL) then
             if (verbose_level_in > 0) then
                 write(*,'(A,3I5,A,ES12.4)') "effdiff_fillmtx WARNING: Near-zero diagonal in ACTIVE cell (", &
                      i,j,k, "), diag_val=", diag_val, " Decoupling."
             end if
             a_out(stencil_idx_start : stencil_idx_start + NSTENCIL - 1) = ZERO
             a_out(stencil_idx_start + STN_C) = ONE
             rhs_out(m_idx) = ZERO
             xinit_out(m_idx) = ZERO
          end if

          ! --- Debug Printing for this active cell (optional) ---
          if (verbose_level_in >= 3) then
            write(*,'(A,3I5,A,I2)') "DEBUG effdiff_fillmtx: Cell (",i,j,k,") dir_k=", dir_k_in
            write(*,'(A,7ES12.4)') "  Stencil A: ", (a_out(stencil_idx_start+s_idx), s_idx=0,NSTENCIL-1)
            write(*,'(A,ES12.4, A,ES12.4, A,ES12.4)') "  RHS terms: div_De=", rhs_term_div_De, &
                                                   " flux_bc=", flux_bc_contrib_rhs, &
                                                   " Total_RHS=", rhs_out(m_idx)
            write(*,'(A,ES12.4)') "  XINIT: ", xinit_out(m_idx)
          end if

        end do ! i
      end do ! j
    end do ! k

    ! Final check on m_idx
    if (m_idx /= npts_valid) then
      call amrex_fi_abort("effdiff_fillmtx: m_idx /= npts_valid. Indexing error.")
    end if

  end subroutine effdiff_fillmtx

end module effdiff_fillmtx_module
