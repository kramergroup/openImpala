module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  
  implicit none

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

contains

  ! ::: -----------------------------------------------------------
  ! ::: This routine fills the hypre matrix coefficients for the 
  ! ::: interior of the domain.
  ! ::: 
  ! ::: INPUTS/OUTPUTS:
  ! ::: a           <=  array to fill for matrix coefficients
  ! ::: rhs         <=  array to fill for rhs
  ! ::: nval         => number of poins (size of rhs / size of a is nval*nstencil (7))
  ! ::: p            => array with phase indices
  ! ::: bxlo, bxhi   => dimensions of valid box 
  ! ::: domlo,domhi  => index extent of problem domain
  ! ::: vlo,vhi      => boundary values at lo/hi end of flow
  ! ::: phase        => index of the active phase
  ! ::: dir          => direction of flow ()
  ! :::
  ! ::: ALGORITHM:
  ! ::: Go over domain and adjust stencil for blocked and mixed cells.
  ! :::
  ! :::    Fluid cells:
  ! :::      fill all points with a normal seven point poisson stencil
  ! :::
  ! :::      0 = x[i-1] - 2 x[i] + x[i+1]
  ! :::        + y[i-1] - 2 y[i] + y[i+1]
  ! :::        + z[i-1] - 2 z[i] + z[i-1]
  ! :::
  ! :::                 c  -x  +x  -y  +y  -z  +z
  ! :::      stencil:   6  -1  -1  -1  -1  -1  -1
  ! :::
  ! :::      We have used x,y,z to designate the direction. They all refer to the same field:
  ! ::: 
  ! :::        phi[i] = x[i] = y[i] = z[i]
  ! :::
  ! :::    Blocked cells:
  ! :::      simply reduce the stencil to a simple phi(i) = 0 condition.
  ! :::
  ! :::                 c  -x  +x  -y  +y  -z  +z
  ! :::      stencil:   1   0   0   0   0   0   0
  ! :::
  ! :::    Mixed cells:
  ! :::      We do this by sustracting from the default stencil when we have a
  ! :::      cell that is touching a boundary cell. Example: 
  ! :::
  ! :::      a boundary in the x direction at the low side:
  ! :::      
  ! :::                 c  -x  +x  -y  +y  -z  +z
  ! :::      default:   6  -1  -1  -1  -1  -1  -1
  ! :::      substact:  1  -1 
  ! :::                 -------------------------
  ! :::      result     5   0  -1  -1  -1  -1  -1
  ! :::
  ! :::      This changes the x-direction equation from
  ! :::        
  ! :::         0 = x[i+1] - 2 x[i] + x[i-1]
  ! ::: 
  ! :::      to
  ! ::: 
  ! :::         0 = x[i+1] - x[i]
  ! :::
  ! :::      which is a normal upwind gradient for zero flux
  ! :::
  ! :::    Domain boundary cells:
  ! :::      At the inlet and outlet in flow direction, von Neumann conditions are used
  ! :::      to fix the concentration. 
  ! :::
  ! :::         v_lo = x[0]     v_hi = x[max]
  ! :::
  ! :::      The kernel and right-hand side are set accordingly
  ! :::        
  ! :::                 c  -x  +x  -y  +y  -z  +z
  ! :::      neumann:   1   0   0   0   0   0   0
  ! :::
  ! :::      Domain boundaries parallel to the flow direction have Dirichlet boundary 
  ! :::      conditions and are configured like internal Dirichlet conditions (see above).
  ! ::: -----------------------------------------------------------

  subroutine tortuosity_fillmtx(a, rhs, xinit, nval, p, p_lo, p_hi, &
                                bxlo, bxhi, domlo, domhi, vlo, vhi, phase, dir) bind(c)

    integer,          intent(in   ) :: nval
    real(amrex_real), intent(inout) :: a(nval*7), rhs(nval), xinit(nval)
    integer,          intent(in   ) :: p_lo(3), p_hi(3), bxlo(3), bxhi(3), domlo(3), domhi(3)
    integer,          intent(in   ) :: p(p_lo(1):p_hi(1), p_lo(2):p_hi(2),p_lo(3):p_hi(3))
    integer,          intent(in   ) :: phase, dir
    real(amrex_real), intent(in   ) :: vlo, vhi

    integer :: i,j,k,m,u
    integer :: idx(7)

    m = 1

    do k = bxlo(3),bxhi(3) 
      do j = bxlo(2), bxhi(2)
        do i = bxlo(1), bxhi(1)

          ! Domain interior
          ! ---------------

          idx = (/ (7*(m-1)+u,u=1,7) /)

          if ( p(i,j,k) .ne. phase) then
          
            ! Constant value in non fluidic phase
            ! p[x,y,z]  =  0
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = 0.0

          else 
            
            ! Set default seven point stencil
            a(idx) = (/6.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0/)
            rhs(m) = 0.0
            
            ! Change to one-sided Dirichlet condition at phase and domain boundaries
            if ( (p(i-1,j,k) .ne. p(i,j,k)) .or. (i .eq. domlo(1)) )  then
              a(idx) = a(idx) + (/-1.0,1.0,0.0,0.0,0.0,0.0,0.0/)
            end if
            if ( (p(i+1,j,k) .ne. p(i,j,k)) .or. (i .eq. domhi(1)) ) then
              a(idx) = a(idx) + (/-1.0,0.0,1.0,0.0,0.0,0.0,0.0/)
            end if
            if ( (p(i,j-1,k) .ne. p(i,j,k)) .or. (j .eq. domlo(2)) ) then
              a(idx) = a(idx) + (/-1.0,0.0,0.0,1.0,0.0,0.0,0.0/)
            end if
            if ( (p(i,j+1,k) .ne. p(i,j,k)) .or. (j .eq. domhi(2)) ) then
              a(idx) = a(idx) + (/-1.0,0.0,0.0,0.0,1.0,0.0,0.0/)
            end if
            if ( (p(i,j,k-1) .ne. p(i,j,k)) .or. (k .eq. domlo(3)) ) then
              a(idx) = a(idx) + (/-1.0,0.0,0.0,0.0,0.0,1.0,0.0/)
            end if 
            if ( (p(i,j,k+1) .ne. p(i,j,k)) .or. (k .eq. domhi(3)) ) then
              a(idx) = a(idx) + (/-1.0,0.0,0.0,0.0,0.0,0.0,1.0/)
            end if

          end if 

          ! Fixed Boundaries 
          ! ----------------

          ! Change to Neumann condition at domain boundaries
          ! perpendicular to flow direction
         if ( p(i,j,k) .eq. phase) then
          if ( ( dir .eq. direction_x ) .and. (p(i-1,j,k) .ne. p(i,j,k)) .and. (i .ne. domlo(1)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vhi
          end if
          if ( ( dir .eq. direction_x ) .and. (p(i+1,j,k) .ne. p(i,j,k)) .and. (i .ne. domhi(1)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vlo
          end if
          if ( ( dir .eq. direction_y ) .and. (p(i,j-1,k) .ne. p(i,j,k)) .and. (j .ne. domlo(2)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vhi
          end if
          if ( ( dir .eq. direction_y ) .and. (p(i,j+1,k) .ne. p(i,j,k)) .and. (j .ne. domhi(2)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vlo
          end if
          if ( ( dir .eq. direction_z ) .and. (p(i,j,k-1) .ne. p(i,j,k)) .and. (k .ne. domlo(3)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vhi
          end if
          if ( ( dir .eq. direction_z ) .and. (p(i,j,k+1) .ne. p(i,j,k)) .and. (k .ne. domhi(3)) ) then
            a(idx) = (/1.0,0.0,0.0,0.0,0.0,0.0,0.0/)
            rhs(m) = vlo
          end if
          
          ! Initial guess
          ! -------------
          
          if ( dir .eq. direction_x ) then
            xinit(m) = vlo + (vhi - vlo) * (i-domlo(1)) / (domhi(1)-domlo(1)) 
          end if
          if ( dir .eq. direction_y ) then
            xinit(m) = vlo + (vhi - vlo) * (j-domlo(2)) / (domhi(2)-domlo(2))
          end if
          if ( dir .eq. direction_z ) then
            xinit(m) = vlo + (vhi - vlo) * (k-domlo(3)) / (domhi(3)-domlo(3))
          end if

          m = m + 1

        end do
      end do
    end do

  end subroutine

end module
