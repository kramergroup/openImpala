
module tortuosity_filcc_module

  use amrex_fort_module, only : amrex_real, amrex_spacedim
  use amrex_bc_types_module
  use amrex_constants_module

  implicit none

  interface tortuosity_filcc
     module procedure tortuosity_filct
     module procedure tortuosity_filbc
     module procedure tortuosity_filic
  end interface tortuosity_filcc

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  integer, parameter :: cell_type_blocked       = b'00000000'
  integer, parameter :: cell_type_free          = b'00000001'

  integer, parameter :: comp_phi = 1  ! fab component for concentration field 
  integer, parameter :: comp_ct  = 2  ! fab component for cell type

  private
  public :: tortuosity_filbc, tortuosity_filic

contains

! ::: -----------------------------------------------------------
! ::: This routine fills the cell type data structure for
! ::: the tortuosity problem. Each cell is either blocked/free or
! ::: has one or more boundaries.
! ::: 
! ::: INPUTS/OUTPUTS:
! ::: q           <=  array to fill
! ::: p            => array with phase indices
! ::: domlo,domhi  => index extent of problem domain
! ::: phase        => index of the active phase

! ::: -----------------------------------------------------------

   subroutine tortuosity_filct(q, q_lo, q_hi, q_ncomp, p, p_lo, p_hi, p_ncomp, &
                               domlo, domhi, phase) bind(c)


      implicit none

      integer,          intent(in   ) :: q_lo(3), q_hi(3)
      integer,          intent(in   ) :: p_lo(3), p_hi(3)        
      integer,          intent(in   ) :: domlo(amrex_spacedim), domhi(amrex_spacedim)
      integer,          intent(in   ) :: q_ncomp, p_ncomp
      real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1),q_lo(2):q_hi(2),q_lo(3):q_hi(3),q_ncomp)
      integer,          intent(in   ) :: p(p_lo(1):p_hi(1),p_lo(2):p_hi(2),p_lo(3):p_hi(3),p_ncomp)
      integer,          intent(in   ) :: phase

      ! Index variables
      integer :: k, j, i 

      ! Domain indices
      integer :: ilo, ihi, jlo, jhi, klo, khi
      ilo = domlo(1)
      ihi = domhi(1)
      jlo = domlo(2)
      jhi = domhi(2)
      klo = domlo(3)
      khi = domhi(3)

      ! Iterate over all cells and fill in types
      do k = klo, khi
         do j = jlo, jhi
            do i = ilo, ihi
               if (p(i,j,k,1) .eq. phase ) then
                  q(i,j,k,comp_ct) = cell_type_free 
               else
                  q(i,j,k,comp_ct) = cell_type_blocked
               end if
            end do
         end do
      end do
   
   end subroutine tortuosity_filct

! ::: -----------------------------------------------------------
! ::: This routine removes disconnected singular points from
! ::: the phase MultiFab. It goes over the MultiFab and adjusts
! ::: the phase information for voxels that are sourrounded by 
! ::: the other phase
! ::: 
! ::: INPUTS/OUTPUTS:
! ::: q            => array with phase indices
! ::: bxlo,bxhi    => valid index extenf of box
! ::: domlo,domhi  => index extent of problem domain
! ::: -----------------------------------------------------------
subroutine tortuosity_remspot(q, q_lo, q_hi, ncomp, bxlo, bxhi, domlo, domhi) bind(c)

   implicit none

   integer,          intent(in   ) :: q_lo(3), q_hi(3), domlo(3), domhi(3), bxlo(3), bxhi(3)
   integer,          intent(in   ) :: ncomp
   integer,          intent(inout) :: q(q_lo(1):q_hi(1),q_lo(2):q_hi(2),q_lo(3):q_hi(3),ncomp)

   integer :: i,j,k     ! running indices
   integer :: p(7)      ! array to hold the phase information for a seven point stencil

   do k = bxlo(3), bxhi(3)
      do j = bxlo(2), bxhi(2)
         do i = bxlo(1), bxhi(1)
            
            p(1) = q(i,j,k,1)

            ! Direction -x
            if ( i == domlo(1) ) then
               p(2) = -1
            else 
               p(2) = q(i-1,j,k,1)
            end if

            ! Direction +x
            if ( i == domhi(1) ) then
               p(3) = -1
            else 
               p(3) = q(i+1,j,k,1)
            end if

            ! Direction -y
            if ( j == domlo(2) ) then
               p(4) = -1
            else 
               p(4) = q(i,j-1,k,1)
            end if
            
            ! Direction +y
            if ( j == domhi(2) ) then
               p(5) = -1
            else 
               p(5) = q(i,j+1,k,1)
            end if

            ! Direction -z
            if ( k == domlo(3) ) then
               p(6) = -1
            else 
               p(6) = q(i,j,k-1,1)
            end if

            ! Direction +z
            if ( k == domhi(3) ) then
               p(7) = -1
            else 
               p(7) = q(i,j,k+1,1)
            end if

            if ((p(1) == p(2)) .or. &
                (p(1) == p(3)) .or. &
                (p(1) == p(4)) .or. &
                (p(1) == p(5)) .or. &
                (p(1) == p(6)) .or. &
                (p(1) == p(7))) then
                  ! Do nothing - point is connected
            else
               if (q(i,j,k,1) == 0) then
                  q(i,j,k,1) = 1 
               else 
                  q(i,j,k,1) = 0
               end if
            end if

         end do
      end do
   end do

end subroutine tortuosity_remspot


! ::: -----------------------------------------------------------
! ::: This routine fills the Dirichlet boundary conditions for
! ::: the tortuosity problem. There are two Dirichlet conditions
! ::: on opposing faces perpendicular to the principal flow direction.
! :::
! ::: INPUTS/OUTPUTS:
! ::: q           <=  array to fill
! ::: p            => array with phase indices
! ::: domlo,domhi  => index extent of problem domain
! ::: vlo,vhi      => values at the boundary
! ::: bc	          => array of boundary flags bc(SPACEDIM,lo:hi)
! ::: -----------------------------------------------------------

  subroutine tortuosity_filbc(q, q_lo, q_hi, ncomp, p, p_lo, p_hi, p_ncomp, &
                              domlo, domhi, vlo, vhi, bc) bind(c)

    implicit none

    integer,          intent(in   ) :: q_lo(3), q_hi(3)
    integer,          intent(in   ) :: p_lo(3), p_hi(3)
    integer,          intent(in   ) :: domlo(amrex_spacedim), domhi(amrex_spacedim)
    integer,          intent(in   ) :: ncomp, p_ncomp
    real(amrex_real), intent(in   ) :: vlo, vhi
    real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1),q_lo(2):q_hi(2),q_lo(3):q_hi(3),ncomp)
    integer,          intent(in   ) :: p(p_lo(1):p_hi(1),p_lo(2):p_hi(2),p_lo(3):p_hi(3),p_ncomp)
    integer,          intent(in   ) :: bc(amrex_spacedim,2)

    integer :: ilo, ihi, jlo, jhi, klo, khi
    integer :: i, j, k, n
    integer :: imin, imax, jmin, jmax, kmin, kmax

    ilo = domlo(1)
    ihi = domhi(1)
    jlo = domlo(2)
    jhi = domhi(2)
    klo = domlo(3)
    khi = domhi(3)

    do n = 1, ncomp

          if (bc(1,1) .eq. amrex_bc_ext_dir) then

            do k = domlo(3), domhi(3)
               do j = domlo(2), domhi(2)
                  do i = domlo(1), domhi(1)
                     if ( p(i,j,k) .e. 0) then
                        if ( (p(i-1,j,k) .ne. p(i,j,k)) )  then
                            q(i,j,k,n) = vhi 
                        end if
                        if ( (p(i+1,j,k) .ne. p(i,j,k)) ) then
                            q(i,j,k,n) = vlo      
                        end if
                     end if
                  end do
               end do
            end do

          end if

          if (bc(2,1) .eq. amrex_bc_ext_dir) then

            do k = domlo(3), domhi(3)
               do j = domlo(2), domhi(2)
                  do i = domlo(1), domhi(1)
                     if ( p(i,j,k) .e. 0) then
                        if ( (p(i,j-1,k) .ne. p(i,j,k)) )  then
                            q(i,j,k,n) = vhi 
                        end if
                        if ( (p(i,j+1,k) .ne. p(i,j,k)) ) then
                            q(i,j,k,n) = vlo 
                        end if
                     end if                            
                  end do
               end do
            end do

          end if

          if (bc(3,1) .eq. amrex_bc_ext_dir) then
          
            do k = domlo(3), domhi(3)
               do j = domlo(2), domhi(2)
                  do i = domlo(1), domhi(1)
                     if ( p(i,j,k) .e. 0) then
                        if ( (p(i,j,k-1) .ne. p(i,j,k)) )  then
                            q(i,j,k,n) = vhi 
                        end if
                        if ( (p(i,j,k+1) .ne. p(i,j,k)) ) then
                            q(i,j,k,n) = vlo 
                        end if
                     end if                            
                  end do
               end do
            end do

          end if

    end do

  end subroutine tortuosity_filbc



! ::: -----------------------------------------------------------
! ::: This routine fills the initial conditions for
! ::: the tortuosity problem. The problem is initialised by a
! ::: linear gradient along the principal flow direction. The
! ::: routine only fill cells of the phase of interest and
! ::: initialises all other cells with zero.
! :::
! ::: INPUTS/OUTPUTS:
! ::: q           <=  array to fill
! ::: p            => array with phase indices
! ::: domlo,domhi  => index extent of problem domain
! ::: vlo,vhi      => values at the boundary
! ::: phase        => index of the phase of interest
! ::: dir          => principal flow direction (X=0, Y=1, Z=2)
! ::: -----------------------------------------------------------

  subroutine tortuosity_filic(q, q_lo, q_hi, ncomp, p, p_lo, p_hi, p_ncomp, &
                              lo, hi, domlo, domhi, vlo, vhi, phase, dir) bind(c)

   implicit none

   integer,          intent(in   ) :: q_lo(3), q_hi(3), p_lo(3), p_hi(3)
   integer,          intent(in   ) :: lo(3), hi(3)
   integer,          intent(in   ) :: domlo(amrex_spacedim), domhi(amrex_spacedim)
   integer,          intent(in   ) :: ncomp, p_ncomp
   real(amrex_real), intent(in   ) :: vlo, vhi
   real(amrex_real), intent(inout) :: q(q_lo(1):q_hi(1),q_lo(2):q_hi(2),q_lo(3):q_hi(3),ncomp)
   integer,          intent(in   ) :: p(p_lo(1):p_hi(1),p_lo(2):p_hi(2),p_lo(3):p_hi(3),p_ncomp)
   integer,          intent(in   ) :: phase
   integer,          intent(in   ) :: dir

   integer :: ilo, ihi, jlo, jhi, klo, khi
   integer :: i, j, k, n
   integer :: imin, imax, jmin, jmax, kmin, kmax

   ilo = domlo(1)
   ihi = domhi(1)

#if AMREX_SPACEDIM >= 2
   jlo = domlo(2)
   jhi = domhi(2)
#endif

#if AMREX_SPACEDIM == 3
   klo = domlo(3)
   khi = domhi(3)
#endif

   do n = 1, ncomp

      if (dir .eq. direction_x) then

         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  if (p(i,j,k,1) == phase) then
                     q(i,j,k,n) = vlo + 1.0*(i-domlo(1)+1)/(domhi(1)-domlo(1)+2) * (vhi-vlo)
                  else
                     q(i,j,k,n) = 0.0
                  end if
               end do
            end do
         end do

      end if

#if AMREX_SPACEDIM >= 2

      if (dir .eq. direction_y) then

         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  if (p(i,j,k,1) == phase) then
                     q(i,j,k,n) = vlo + 1.0*(j-domlo(2))/(domhi(2)-domlo(2)) * (vhi-vlo)
                  else
                     q(i,j,k,n) = 0.0
                  end if
               end do
            end do
         end do

      end if

#endif


#if AMREX_SPACEDIM == 3

      if (dir .eq. direction_z) then

         do k = lo(3), hi(3)
            do j = lo(2), hi(2)
               do i = lo(1), hi(1)
                  if (p(i,j,k,1) == phase) then
                     q(i,j,k,n) = vlo + 1.0*(k-domlo(3))/(domhi(3)-domlo(3)) * (vhi-vlo)
                  else
                     q(i,j,k,n) = 0.0
                  end if
               end do
            end do
         end do

      end if

#endif

   end do

 end subroutine tortuosity_filic

end module tortuosity_filcc_module
