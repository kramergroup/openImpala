module tortuosity_poisson_3d_module

  use amrex_fort_module, only : amrex_real
  implicit none

  integer, parameter :: direction_x = 0
  integer, parameter :: direction_y = 1
  integer, parameter :: direction_z = 2

  integer, parameter :: cell_type_blocked       = b'00000000'
  integer, parameter :: cell_type_free          = b'00000001'
      
  integer, parameter :: cell_type_boundary_x_lo = b'00000010'
  integer, parameter :: cell_type_boundary_x_hi = b'00000100'
  integer, parameter :: cell_type_boundary_y_lo = b'00001000'
  integer, parameter :: cell_type_boundary_y_hi = b'00010000'
  integer, parameter :: cell_type_boundary_z_lo = b'00100000'
  integer, parameter :: cell_type_boundary_z_hi = b'01000000'

  integer, parameter :: cell_type_boundary_cell = b'01111110' 

  integer, parameter :: comp_phi = 1  ! fab component for concentration field 
  integer, parameter :: comp_ct  = 2  ! fab component for cell type

  private
  public :: tortuosity_poisson_flux, tortuosity_poisson_update, tortuosity_poisson_fio

contains

  subroutine tortuosity_poisson_flux (lo, hi, fx, fxlo, fxhi, fy, fylo, fyhi, &
       fz, fzlo, fzhi, sol, slo, shi, scomp, dxinv, face_only) &
       bind(c, name='tortuosity_poisson_flux')

    integer, dimension(3), intent(in   ) :: lo, hi, fxlo, fxhi, fylo, fyhi, fzlo, fzhi, slo, shi
    integer,               intent(in   ) :: scomp
    integer, value,        intent(in   ) :: face_only
    real(amrex_real),      intent(inout) :: fx (fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3))
    real(amrex_real),      intent(inout) :: fy (fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3))
    real(amrex_real),      intent(inout) :: fz (fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3))
    real(amrex_real),      intent(in   ) :: sol( slo(1): shi(1), slo(2): shi(2), slo(3): shi(3),scomp)
    
    real(amrex_real) :: dxinv(3)

    integer :: i,j,k
    real(amrex_real) :: dhx, dhy, dhz

    dhx = dxinv(1)
    dhy = dxinv(2)
    dhz = dxinv(3)

    if (face_only .eq. 1) then
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)+1, hi(1)+1-lo(1)
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i-1,j,k,comp_ct) .eq. cell_type_blocked) ) then
                  fx(i,j,k) = 0.0
               else
                  fx(i,j,k) = dhx * (sol(i,j,k,comp_phi) - sol(i-1,j,k,comp_phi))
               end if
             end do
          end do
       end do
       
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)+1, hi(2)+1-lo(2)
             do i = lo(1), hi(1)
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i,j-1,k,comp_ct) .eq. cell_type_blocked) ) then
                  fy(i,j,k) = 0.0
               else
                  fy(i,j,k) = dhy * (sol(i,j,k,comp_phi) - sol(i,j-1,k,comp_phi))
               end if
             end do
          end do
       end do
       
       do       k = lo(3), hi(3)+1, hi(3)+1-lo(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i,j,k-1,comp_ct) .eq. cell_type_blocked) ) then
                  fz(i,j,k) = 0.0
               else
                  fz(i,j,k) = dhz * (sol(i,j,k,comp_phi) - sol(i,j,k-1,comp_phi))
               end if
             end do
          end do
       end do

    else

       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)+1
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i-1,j,k,comp_ct) .eq. cell_type_blocked) ) then
                  fx(i,j,k) = 0.0
               else
                  fx(i,j,k) = dhx * (sol(i,j,k,comp_phi) - sol(i-1,j,k,comp_phi))
               end if
             end do
          end do
       end do
 
       do       k = lo(3), hi(3)
          do    j = lo(2), hi(2)+1
             do i = lo(1), hi(1)
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i,j-1,k,comp_ct) .eq. cell_type_blocked) ) then
                  fy(i,j,k) = 0.0
               else
                  fy(i,j,k) = dhy * (sol(i,j,k,comp_phi) - sol(i,j-1,k,comp_phi))
               end if
             end do
          end do
       end do
       
       do       k = lo(3), hi(3)+1
          do    j = lo(2), hi(2)
             do i = lo(1), hi(1)
               if ( (sol(i,j,k,comp_ct) .eq. cell_type_blocked) .or. (sol(i,j,k-1,comp_ct) .eq. cell_type_blocked) ) then
                  fz(i,j,k) = 0.0
               else
                  fz(i,j,k) = dhz * (sol(i,j,k,comp_phi) - sol(i,j,k-1,comp_phi))
               end if
             end do
          end do
       end do
    end if

  end subroutine tortuosity_poisson_flux

  subroutine tortuosity_poisson_update(lo,hi, p,plo,phi,pcomp, n,nlo,nhi,ncomp, &
                                       fx,fxlo,fxhi, fy,fylo,fyhi, fz,fzlo,fzhi, &
                                       dxinv, dt) bind(c, name='tortuosity_poisson_update')
    
    integer, dimension(3), intent(in   ) :: lo,hi, plo,phi, nlo,nhi, fxlo,fxhi, fylo,fyhi, fzlo,fzhi
    integer,               intent(in   ) :: pcomp, ncomp
    real(amrex_real),      intent(in   ) :: p(plo(1):phi(1),plo(2):phi(2),plo(3):phi(3),pcomp)
    real(amrex_real),      intent(inout) :: n(plo(1):nhi(1),nlo(2):nhi(2),nlo(3):nhi(3),ncomp)
    real(amrex_real),      intent(inout) :: fx (fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3))
    real(amrex_real),      intent(inout) :: fy (fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3))
    real(amrex_real),      intent(inout) :: fz (fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3))
    real(amrex_real),      intent(in   ) :: dxinv(3), dt

    real(amrex_real) :: dtdx(3)
    integer          :: i,j,k

    dtdx = dt*dxinv

    do k = lo(3),hi(3)
      do j = lo(2), hi(2)
         do i = lo(1), hi(1)
            n(i,j,k,comp_phi) = p(i,j,k,comp_phi) &
                     + dtdx(1) * (fx(i+1,j  ,k  ) - fx(i,j,k)) &
                     + dtdx(2) * (fy(i  ,j+1,k  ) - fy(i,j,k)) &
                     + dtdx(3) * (fz(i  ,j  ,k+1) - fz(i,j,k))
         end do
      end do
   end do

  end subroutine tortuosity_poisson_update

  subroutine tortuosity_poisson_fio(lo,hi,fx,fxlo,fxhi,fy,fylo,fyhi,fz,fzlo,fzhi,dir,flux_in,flux_out) &
   bind(c,name='tortuosity_poisson_fio')

    integer, dimension(3), intent(in   ) :: lo,hi, fxlo,fxhi, fylo,fyhi, fzlo,fzhi
    real(amrex_real),      intent(inout) :: fx (fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3))
    real(amrex_real),      intent(inout) :: fy (fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3))
    real(amrex_real),      intent(inout) :: fz (fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3))
    integer,               intent(in   ) :: dir
    real(amrex_real),      intent(inout) :: flux_in, flux_out

    integer :: i,j 

    flux_in = 0.0
    flux_out = 0.0

    select case (dir)
    case (direction_x)
      if (fxlo(1) .le. lo(1)) then
         do i = fxlo(2),fxhi(2)
            do j = fxlo(3),fxhi(3)
               flux_in = flux_in + fx(fxlo(1),i,j)
            end do
         end do
         
      end if
      if (fxhi(1) .ge. hi(1)) then
         do i = fxlo(2),fxhi(2)
            do j = fxlo(3),fxhi(3)
               flux_out = flux_out + fx(fxhi(1),i,j)
            end do
         end do
      end if
    case (direction_y)
      if (fxlo(2) .le. lo(2)) then
         do i = fxlo(1),fxhi(1)
            do j = fxlo(3),fxhi(3)
               flux_in = flux_in + fy(i,fylo(2),j)
            end do
         end do
      end if
      if (fxhi(2) .ge. hi(2)) then
         do i = fxlo(1),fxhi(1)
            do j = fxlo(3),fxhi(3)
               flux_out = flux_out + fy(i,fyhi(2),j)
            end do
         end do
      end if
    case (direction_z)
      if (fxlo(3) .le. lo(3)) then
         do i = fxlo(1),fxhi(1)
            do j = fxlo(2),fxhi(2)
               flux_in = flux_in + fz(i,j,fzlo(3))
            end do
         end do
      end if
      if (fxhi(3) .ge. hi(3)) then
         do i = fxlo(1),fxhi(1)
            do j = fxlo(2),fxhi(2)
               flux_out = flux_out + fz(i,j,fzhi(3))
            end do
         end do
      end if
    end select

  end subroutine tortuosity_poisson_fio

end module tortuosity_poisson_3d_module
