#include "BoxCrop.H"
#include <AMReX_DistributionMapping.H>

template <typename MultiFab>
void BoxCropFilter<MultiFab>::filter(const MultiFab& source, MultiFab& target) {

    // Create the receiving MultiFab
    // TODO: Check if not using ghost cells is ok here. Not sure if we can later add ghost cells
    target.define(m_cropBox,m_dm,source.nComp(),source.nGrow());
    target.ParallelCopy(source);

}

template void BoxCropFilter<amrex::iMultiFab>::filter(const amrex::iMultiFab&, amrex::iMultiFab&);
template void BoxCropFilter<amrex::MultiFab>::filter(const amrex::MultiFab&, amrex::MultiFab&);