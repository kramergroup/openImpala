#include "BoxCrop.H"
#include <AMReX_DistributionMapping.H>

template <typename MultiFab>
MultiFab BoxCropFilter<MultiFab>::filter(const MultiFab& source) {

    // Create the receiving MultiFab
    // TODO: Check if not using ghost cells is ok here. Not sure if we can later add ghost cells
    MultiFab mf_target;
    mf_target.define(m_cropBox,m_dm,source.nComp(),source.nGrow());
    mf_target.ParallelCopy(source);
    
    
    return mf_target;
}

template amrex::iMultiFab BoxCropFilter<amrex::iMultiFab>::filter(const amrex::iMultiFab&);
template amrex::MultiFab BoxCropFilter<amrex::MultiFab>::filter(const amrex::MultiFab&);