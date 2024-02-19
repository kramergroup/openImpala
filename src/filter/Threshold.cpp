#include "Threshold.H"


template <typename MF>
void ThresholdFilter<MF>::filter(const MF& source, amrex::iMultiFab& target) {

    using ST = typename MF::value_type;
    using TT = typename amrex::iMultiFab::value_type;

    // Reshape the target MultiFab
    target.define(source.boxarray,source.distributionMap,source.nComp(),source.nGrow());
    
    for (amrex::MFIter mfi(source); mfi.isValid(); ++mfi) // Loop over grids
    {
        const amrex::Box& bx = mfi.validbox();
        const amrex::Array4<const ST>& source_array = source.array(mfi);
        amrex::Array4<TT> target_array = target.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){

            target_array(i,j,k) = source_array(i,j,k) < m_threshold;

        });
    }

}

template void ThresholdFilter<amrex::iMultiFab>::filter(const amrex::iMultiFab&,amrex::iMultiFab&);
template void ThresholdFilter<amrex::MultiFab>::filter(const amrex::MultiFab&,amrex::iMultiFab&);