#include "VolumeFraction.H"

#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX.H>

VolumeFraction::VolumeFraction(const amrex::iMultiFab& fm, const int phase) : m_mf(fm), m_phase(phase)
{

}

amrex::Real VolumeFraction::value(bool local) const {

  size_t v_phase(0);
  size_t v_all(0);

  // Iterate over all boxes and count cells with value=m_phase
  for (amrex::MFIter mfi(m_mf); mfi.isValid(); ++mfi) // Loop over grids
  {
    const amrex::Box& box = mfi.validbox();
    const amrex::IArrayBox& fab = m_mf[mfi];
    
    size_t idx;
    // Iterate over all cells in Box and threshold
    for (amrex::BoxIterator bit(box); bit.ok(); ++bit) 
    {
      if ( fab(bit(),0) == m_phase ) ++v_phase;
      ++v_all;
    } 
  }

  // Reduce parallel processes
  if (!local) {
        amrex::ParallelAllReduce::Sum(v_phase, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(v_all, amrex::ParallelContext::CommunicatorSub());
  }

  return amrex::Real(v_phase) / amrex::Real(v_all);;

}

