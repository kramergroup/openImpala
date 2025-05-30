// Add includes if needed (already present based on your snippet)
#include "VolumeFraction.H"
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MFIter.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>

namespace OpenImpala
{

// Constructor remains the same...
VolumeFraction::VolumeFraction(const amrex::iMultiFab& fm, const int phase, int comp)
  : m_mf(fm), m_phase(phase), m_comp(comp)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_comp >= 0 && m_comp < m_mf.nComp(),
                                     "VolumeFraction: Component index out of bounds.");
}

// --- value() Method --- MODIFIED SIGNATURE
void VolumeFraction::value(long long& phase_count, long long& total_count, bool local) const
{
    // Use long long for counts to avoid overflow and for portability with MPI reductions
    long long local_phase_count = 0;
    long long local_total_count = 0;

    // Get the target phase ID and component index for use in the lambda/loop
    const int target_phase = m_phase;
    const int phase_comp = m_comp;

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:local_phase_count, local_total_count)
#endif
    for (amrex::MFIter mfi(m_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); // Use tilebox for potential OMP tiling
        const auto& fab = m_mf.const_array(mfi, phase_comp); // Get Array4 for the correct component

        amrex::Loop(bx, [&] (int i, int j, int k) // Capture locals by reference for OMP reduction
        {
            if (fab(i, j, k) == target_phase) {
                local_phase_count += 1; // Directly increment thread-local sum (OMP handles reduction)
            }
        });

        local_total_count += bx.numPts(); // Add number of cells in this tilebox
    }

    // Perform parallel reduction across MPI ranks if global value is requested
    if (!local)
    {
        // Use ParallelAllReduce to sum across all ranks into rank 0, then broadcast
        // Or simply sum across all ranks and leave the result on all ranks
        // ParallelAllReduce::Sum sums on all ranks.
        amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
        amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
    }

    // Assign the final counts (either local or globally reduced) to the output parameters
    phase_count = local_phase_count;
    total_count = local_total_count;

    // No return value needed (void function)
    // The calculation phase_count / total_count will now happen in the calling code.
}

} // namespace OpenImpala
