#include "VolumeFraction.H" // Include the header for the class definition

// AMReX includes
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H> // For ParallelAllReduce::Sum
// #include <AMReX_Reduce.H>       // <<< REMOVED: No longer using Reduce::Sum >>>
#include <AMReX_MultiFabUtil.H>   // For Array4, amrex::Loop
#include <AMReX_MFIter.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX.H>

namespace OpenImpala // Use the same namespace as in the header
{

    // --- Constructor ---

    VolumeFraction::VolumeFraction(const amrex::iMultiFab& fm, const int phase, int comp)
      : m_mf(fm),        // Initialize const reference member
        m_phase(phase),  // Initialize const int member
        m_comp(comp)     // Initialize int member
    {
        // Check that the specified component index is valid for the given MultiFab
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_comp >= 0 && m_comp < m_mf.nComp(),
                                         "VolumeFraction: Component index out of bounds.");
    }

    // --- value() Method ---

    amrex::Real VolumeFraction::value(bool local) const
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
        // FIX 1: Use amrex:: namespace (already fixed)
        for (amrex::MFIter mfi(m_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox(); // Use tilebox for potential OMP tiling
            const auto& fab = m_mf.const_array(mfi, phase_comp); // Get Array4 for the correct component

            // FIX 2: Replace Reduce::Sum with amrex::Loop
            amrex::Loop(bx, [&] (int i, int j, int k) // Capture locals by reference for OMP reduction
            {
                if (fab(i, j, k) == target_phase) {
                    local_phase_count += 1; // Directly increment thread-local sum (OMP handles reduction)
                }
            });
            // End of replacement

            local_total_count += bx.numPts(); // Add number of cells in this tilebox
        }

        // Perform parallel reduction across MPI ranks if global value is requested
        if (!local)
        {
            amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
            amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
        }

        // Calculate the volume fraction, protecting against division by zero
        return (local_total_count > 0)
               ? static_cast<amrex::Real>(local_phase_count) / local_total_count
               : 0.0;
    }

} // namespace OpenImpala
