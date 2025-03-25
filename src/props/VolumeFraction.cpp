#include "VolumeFraction.H" // Include the header for the class definition

// AMReX includes
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelReduce.H> // For ParallelAllReduce::Sum
#include <AMReX_Reduce.H>       // For amrex::ReduceSum
#include <AMReX_MultiFabUtil.H> // For amrex::Loop, Array4

namespace OpenImpala // Use the same namespace as in the header
{

    // --- Constructor ---

    VolumeFraction::VolumeFraction(const amrex::iMultiFab& fm, const int phase, int comp)
      : m_mf(fm),         // Initialize const reference member
        m_phase(phase),   // Initialize const int member
        m_comp(comp)      // Initialize int member
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

        // Get the target phase ID and component index for use in the lambda
        const int target_phase = m_phase;
        const int phase_comp = m_comp;

        // Iterate over all valid boxes in the MultiFab
        // Using amrex::ReduceSum is often more concise and potentially faster than manual looping
        // Requires C++14 for generic lambdas if used, but standard lambdas work fine too.
#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:local_phase_count, local_total_count)
#endif
        for (amrex::MFIter mfi(m_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.tilebox(); // Use tilebox for potential OMP tiling
            const auto& fab = m_mf.const_array(mfi, phase_comp); // Get Array4 for the correct component

            // Use ReduceSum on the tilebox to count cells where fab(i,j,k) == target_phase
            // The lambda returns 1LL (long long 1) if condition met, 0LL otherwise
            long long box_phase_count = amrex::Reduce::Sum(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> long long {
                return (fab(i, j, k) == target_phase) ? 1LL : 0LL;
            });

            local_phase_count += box_phase_count;
            local_total_count += bx.numPts(); // Add number of cells in this tilebox
        }

        // Perform parallel reduction across MPI ranks if global value is requested
        if (!local)
        {
            // Reduce counts over the communicator group associated with the MultiFab's DistributionMapping
            // CommunicatorSub() is typically correct here. Use CommunicatorAll() if reducing over MPI_COMM_WORLD is intended.
            amrex::ParallelAllReduce::Sum(local_phase_count, amrex::ParallelContext::CommunicatorSub());
            amrex::ParallelAllReduce::Sum(local_total_count, amrex::ParallelContext::CommunicatorSub());
        }

        // Calculate the volume fraction, protecting against division by zero
        return (local_total_count > 0)
               ? static_cast<amrex::Real>(local_phase_count) / local_total_count
               : 0.0;
    }

} // namespace OpenImpala
