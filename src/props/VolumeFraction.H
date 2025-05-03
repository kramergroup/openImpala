#ifndef OPENIMPALA_VOLUME_FRACTION_H // Using project-specific guard
#define OPENIMPALA_VOLUME_FRACTION_H

#include <AMReX_iMultiFab.H>
#include <AMReX.H>
#include <AMReX_REAL.H> // Include for amrex::Real

namespace OpenImpala // Add appropriate namespace
{
    /**
     * @brief Computes the volume fraction of a specific phase within an amrex::iMultiFab.
     *
     * This class takes a reference to an integer MultiFab containing phase information
     * (where phases are indicated by integer values) and provides a method to
     * calculate the volume fraction of a specified phase.
     *
     * @note The calculation uses the valid regions of the input iMultiFab; ghost cells are ignored.
     * @warning The user must ensure the iMultiFab passed to the constructor remains valid
     * for the entire lifetime of the VolumeFraction object, as only a reference is stored.
     */
    class VolumeFraction
    {
      public:
        /**
         * @brief Constructs a new VolumeFraction property calculator.
         *
         * This constructor creates a VolumeFraction object linked to the provided phase data.
         * The constructor itself performs minimal work.
         *
         * @param fm The amrex::iMultiFab containing the phase information. Must remain valid for the lifetime of this object.
         * @param phase The integer ID of the phase to calculate the volume fraction for.
         * @param comp The component index within 'fm' containing the phase IDs (default: 0).
         */
        explicit VolumeFraction(const amrex::iMultiFab& fm, const int phase = 0, int comp = 0);

        // --- Rule of Five/Six: Prevent copying, allow default move/destruction ---

        /** @brief Deleted copy constructor (due to reference member). */
        VolumeFraction(const VolumeFraction&) = delete;

        /** @brief Deleted copy assignment operator (due to reference member). */
        VolumeFraction& operator=(const VolumeFraction&) = delete;

        /** @brief Default move constructor. */
        VolumeFraction(VolumeFraction&&) = default;

        /** @brief Default move assignment operator. */
        VolumeFraction& operator=(VolumeFraction&&) = default;

        /** @brief Default virtual destructor (safer if inherited from). */
        virtual ~VolumeFraction() = default;

        // --- Computation Method ---

        /**
         * @brief Computes the count of the specified phase and the total cell count.
         *
         * Iterates over the valid cells of the stored iMultiFab, counts cells matching
         * the specified phase ID, and counts the total number of valid cells.
         * Performs parallel reduction across MPI ranks unless 'local' is true.
         * This operation can be costly depending on the grid size.
         *
         * @param[out] phase_count The number of cells belonging to the specified phase (local or global).
         * @param[out] total_count The total number of cells considered (local or global).
         * @param[in] local If true, compute counts using only data local to the current MPI rank
         * (skips MPI reduction). If false (default), compute the global counts
         * across all ranks.
         */
        void value(long long& phase_count, long long& total_count, bool local = false) const;

        // --- Optional: Convenience method to get VF directly ---
        /*
        amrex::Real value_vf(bool local = false) const {
           long long pc, tc;
           value(pc, tc, local); // Call the method that returns counts
           return (tc > 0) ? static_cast<amrex::Real>(pc) / tc : 0.0;
        }
        */

      private:
        /** @brief Reference to the MultiFab with phase information (lifetime managed externally!). */
        const amrex::iMultiFab& m_mf;

        /** @brief Phase index to calculate the volume fraction for. */
        const int m_phase;

        /** @brief Component index within m_mf containing phase data. */
        const int m_comp;

        // Note: Constructor implementation (in .cpp or inline if desired) MUST use
        // an initializer list:
        // VolumeFraction::VolumeFraction(...) : m_mf(fm), m_phase(phase), m_comp(comp) {}
    };

} // namespace OpenImpala

#endif // OPENIMPALA_VOLUME_FRACTION_H
