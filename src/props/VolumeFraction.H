#ifndef VolumeFraction_H
#define VolumeFraction_H

#include <AMReX_iMultiFab.H>
#include <AMReX.H>

/** Computes porosity of a porous structure
 * 
 * This class computes the volume fraction of a phase in
 * a porous structure. It expects a iMultiFab with phase information. Phases
 * should be indicated by integer values.
 */

class VolumeFraction 
{

  public:

    /** Constructs a new VolumeFraction property
     * 
     * This constructor creates a VolumeFraction property. The constructor is lean.
     *  
     * @param fm amrex::iMultiFab containing the phase information
     * @param phase the phase to evaluate (default: 0)
     */
    VolumeFraction(const amrex::iMultiFab& fm, const int phase = 0);

    /** Compute volume fraction
     * 
     * Computes the volume fraction of the phase in the iMultiFab
     * This operator is costly and can execute in parallel.
     * 
     * @param bool only process local values in parallel computations if true (default: false)
     * 
     * @return volume fraction of the phase
     */
    amrex::Real value(bool local=false) const;

  private:

    /** Reference to the MultiFab with phase information */
    const amrex::iMultiFab& m_mf;

    /** Phase index */
    const int m_phase;

};

#endif
