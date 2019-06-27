#ifndef TortuosityDirect_H
#define TortuosityDirect_H

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include "Tortuosity.H"

 
/** Computes the tortuosity of a porous structure
 * 
 * This class solved the Poisson equation on a porous structure and extracts
 * the tortuosity factor using a simple Finite Volume iterative solver.
 * 
 * The implementation is parallel but not very efficient. It mainly serves as
 * a test basis for more advanced solver.
 * 
 */
class TortuosityDirect : public Tortuosity
{

  public: 

    /** Construct new Tortuosity solver 
     * 
     * This Tortuosity solver calculates the tortuosity of a phase solving the Laplace equation on the space occupied by the phase
     * with a flow in the given direction.
     * 
     * @param mf the amrex::iMultiFab describing the phase space
     * @param phase the index of the phase for which the tortuosity should be calculated
     * @param dir the principal direction of the flow
     * 
     */
    TortuosityDirect(const amrex::Geometry& geom, 
                     const amrex::BoxArray& ba, 
                     const amrex::DistributionMapping& dm, 
                     const amrex::iMultiFab& mf, 
                     const int phase, 
                     const Direction dir);

    
    amrex::Real value(const bool refresh = false) override;
    
  private: 

    /** Compute solution
     * 
     * This function computes the tortuosity value from a solution of Laplace's equation
     * on the multi-phase domain.
     * 
     */
    bool solve();

    /** Compute global fluxes
     * 
     * This function computes the global fluxes in and out of the domain.
     * A call to solve() is expected before this funcition is called.
     * 
     * @param the flux into the domain
     * @param the flux out of the domain
     */
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;

    /** Compute the change residual between iterations
     * 
     * This functions sums the change in potential to determine a stop criterium
     * for the iterative solution.
     * 
     * @param old scalar field MultiFab
     * @param new scalar field MultiFab
     *
     */
    amrex::Real residual(const amrex::MultiFab& phiold, const amrex::MultiFab& phinew) const;

    /** Create boundary condition description of the domain 
     * 
     * This function creates a BCRec describing the boundary condition type.
     * 
     */
    void initialiseBoundaryConditions() ;

    /** Create data structures to hold fluxes during the computation
     * 
     * This function initialises an AMREX_SPACEDIM sized array of MultiFabs 
     * to hold fluxes during the computation. Each Multifab is nodal in the
     * direction of the flux and cell centered in the other directions.
     */
    void initialiseFluxMultiFabs();

    /** Fill domain boundary values
     * 
     * This method fills the domain boundary values. It will add ghost cells for every non-periodic 
     * direction and use these to express the boundary condition
     * 
     * @param amrex::MultiFab holding the values
     * 
     */
    void fillDomainBoundary(amrex::MultiFab& phi);

    /** Fill Cell Types
     * 
     * This method fills a iMultiFab with cell type information. This will be passed to the 
     * operator kernel to implement different kernels depending on cell type.
     * 
     * @param amrex::MultiFab holding the values
     * 
     */
    void fillCellTypes(amrex::MultiFab& phi);

    /** Fill initial state
     * 
     * This methods linearly interpolates between inlet and outlet values
     * 
     * @param amrex::MultiFab holding the values
     * 
     */
    void fillInitialState(amrex::MultiFab& phi);


    /** Advance solution 
     * 
     * This routine is the core routine advancing the iterative solution
     * 
     * @param last MultiFab with current solution
     * @param new MultiFab with advanced solution
     */
    void advance(amrex::MultiFab& phi_old, amrex::MultiFab& phi_new);

    /** Reference to the multifab with phase data */
    const amrex::iMultiFab& m_mf_phase;

    /** flux multifab arrays **/
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> m_flux;
    
    const amrex::Geometry& m_geom;

    const amrex::BoxArray& m_ba;

    const amrex::DistributionMapping& m_dm;

    /** Index of the phase of interest */
    const int m_phase;
    
    /** Principal direction of the flow */
    const Direction m_dir;
    
    /** Data structure describing the boundary condition type */
    amrex::BCRec m_bc;

    /** Maximum Number of iterative steps */
    size_t m_n_steps = 100000;

    /** Convergence criterium for change of phi */
    amrex::Real m_eps = 1e-4;

    /** Value buffer */
    amrex::Real m_value;
    
    /** Flags repeated call of value() */
    bool m_first_call;

    /** Inlet concentration value */
    const amrex::Real m_vlo = -1.0;

    /** Outlet concentration value */
    const amrex::Real m_vhi = +1.0;
};

#endif