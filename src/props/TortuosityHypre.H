#ifndef TortuosityHypre_H
#define TortuosityHypre_H

#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

#include <HYPRE.h>
#include <HYPRE_struct_ls.h>

#include "Tortuosity.H"

 
/** Computes the tortuosity of a porous structure
 * 
 * This class solved the Poisson equation on a porous structure and extracts
 * the tortuosity factor using a simple Finite Volume iterative solver.
 * 
 * The implementation uses the Hypre library to solve the stationary Poisson equation
 * 
 */
class TortuosityHypre : public Tortuosity
{

  public: 

    enum SolverType {
      Jacobi,
      GMRES,
      FlexGMRES
    };

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
    TortuosityHypre(const amrex::Geometry& geom, 
                    const amrex::BoxArray& ba, 
                    const amrex::DistributionMapping& dm, 
                    amrex::iMultiFab& mf, 
                    const int phase, 
                    const Direction dir,
                    const SolverType = GMRES);

    
    amrex::Real value(const bool refresh = false) override;
    
    /**
     * Extract lower bound from box in HYPRE format
     * 
     * @param box describing rectangluar domain
     *
     */
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> loV (const amrex::Box& b);
    
    /**
     * Extract upper bound from box in HYPRE format
     * 
     * @param box describing rectangluar domain
     *
     */
    static amrex::Array<HYPRE_Int,AMREX_SPACEDIM> hiV (const amrex::Box& b);


  private: 

    /** Compute solution
     * 
     * This function computes the tortuosity value from a solution of Laplace's equation
     * on the multi-phase domain.
     * 
     */
    bool solve();


    /** Setup HYPRE grids
     * 
     * This function sets up the Hypre solve grid structure
     * 
     */
    void setupGrids();


    /** Setup HYPRE stencil
     * 
     * This function initialises the Hypre stencil
     *
     */
    void setupStencil();

    /** Setup HYPRE matrix
     * 
     * This function initialises the Hypre matrix A
     *
     */
    void setupMatrixEquation();

    /** Precondition PhaseFab
     * 
     * This function goes over the phase fab and removed 
     * disconnected points which would lead to ill-defined
     * behaviour during solving
     * 
     */
    void preconditionPhaseFab();

    /** Retrieve solution from Hypre data structures
     * 
     * This function retrieves the solution field from 
     * the Hypre data structures
     * 
     */
    void getSolution (amrex::MultiFab& soln, int ncomp=0);


    /** Retrieve the cell types 
     * 
     * This function fills a MultiFab with logical
     * information of the domain
     * 
     * Values: 1.0 - fluid phase
     *         0.0 - solid phase
     */
    void getCellTypes(amrex::MultiFab& phi, int ncomp=1);

        /** Compute global fluxes
     * 
     * This function computes the global fluxes in and out of the domain.
     * A call to solve() is expected before this funcition is called.
     * 
     * @param the flux into the domain
     * @param the flux out of the domain
     */
    void global_fluxes(amrex::Real& fxin, amrex::Real& fxout) const;

    /** Define the Hypre solver to us
     * 
     * This variable defines the Hypre solver to use. A number of 
     * solvers have been made available. They mostly differ in 
     * their time to solve the problem and numerical stability.
     * 
     */
    const SolverType m_solvertype;



    /** Fab geometry 
     *
     * These variable define the Amrex box geometry and 
     * distribution. All MultiFabs are expected to have 
     * the same layout defined by these variables.
     *  
     */
    const amrex::Geometry&            m_geom;
    const amrex::BoxArray&            m_ba;
    const amrex::DistributionMapping& m_dm;

    /** Reference to the multifab with phase data */
    amrex::iMultiFab& m_mf_phase;

    /** Multifab holding the solution */
    amrex::MultiFab m_mf_phi;



    /** Index of the phase of interest */
    const int m_phase;
    
    /** Principal direction of the flow */
    const Direction m_dir;


    /** Convergence criteria 
     * 
     * These variables define the convergence criteria 
     * for the solver. 
     * 
     */
    amrex::Real  m_eps = 1e-4;
    unsigned int m_maxiter = 1e4;

    /** Value buffer */
    amrex::Real m_value;
    
    /** Flags repeated call of value() */
    bool m_first_call;

    /** Inlet concentration value */
    const amrex::Real m_vlo = -1.0;

    /** Outlet concentration value */
    const amrex::Real m_vhi = +1.0;



    /** Hypre variables
     * 
     * These variables hold the Hypre data structures.
     * 
     */
    HYPRE_StructGrid    m_grid;
    HYPRE_StructStencil m_stencil;
    HYPRE_StructMatrix  m_A;  // Matrix
    HYPRE_StructVector  m_b;  // RHS vector
    HYPRE_StructVector  m_x;  // Solution vector

};

#endif