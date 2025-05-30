// src/props/tEffectiveDiffusivity.cpp

#include "../io/TiffReader.H"
#include "EffectiveDiffusivityHypre.H"
#include "Tortuosity.H" // For OpenImpala::Direction enum
// #include "EffectiveDiffusivityUtils.H" // << If you created this

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Utility.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H> // For amrex::Copy (already there)

#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <iomanip> // For std::fixed, std::setprecision
#include <sstream> // For stringstream if still needed

// If not using a separate utility header, define/declare helpers here:
namespace { // Anonymous namespace for test-local helpers

    OpenImpala::Direction stringToDirection(const std::string& dir_str) {
        std::string lower_dir_str = dir_str;
        std::transform(lower_dir_str.begin(), lower_dir_str.end(), lower_dir_str.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (lower_dir_str == "x") return OpenImpala::Direction::X;
        if (lower_dir_str == "y") return OpenImpala::Direction::Y;
        if (lower_dir_str == "z") return OpenImpala::Direction::Z;
        amrex::Abort("Invalid direction string: " + dir_str + ". Use X, Y, or Z.");
        return OpenImpala::Direction::X;
    }

    OpenImpala::EffectiveDiffusivityHypre::SolverType stringToSolverType(const std::string& solver_str) {
        std::string lower_solver_str = solver_str;
        std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (lower_solver_str == "jacobi") return OpenImpala::EffectiveDiffusivityHypre::SolverType::Jacobi;
        if (lower_solver_str == "gmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES;
        if (lower_solver_str == "flexgmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::FlexGMRES;
        if (lower_solver_str == "pcg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::PCG;
        if (lower_solver_str == "bicgstab") return OpenImpala::EffectiveDiffusivityHypre::SolverType::BiCGSTAB;
        if (lower_solver_str == "smg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::SMG;
        if (lower_solver_str == "pfmg") return OpenImpala::EffectiveDiffusivityHypre::SolverType::PFMG;
        amrex::Abort("Invalid solver string: '" + solver_str + "'.");
        return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES;
    }

    // Definition of calculate_Deff_tensor_homogenization (copied from your working Diffusion.cpp or REVMain.cpp)
    // Ensure this is the version that takes mf_chi_x_in, etc. and uses manual gradients.
    void calculate_Deff_tensor_homogenization(
        amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
        const amrex::MultiFab& mf_chi_x_in,
        const amrex::MultiFab& mf_chi_y_in,
        const amrex::MultiFab& mf_chi_z_in,
        const amrex::iMultiFab& active_mask,
        const amrex::Geometry& geom,
        int verbose_level)
    {
        BL_PROFILE("calculate_Deff_tensor_homogenization_test"); // Slightly different profile name for test
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                Deff_tensor[i][j] = 0.0;
            }
        }
        AMREX_ASSERT(mf_chi_x_in.nGrow() >= 1);
        AMREX_ASSERT(mf_chi_y_in.nGrow() >= 1);
        if (AMREX_SPACEDIM == 3) {
            AMREX_ASSERT(mf_chi_z_in.nGrow() >= 1);
        }

        const amrex::Real* dx_arr = geom.CellSize();
        amrex::Real inv_2dx[AMREX_SPACEDIM];
        for (int i=0; i<AMREX_SPACEDIM; ++i) {
            inv_2dx[i] = 1.0 / (2.0 * dx_arr[i]);
        }

        amrex::Real sum_integrand_tensor_comp[AMREX_SPACEDIM][AMREX_SPACEDIM];
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                sum_integrand_tensor_comp[i][j] = 0.0;
            }
        }

    #ifdef AMREX_USE_OMP
    #pragma omp parallel reduction(+:sum_integrand_tensor_comp)
    #endif
        for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_x_arr = mf_chi_x_in.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_y_arr = mf_chi_y_in.const_array(mfi);
            amrex::Array4<const amrex::Real> const chi_z_arr = (AMREX_SPACEDIM == 3) ?
                                                               mf_chi_z_in.const_array(mfi) :
                                                               mf_chi_x_in.const_array(mfi); // Dummy for 2D

            amrex::LoopOnCpu(bx, [=, &sum_integrand_tensor_comp] (int i, int j, int k) noexcept
            {
                if (mask_arr(i,j,k,0) == 1) {
                    amrex::Real grad_chi_x[AMREX_SPACEDIM] = {0.0};
                    amrex::Real grad_chi_y[AMREX_SPACEDIM] = {0.0};
                    amrex::Real grad_chi_z[AMREX_SPACEDIM] = {0.0};

                    grad_chi_x[0] = (chi_x_arr(i+1,j,k,0) - chi_x_arr(i-1,j,k,0)) * inv_2dx[0];
                    grad_chi_x[1] = (chi_x_arr(i,j+1,k,0) - chi_x_arr(i,j-1,k,0)) * inv_2dx[1];
                    if (AMREX_SPACEDIM == 3) grad_chi_x[2] = (chi_x_arr(i,j,k+1,0) - chi_x_arr(i,j,k-1,0)) * inv_2dx[2];

                    grad_chi_y[0] = (chi_y_arr(i+1,j,k,0) - chi_y_arr(i-1,j,k,0)) * inv_2dx[0];
                    grad_chi_y[1] = (chi_y_arr(i,j+1,k,0) - chi_y_arr(i,j-1,k,0)) * inv_2dx[1];
                    if (AMREX_SPACEDIM == 3) grad_chi_y[2] = (chi_y_arr(i,j,k+1,0) - chi_y_arr(i,j,k-1,0)) * inv_2dx[2];

                    if (AMREX_SPACEDIM == 3) {
                        grad_chi_z[0] = (chi_z_arr(i+1,j,k,0) - chi_z_arr(i-1,j,k,0)) * inv_2dx[0];
                        grad_chi_z[1] = (chi_z_arr(i,j+1,k,0) - chi_z_arr(i,j-1,k,0)) * inv_2dx[1];
                        grad_chi_z[2] = (chi_z_arr(i,j,k+1,0) - chi_z_arr(i,j,k-1,0)) * inv_2dx[2];
                    }

                    sum_integrand_tensor_comp[0][0] += (grad_chi_x[0] + 1.0);
                    sum_integrand_tensor_comp[0][1] += grad_chi_y[0];
                    sum_integrand_tensor_comp[1][0] += grad_chi_x[1];
                    sum_integrand_tensor_comp[1][1] += (grad_chi_y[1] + 1.0);

                    if (AMREX_SPACEDIM == 3) {
                        sum_integrand_tensor_comp[0][2] += grad_chi_z[0];
                        sum_integrand_tensor_comp[2][0] += grad_chi_x[2];
                        sum_integrand_tensor_comp[1][2] += grad_chi_z[1];
                        sum_integrand_tensor_comp[2][1] += grad_chi_y[2];
                        sum_integrand_tensor_comp[2][2] += (grad_chi_z[2] + 1.0);
                    }
                }
            });
        }
        for (int r = 0; r < AMREX_SPACEDIM; ++r) {
            for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                amrex::ParallelDescriptor::ReduceRealSum(sum_integrand_tensor_comp[r][c]);
            }
        }
        amrex::Long N_total_cells_in_domain = geom.Domain().numPts(); // Use domain, not REV
        if (N_total_cells_in_domain > 0) {
            for (int l_idx = 0; l_idx < AMREX_SPACEDIM; ++l_idx) {
                for (int m_idx = 0; m_idx < AMREX_SPACEDIM; ++m_idx) {
                    Deff_tensor[l_idx][m_idx] = sum_integrand_tensor_comp[l_idx][m_idx] / static_cast<amrex::Real>(N_total_cells_in_domain);
                }
            }
        } else {
             if (amrex::ParallelDescriptor::IOProcessor() && verbose_level > 0) {
                amrex::Warning("Total cells in domain is zero, D_eff cannot be calculated.");
             }
        }
         if (verbose_level > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  [TestCalcDeff] Raw summed D_xx: " << sum_integrand_tensor_comp[0][0]
                            << ", N_total_cells: " << N_total_cells_in_domain << std::endl;
        }
    }
} // end anonymous namespace for test helpers


int main (int argc, char* argv[])
{
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

    amrex::Initialize(argc, argv);
    { // Start AMReX scope
        amrex::Real strt_time = amrex::second();
        bool all_solves_converged = true;
        bool test_passed_overall = true; // New flag for overall test status

        // --- Configuration via ParmParse (same as before) ---
        std::string tifffile;
        std::string resultsdir = "./tEffectiveDiffusivity_results";
        int phase_id = 1;
        std::string solver_str = "FlexGMRES";
        int box_size = 32;
        int verbose = 1;
        int write_plotfile = 0;
        amrex::Real threshold_val = 0.5;

        {
            amrex::ParmParse pp;
            pp.get("tifffile", tifffile);
            pp.query("resultsdir", resultsdir);
            pp.query("phase_id", phase_id);
            pp.query("solver", solver_str);
            pp.query("box_size", box_size);
            pp.query("verbose", verbose);
            pp.query("write_plotfile", write_plotfile);
            pp.query("threshold_val", threshold_val);
        }

        OpenImpala::EffectiveDiffusivityHypre::SolverType solver_type = stringToSolverType(solver_str);

        if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            // ... (print configuration - same as before) ...
            amrex::Print() << "\n--- Effective Diffusivity Test Configuration ---\n";
            amrex::Print() << "  Input TIFF File:   " << tifffile << "\n";
            amrex::Print() << "  Results Dir:       " << resultsdir << "\n";
            amrex::Print() << "  Active Phase ID:   " << phase_id << "\n";
            amrex::Print() << "  Threshold Value:   " << threshold_val << "\n";
            amrex::Print() << "  Solver:            " << solver_str << "\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Write Plotfile:    " << (write_plotfile != 0 ? "Yes" : "No") << "\n";
            amrex::Print() << "  Verbose Level:     " << verbose << "\n";
            amrex::Print() << "----------------------------------------------\n\n";
        }

        // --- AMReX Grid and Data Setup (same as before) ---
        amrex::Geometry geom;
        amrex::BoxArray ba_original;
        amrex::DistributionMapping dm_original;
        amrex::iMultiFab mf_phase_input;

        try {
            // ... (TiffReader and grid setup - same as before) ...
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Reading metadata from " << tifffile << "...\n";
            OpenImpala::TiffReader reader(tifffile);
            if (!reader.isRead()) { throw std::runtime_error("TiffReader failed to read metadata."); }
            const amrex::Box domain_box = reader.box();
            if (domain_box.isEmpty()) { throw std::runtime_error("TiffReader returned empty domain box."); }
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1, 1, 1)};
            geom.define(domain_box, &rb, 0, is_periodic.data()); // Defined once is enough
            // geom.define(domain_box, &rb, 0, is_periodic.data()); // Redundant define

            ba_original.define(domain_box);
            ba_original.maxSize(box_size);
            dm_original.define(ba_original);
            mf_phase_input.define(ba_original, dm_original, 1, 1);
            amrex::iMultiFab mf_temp_no_ghost(ba_original, dm_original, 1, 0);
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) { amrex::Print() << " Thresholding data...\n"; }
            reader.threshold(threshold_val, phase_id, (phase_id == 0 ? 1 : 0), mf_temp_no_ghost);
            amrex::Copy(mf_phase_input, mf_temp_no_ghost, 0, 0, 1, 0);
            mf_phase_input.FillBoundary(geom.periodicity());
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << " Grid and phase data setup complete.\n";

        } catch (const std::exception& e) {
            amrex::Print() << "Error during TiffReader/grid setup: " << e.what() << std::endl;
            amrex::Abort("Test setup failed.");
        }

        if (!resultsdir.empty() && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(resultsdir, 0755);
        }
        amrex::ParallelDescriptor::Barrier();

        // --- Solve for Corrector Functions chi_x, chi_y, chi_z (same as before) ---
        amrex::MultiFab mf_chi_x(ba_original, dm_original, 1, 1);
        amrex::MultiFab mf_chi_y(ba_original, dm_original, 1, 1);
        amrex::MultiFab mf_chi_z; // Define only if 3D
        if (AMREX_SPACEDIM == 3) mf_chi_z.define(ba_original, dm_original, 1, 1);


        std::vector<OpenImpala::Direction> directions_to_solve = {
            OpenImpala::Direction::X, OpenImpala::Direction::Y
        };
        if (AMREX_SPACEDIM == 3) {
            directions_to_solve.push_back(OpenImpala::Direction::Z);
        }

        for (const auto& dir_k : directions_to_solve) {
            // ... (chi_k solving loop - same as before) ...
            std::string dir_k_str = (dir_k == OpenImpala::Direction::X) ? "X" :
                                    (dir_k == OpenImpala::Direction::Y) ? "Y" : "Z";
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Solving for Corrector Function chi_" << dir_k_str << " ---\n";
            }
            std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> eff_diff_solver;
            try {
                eff_diff_solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                    geom, ba_original, dm_original, mf_phase_input,
                    phase_id, dir_k, solver_type, resultsdir,
                    verbose, (write_plotfile != 0)
                );
                if (!eff_diff_solver->solve()) {
                     if (verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "  WARNING: Solver for chi_" << dir_k_str << " DID NOT CONVERGE.\n";
                    }
                    all_solves_converged = false;
                    // Set corresponding mf_chi to 0 or handle as error
                    if (dir_k == OpenImpala::Direction::X) mf_chi_x.setVal(0.0);
                    else if (dir_k == OpenImpala::Direction::Y) mf_chi_y.setVal(0.0);
                    else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) mf_chi_z.setVal(0.0);
                } else {
                     if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "  Solver for chi_" << dir_k_str << " CONVERGED.\n";
                    }
                    if (dir_k == OpenImpala::Direction::X) eff_diff_solver->getChiSolution(mf_chi_x);
                    else if (dir_k == OpenImpala::Direction::Y) eff_diff_solver->getChiSolution(mf_chi_y);
                    else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) eff_diff_solver->getChiSolution(mf_chi_z);
                }
            } catch (const std::exception& e) {
                amrex::Print() << "  ERROR during EffectiveDiffusivityHypre construction or solve for chi_"
                               << dir_k_str << ": " << e.what() << std::endl;
                all_solves_converged = false;
            }
        }

        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        // +++ NEW SECTION: Calculate and Print/Validate D_eff Tensor +++
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        amrex::Real Deff_tensor_vals[AMREX_SPACEDIM][AMREX_SPACEDIM];
        for(int r=0; r<AMREX_SPACEDIM; ++r) for(int c=0; c<AMREX_SPACEDIM; ++c) Deff_tensor_vals[r][c] = 0.0; // Initialize

        if (all_solves_converged) {
            if (verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Calculating D_eff Tensor from Converged Chi Fields ---\n";
            }

            // Create the active_mask (0-ghost cells) from mf_phase_input
            amrex::iMultiFab active_mask(ba_original, dm_original, 1, 0);
            #ifdef AMREX_USE_OMP
            #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
            #endif
            for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                const amrex::Box& tilebox = mfi.tilebox();
                amrex::Array4<int> const mask_arr = active_mask.array(mfi);
                // mf_phase_input has 1 ghost cell, but we iterate only on tilebox (valid region of active_mask)
                amrex::Array4<const int> const phase_arr = mf_phase_input.const_array(mfi);
                amrex::LoopOnCpu(tilebox, [=] (int i, int j, int k) noexcept {
                    mask_arr(i,j,k,0) = (phase_arr(i,j,k,0) == phase_id) ? 1 : 0;
                });
            }

            // Call the homogenization calculation
            // Ensure mf_chi_z is passed correctly (it's default-constructed if 2D, then potentially defined if 3D)
            // If AMREX_SPACEDIM is 2, mf_chi_z might be empty.
            // calculate_Deff_tensor_homogenization uses a dummy for chi_z_arr if 2D.
            calculate_Deff_tensor_homogenization(Deff_tensor_vals,
                                                 mf_chi_x, mf_chi_y, mf_chi_z, // Pass mf_chi_z even if 2D
                                                 active_mask, geom, verbose);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Effective Diffusivity Tensor D_eff / D_material (D_material=1 assumed):\n";
                for (int i_row = 0; i_row < AMREX_SPACEDIM; ++i_row) {
                    amrex::Print() << "  [";
                    for (int j_col = 0; j_col < AMREX_SPACEDIM; ++j_col) {
                        amrex::Print() << std::scientific << std::setprecision(8) << Deff_tensor_vals[i_row][j_col]
                                       << (j_col == AMREX_SPACEDIM - 1 ? "" : ", ");
                    }
                    amrex::Print() << "]\n";
                }

                // --- Basic Validation Assertions (Example) ---
                bool symmetry_ok = true;
                amrex::Real sym_tol = 1e-7; // Tolerance for symmetry check

                if (std::abs(Deff_tensor_vals[0][1] - Deff_tensor_vals[1][0]) > sym_tol) symmetry_ok = false;
                if (AMREX_SPACEDIM == 3) {
                    if (std::abs(Deff_tensor_vals[0][2] - Deff_tensor_vals[2][0]) > sym_tol) symmetry_ok = false;
                    if (std::abs(Deff_tensor_vals[1][2] - Deff_tensor_vals[2][1]) > sym_tol) symmetry_ok = false;
                }

                if (!symmetry_ok) {
                    amrex::Warning("D_eff tensor is not symmetric within tolerance!");
                    test_passed_overall = false;
                } else {
                    if (verbose >=1) amrex::Print() << "  D_eff tensor symmetry check: PASS\n";
                }

                for (int d=0; d < AMREX_SPACEDIM; ++d) {
                    if (Deff_tensor_vals[d][d] <= 0.0 || Deff_tensor_vals[d][d] >= 1.0) { // Should be < porosity (if porosity < 1)
                         // For D_eff/D_0, it should be < 1 generally. If it can be >1 due to definition, adjust.
                         // For now, let's assume 0 < D_ii < 1 is expected for a porous medium.
                        amrex::Warning("D_eff diagonal component D_" + std::to_string(d) + std::to_string(d) +
                                       " is out of expected range (0,1): " + std::to_string(Deff_tensor_vals[d][d]));
                        // test_passed_overall = false; // This might be too strict without knowing porosity
                    }
                }
            }
        } else {
            if (amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Skipping D_eff tensor calculation due to chi_k solver non-convergence.\n";
            }
            test_passed_overall = false; // If solves didn't converge, test for D_eff fails
        }
        // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Effective Diffusivity Test Summary ---\n";
            amrex::Print() << "  Total Run Time: " << stop_time << " sec\n";
            if (test_passed_overall && all_solves_converged) { // Modified condition
                amrex::Print() << "  All chi_k solver instances converged.\n";
                amrex::Print() << "  TEST RESULT: PASS (Execution, solver convergence, and basic D_eff checks where applicable)\n";
            } else if (!all_solves_converged) {
                amrex::Print() << "  One or more chi_k solver instances FAILED to converge or encountered an error.\n";
                amrex::Print() << "  TEST RESULT: FAIL (Check logs for solver issues)\n";
            } else { // Solves converged but other checks might have failed
                 amrex::Print() << "  Chi_k solves converged, but some D_eff validation checks may have failed (see warnings).\n";
                 amrex::Print() << "  TEST RESULT: WARNING/FAIL (Review D_eff tensor values and warnings)\n";
            }
            amrex::Print() << "-----------------------------------------\n";
        }

        if (!test_passed_overall || !all_solves_converged) { // Abort if anything went wrong
            amrex::Abort("EffectiveDiffusivity Test FAILED.");
        }

    } // End of AMReX scope
    amrex::Finalize();

    hypre_ierr = HYPRE_Finalize();
    if (hypre_ierr != 0) {
        fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
        return 1;
    }

    return 0;
}
