// Main application driver for diffusion/tortuosity calculations.
// Reads various image formats (TIFF, DAT, HDF5), calculates volume fraction,
// and computes effective diffusivity/tortuosity using either flow-through
// or homogenization methods.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

// AMReX includes
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H>
#include <AMReX_MultiFabUtil.H> // For amrex::Average and potentially amrex::ComputeGradient

// OpenImpala includes
#include "../io/DatReader.H"
#include "../io/HDF5Reader.H"
#include "../io/TiffReader.H"
#include "TortuosityHypre.H"
#include "EffectiveDiffusivityHypre.H" // <<< NEW INCLUDE
#include "VolumeFraction.H"
#include "Tortuosity.H" // For OpenImpala::Direction

// HYPRE for explicit Init/Finalize (good practice, though AMReX often handles it)
#include <HYPRE.h>
#include <mpi.h>


// Anonymous namespace for local helpers
namespace
{
    OpenImpala::Direction string_to_direction(const std::string& s) {
        std::string upper_s = s;
        std::transform(upper_s.begin(), upper_s.end(), upper_s.begin(),
                       [](unsigned char c){ return std::toupper(c); });
        if (upper_s == "X") { return OpenImpala::Direction::X; }
        if (upper_s == "Y") { return OpenImpala::Direction::Y; }
        if (upper_s == "Z") { return OpenImpala::Direction::Z; }
        amrex::Warning("Invalid direction string '" + s + "' for tortuosity/flow-through.");
        return OpenImpala::Direction::X;
    }

    std::string direction_to_string_upper(OpenImpala::Direction dir) {
        switch (dir) {
            case OpenImpala::Direction::X: return "X";
            case OpenImpala::Direction::Y: return "Y";
            case OpenImpala::Direction::Z: return "Z";
            default: return "Unknown";
        }
    }

    // Assuming TortuosityHypre::SolverType and EffectiveDiffusivityHypre::SolverType are compatible
    // If not, you'll need separate helpers or ensure the enum definitions are identical.
    OpenImpala::TortuosityHypre::SolverType string_to_solver_type(const std::string& s) {
        std::string upper_s = s;
        std::transform(upper_s.begin(), upper_s.end(), upper_s.begin(), ::toupper);
        if (upper_s == "JACOBI")    { return OpenImpala::TortuosityHypre::SolverType::Jacobi; }
        if (upper_s == "GMRES")     { return OpenImpala::TortuosityHypre::SolverType::GMRES; }
        if (upper_s == "FLEXGMRES") { return OpenImpala::TortuosityHypre::SolverType::FlexGMRES; }
        if (upper_s == "PCG")       { return OpenImpala::TortuosityHypre::SolverType::PCG; }
        if (upper_s == "BICGSTAB")  { return OpenImpala::TortuosityHypre::SolverType::BiCGSTAB; }
        if (upper_s == "SMG")       { return OpenImpala::TortuosityHypre::SolverType::SMG; }
        if (upper_s == "PFMG")      { return OpenImpala::TortuosityHypre::SolverType::PFMG; }
        amrex::Warning("Invalid solver type string '" + s + "', defaulting to FlexGMRES.");
        return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    }


    // Function to calculate D_eff tensor for homogenization method
    // Ideally, this would be in EffectiveDiffusivityUtils.cpp/H
    void calculate_Deff_tensor_homogenization(
        amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
        const amrex::MultiFab& mf_chi_x,
        const amrex::MultiFab& mf_chi_y,
        const amrex::MultiFab& mf_chi_z, // Will be empty/ignored in 2D
        const amrex::iMultiFab& active_mask, // Mask where D_material = 1
        const amrex::Geometry& geom,
        int verbose_level)
    {
        BL_PROFILE("calculate_Deff_tensor_homogenization");
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                Deff_tensor[i][j] = 0.0;
            }
        }

        const amrex::Real* dx_arr = geom.CellSize(); 

        // Temporary MultiFabs for gradients (each with AMREX_SPACEDIM components)
        // Gradients should be cell-centered. Chi fields have 1 ghost cell.
        // Ensure chi fields have up-to-date periodic ghost cells. This should be done by getChiSolution.
        amrex::MultiFab grad_chi_x(mf_chi_x.boxArray(), mf_chi_x.DistributionMap(), AMREX_SPACEDIM, 0);
        amrex::MultiFab grad_chi_y(mf_chi_y.boxArray(), mf_chi_y.DistributionMap(), AMREX_SPACEDIM, 0);
        amrex::MultiFab grad_chi_z;
        if (AMREX_SPACEDIM == 3) {
            grad_chi_z.define(mf_chi_z.boxArray(), mf_chi_z.DistributionMap(), AMREX_SPACEDIM, 0);
        }

        // Compute gradients for each chi field.
// We want all gradient components (d/dx, d/dy, d/dz) for each chi_k (which has 1 component).
// The output grad_chi_k will have AMREX_SPACEDIM components.
// Signature: computeGradient(MultiFab& grad, const MultiFab& S, int S_comp, int grad_comp, int ncomp_S, const Geometry& geom)
// Here ncomp_S refers to the number of components in S (mf_chi_x) for which we are computing gradients.
// Since mf_chi_x has 1 component, ncomp_S should be 1.
// The output grad_chi_x will have AMREX_SPACEDIM components for this single component of S.

const int S_comp = 0;       // Starting component of the source MultiFab (mf_chi_x, etc.)
const int grad_comp = 0;    // Starting component of the destination gradient MultiFab (grad_chi_x, etc.)
const int ncomp_S = 1;      // Number of components in the source MultiFab (mf_chi_x) to process.
                            // computeGradient will produce AMREX_SPACEDIM gradient components for each of these ncomp_S.

amrex::computeGradient(grad_chi_x, mf_chi_x, S_comp, grad_comp, ncomp_S, geom);
amrex::computeGradient(grad_chi_y, mf_chi_y, S_comp, grad_comp, ncomp_S, geom);
if (AMREX_SPACEDIM == 3) {
    amrex::computeGradient(grad_chi_z, mf_chi_z, S_comp, grad_comp, ncomp_S, geom);
}

        // Accumulators for sum ( Integrand_Tensor_Component_lm ) over PORE cells
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
            const amrex::Box& bx = mfi.validbox(); // Sum only over valid cells
            amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);

            amrex::Array4<const amrex::Real> const gcx_arr = grad_chi_x.const_array(mfi);
            amrex::Array4<const amrex::Real> const gcy_arr = grad_chi_y.const_array(mfi);
            amrex::Array4<const amrex::Real> const gcz_arr = (AMREX_SPACEDIM == 3) ? grad_chi_z.const_array(mfi) : grad_chi_x.const_array(mfi); // Dummy if 2D

            amrex::LoopOnCpu(bx, [=, &sum_integrand_tensor_comp] (int i, int j, int k) noexcept // New
            {
                if (mask_arr(i,j,k,0) == 1) { // If D_material = 1 in this cell (pore)
                    // Integrand for D_eff_lm is ( (d(chi_x)/dl * (e_x)_m) + ... + delta_lm )
                    // where l is row index (0=x, 1=y, 2=z), m is col index

                    // D_eff_xx: (d(chi_x)/dx + 1)
                    sum_integrand_tensor_comp[0][0] += (gcx_arr(i,j,k,0) + 1.0);
                    // D_eff_xy: (d(chi_y)/dx)
                    sum_integrand_tensor_comp[0][1] += gcy_arr(i,j,k,0);
                    // D_eff_yx: (d(chi_x)/dy)
                    sum_integrand_tensor_comp[1][0] += gcx_arr(i,j,k,1);
                    // D_eff_yy: (d(chi_y)/dy + 1)
                    sum_integrand_tensor_comp[1][1] += (gcy_arr(i,j,k,1) + 1.0);

                    if (AMREX_SPACEDIM == 3) {
                        // D_eff_xz: (d(chi_z)/dx)
                        sum_integrand_tensor_comp[0][2] += gcz_arr(i,j,k,0);
                        // D_eff_zx: (d(chi_x)/dz)
                        sum_integrand_tensor_comp[2][0] += gcx_arr(i,j,k,2);
                        // D_eff_yz: (d(chi_z)/dy)
                        sum_integrand_tensor_comp[1][2] += gcz_arr(i,j,k,1);
                        // D_eff_zy: (d(chi_y)/dz)
                        sum_integrand_tensor_comp[2][1] += gcy_arr(i,j,k,2);
                        // D_eff_zz: (d(chi_z)/dz + 1)
                        sum_integrand_tensor_comp[2][2] += (gcz_arr(i,j,k,2) + 1.0);
                    }
                }
            });
        }

        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                amrex::ParallelDescriptor::ReduceRealSum(sum_integrand_tensor_comp[i][j]);
            }
        }

        amrex::Long N_total_cells_in_REV = geom.Domain().numPts();
        if (N_total_cells_in_REV > 0) {
            for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                    Deff_tensor[i][j] = sum_integrand_tensor_comp[i][j] / static_cast<amrex::Real>(N_total_cells_in_REV);
                }
            }
        } else {
             if (amrex::ParallelDescriptor::IOProcessor() && verbose_level > 0) {
                amrex::Warning("Total cells in REV is zero, D_eff cannot be calculated.");
             }
        }
        if (verbose_level > 1 && amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Raw summed integrand (D_xx example): " << sum_integrand_tensor_comp[0][0] << std::endl;
             amrex::Print() << "  N_total_cells_in_REV: " << N_total_cells_in_REV << std::endl;
        }
    }

} // End anonymous namespace


int main(int argc, char* argv[])
{
    // It's good practice to init HYPRE explicitly if direct HYPRE calls are made
    // before AMReX might have a chance to do so, or if there are multiple solver instances.
    HYPRE_Init(); // Ensure HYPRE is initialized

    amrex::Initialize(argc, argv);
    { // Start AMReX scope
        amrex::Real strt_time = amrex::second();

        amrex::ParmParse pp;

        std::string filename;
        pp.get("filename", filename);

        std::string data_path_str = "."; pp.query("data_path", data_path_str);
        std::string hdf5_dataset;      pp.query("hdf5_dataset", hdf5_dataset);
        int raw_width = 0, raw_height = 0, raw_depth = 0;
        std::string raw_datatype_str;
        pp.query("raw_width", raw_width); pp.query("raw_height", raw_height);
        pp.query("raw_depth", raw_depth); pp.query("raw_datatype", raw_datatype_str);

        int phase_id = 1; pp.query("phase_id", phase_id);
        double threshold_value = 127.5; pp.query("threshold_value", threshold_value);

        // NEW: Calculation method parameter
        std::string calculation_method_str = "flow_through"; // Default
        pp.query("calculation_method", calculation_method_str);
        bool use_homogenization = false;
        if (calculation_method_str == "homogenization") {
            use_homogenization = true;
        } else if (calculation_method_str != "flow_through") {
            amrex::Warning("Unknown calculation_method '" + calculation_method_str + "'. Defaulting to 'flow_through'.");
        }

        std::string direction_str = "All"; pp.query("direction", direction_str); // Used by flow-through
        std::string solver_type_str = "FlexGMRES"; pp.query("solver_type", solver_type_str);
        // hypre.eps and hypre.maxiter are read by HYPRE solver classes via ParmParse("hypre")

        int box_size = 32; pp.query("box_size", box_size);
        std::string results_dir_str = "DiffusionResults"; pp.query("results_dir", results_dir_str);
        std::string output_filename = "diffusion_results.txt"; pp.query("output_filename", output_filename);
        int write_plotfile = 0; pp.query("write_plotfile", write_plotfile);
        int verbose = 1; pp.query("verbose", verbose);

        std::filesystem::path data_path = data_path_str;
        std::filesystem::path results_dir_fs = results_dir_str; // Use different name to avoid conflict
        // ... (Path handling for '~' as before) ...
        if (!results_dir_fs.empty() && results_dir_fs.string().front() == '~') { /* ... */ }
        if (!data_path.empty() && data_path.string().front() == '~') { /* ... */ }
        std::filesystem::path full_input_path = data_path / filename;
        if (amrex::ParallelDescriptor::IOProcessor()) { /* ... Create results_dir_fs ... */ }
        amrex::ParallelDescriptor::Barrier();

        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- OpenImpala Calculation ---\n";
            amrex::Print() << " Calculation Method:   " << calculation_method_str << "\n";
            amrex::Print() << " Input File:           " << full_input_path << "\n";
            // ... (other parameter prints) ...
            amrex::Print() << "-----------------------------------\n\n";
        }

        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase_input; // Used by both methods
        amrex::Box domain_box;

        try {
            // ... (File reading logic as before to populate domain_box, then ba, dm) ...
            // This part is identical to your existing Diffusion.cpp up to mf_phase definition
            std::string ext;
            if (full_input_path.has_extension()) { ext = full_input_path.extension().string(); std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower); }

            if (ext == ".tif" || ext == ".tiff") { OpenImpala::TiffReader reader(full_input_path.string()); domain_box = reader.box(); }
            // ... (add other readers: DAT, RAW, HDF5 to set domain_box) ...
            else { amrex::Abort("File format not recognized: " + filename); }

            if (domain_box.isEmpty()) throw std::runtime_error("Reader returned empty domain box.");
            ba.define(domain_box);
            ba.maxSize(box_size);
            dm.define(ba);

            // mf_phase_input needs 1 ghost cell for both solver types
            mf_phase_input.define(ba, dm, 1, 1);
            amrex::iMultiFab mf_temp_no_ghost(ba, dm, 1, 0);

            if (ext == ".tif" || ext == ".tiff") {
                 OpenImpala::TiffReader reader(full_input_path.string());
                 reader.threshold(threshold_value, phase_id /*val_gt*/, (phase_id == 0 ? 1 : 0) /*val_le*/, mf_temp_no_ghost);
            }
            // ... (add other readers to fill mf_temp_no_ghost) ...
            amrex::Copy(mf_phase_input, mf_temp_no_ghost, 0, 0, 1, 0);


            // --- Conditional Geometry Setup ---
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> amrex_periodicity;
            if (use_homogenization) {
                amrex_periodicity = {AMREX_D_DECL(1, 1, 1)}; // Fully periodic
                if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << " Method: Homogenization. AMReX Geometry: PERIODIC.\n";
            } else {
                amrex_periodicity = {AMREX_D_DECL(0, 0, 0)}; // Non-periodic
                if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << " Method: Flow-through. AMReX Geometry: NON-PERIODIC.\n";
            }
            geom.define(domain_box, &rb, 0, amrex_periodicity.data());

            // Fill ghost cells of mf_phase_input according to the determined periodicity
            if (mf_phase_input.nGrow() > 0) {
                mf_phase_input.FillBoundary(geom.periodicity());
            }

        } catch (const std::exception& e) {
            amrex::Abort("Error during file reading or grid setup: " + std::string(e.what()));
        }

        // --- Conditional Solver Execution ---
        if (use_homogenization) {
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << "\n--- Effective Diffusivity via Homogenization ---\n";

            // Optional: Warning for non-power-of-two with periodic (though PFMG handled 100^3)
            bool is_any_periodic_non_pow2 = false;
            if (geom.isAnyPeriodic()) {
                for (int d=0; d<AMREX_SPACEDIM; ++d) {
                    if (geom.isPeriodic(d)) {
                        int len = geom.Domain().length(d);
                        if (len > 0 && (len & (len-1)) != 0) { is_any_periodic_non_pow2 = true; break;}
                    }
                }
            }
            if (is_any_periodic_non_pow2 && verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << " WARNING: Homogenization on periodic domain with non-power-of-two dimensions ("
                                << geom.Domain().length() << "). Ensure HYPRE solver (e.g. FlexGMRES+PFMG) is tolerant.\n";
            }

            amrex::MultiFab mf_chi_x(ba, dm, 1, 1);
            amrex::MultiFab mf_chi_y(ba, dm, 1, 1);
            amrex::MultiFab mf_chi_z; if (AMREX_SPACEDIM == 3) mf_chi_z.define(ba, dm, 1, 1);
            bool all_chi_solves_converged = true;

            std::vector<OpenImpala::Direction> directions_to_solve_chi =
                {OpenImpala::Direction::X, OpenImpala::Direction::Y};
            if (AMREX_SPACEDIM == 3) directions_to_solve_chi.push_back(OpenImpala::Direction::Z);

            auto current_solver_type_effdiff = static_cast<OpenImpala::EffectiveDiffusivityHypre::SolverType>(string_to_solver_type(solver_type_str));


            for (const auto& dir_k : directions_to_solve_chi) {
                std::string dir_k_str = direction_to_string_upper(dir_k);
                if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "\n--- Solving for Corrector Function chi_" << dir_k_str << " ---\n";

                OpenImpala::EffectiveDiffusivityHypre solver_chi_k(
                    geom, ba, dm, mf_phase_input,
                    phase_id, dir_k, current_solver_type_effdiff, results_dir_fs.string(),
                    verbose, (write_plotfile != 0)
                );
                bool converged = solver_chi_k.solve();
                if (converged) {
                    if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                        amrex::Print() << "  Solver for chi_" << dir_k_str << " CONVERGED.\n";
                    if (dir_k == OpenImpala::Direction::X) solver_chi_k.getChiSolution(mf_chi_x);
                    else if (dir_k == OpenImpala::Direction::Y) solver_chi_k.getChiSolution(mf_chi_y);
                    else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) solver_chi_k.getChiSolution(mf_chi_z);
                } else {
                    all_chi_solves_converged = false;
                    if (amrex::ParallelDescriptor::IOProcessor())
                         amrex::Print() << "  WARNING: Solver for chi_" << dir_k_str << " DID NOT CONVERGE.\n";
                    // Set chi to zero if solve failed
                    if (dir_k == OpenImpala::Direction::X) mf_chi_x.setVal(0.0);
                    else if (dir_k == OpenImpala::Direction::Y) mf_chi_y.setVal(0.0);
                    else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) mf_chi_z.setVal(0.0);
                    // Fill boundaries for consistency
                    if (mf_chi_x.nGrow() > 0 && dir_k == OpenImpala::Direction::X) mf_chi_x.FillBoundary(geom.periodicity());
                    // ... similar for Y and Z
                }
            }

            if (all_chi_solves_converged) {
                if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "\n--- Calculating D_eff Tensor ---\n";

                amrex::Real Deff_tensor_vals[AMREX_SPACEDIM][AMREX_SPACEDIM];
                // Re-create active_mask (0-grow for sum) for D_eff calc if not reusing from solver
                amrex::iMultiFab active_mask_for_Deff(ba, dm, 1, 0);
                #ifdef AMREX_USE_OMP
                #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
                #endif
                for (amrex::MFIter mfi(active_mask_for_Deff); mfi.isValid(); ++mfi) {
                    const amrex::Box& bx = mfi.tilebox();
                    amrex::Array4<int> const mask_arr = active_mask_for_Deff.array(mfi);
                    amrex::Array4<const int> const phase_arr = mf_phase_input.const_array(mfi); // Original phase
                    amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept {
                        mask_arr(i,j,k,0) = (phase_arr(i,j,k,0) == phase_id) ? 1 : 0;
                    });
                }

                calculate_Deff_tensor_homogenization(
                    Deff_tensor_vals, mf_chi_x, mf_chi_y, mf_chi_z,
                    active_mask_for_Deff, geom, verbose);

                if (amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << "Effective Diffusivity Tensor D_eff / D_material (D_material=1 assumed):\n";
                    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                        amrex::Print() << "  [";
                        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                            amrex::Print() << std::scientific << std::setprecision(8) << Deff_tensor_vals[i][j]
                                           << (j == AMREX_SPACEDIM - 1 ? "" : ", ");
                        }
                        amrex::Print() << "]\n";
                    }
                    // Write to file
                    std::filesystem::path output_filepath = results_dir_fs / output_filename;
                    std::ofstream outfile(output_filepath);
                    if (outfile.is_open()) {
                        outfile << "# Effective Diffusivity Calculation Results (Homogenization)\n";
                        outfile << "# Input File: " << full_input_path.string() << "\n";
                        outfile << "# Analysis Phase ID: " << phase_id << "\n";
                        outfile << "# Threshold Value: " << threshold_value << "\n";
                        outfile << "# Solver: " << solver_type_str << "\n";
                        outfile << "# -----------------------------\n";
                        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                            for (int j = 0; j < AMREX_SPACEDIM; ++j) {
                                outfile << "Deff_" << direction_to_string_upper(static_cast<OpenImpala::Direction>(i))
                                        << direction_to_string_upper(static_cast<OpenImpala::Direction>(j))
                                        << ": " << std::scientific << std::setprecision(9) << Deff_tensor_vals[i][j] << "\n";
                            }
                        }
                        outfile.close();
                    } else { amrex::Warning("Could not open output file: " + output_filepath.string()); }
                }
            } else {
                if (amrex::ParallelDescriptor::IOProcessor())
                     amrex::Print() << "Skipping D_eff calculation due to chi_k solver non-convergence.\n";
            }

        } else { // Flow-through method
            if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << "\n--- Tortuosity/Effective Diffusivity via Flow-Through ---\n";

            OpenImpala::VolumeFraction vf_calc(mf_phase_input, phase_id, 0);
            amrex::Real actual_vf = vf_calc.value_vf(false);
            if (amrex::ParallelDescriptor::IOProcessor() && verbose > 0) {
                amrex::Print() << "  Volume Fraction (Phase " << phase_id << "): "
                               << std::fixed << std::setprecision(6) << actual_vf << "\n";
            }

            std::vector<OpenImpala::Direction> directions_to_compute_tort;
            std::string upper_direction_str_tort = direction_str;
            std::transform(upper_direction_str_tort.begin(), upper_direction_str_tort.end(), upper_direction_str_tort.begin(), ::toupper);
            if (upper_direction_str_tort == "ALL") {
                directions_to_compute_tort = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
                if (AMREX_SPACEDIM ==3) directions_to_compute_tort.push_back(OpenImpala::Direction::Z);
            } else { /* ... parse single direction ... */ directions_to_compute_tort.push_back(string_to_direction(direction_str));}
            // ... (Remove duplicates if necessary) ...

            std::map<OpenImpala::Direction, amrex::Real> tortuosity_map_results;
            amrex::Real v_lo = 0.0; pp.query("v_lo", v_lo); // Specific to flow-through
            amrex::Real v_hi = 1.0; pp.query("v_hi", v_hi); // Specific to flow-through
            auto current_solver_type_tort = string_to_solver_type(solver_type_str);


            for (const auto& dir_tort : directions_to_compute_tort) {
                // ... (skip if actual_vf is zero) ...
                OpenImpala::TortuosityHypre tortuosity_solver(
                    geom, ba, dm, mf_phase_input, actual_vf, phase_id, dir_tort,
                    current_solver_type_tort, results_dir_fs.string(), v_lo, v_hi, verbose, (write_plotfile != 0)
                );
                tortuosity_map_results[dir_tort] = tortuosity_solver.value();
                // ... (print tortuosity) ...
            }
            // Write tortuosity_map_results to file
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::filesystem::path output_filepath = results_dir_fs / output_filename;
                std::ofstream outfile(output_filepath);
                 if (outfile.is_open()) {
                    outfile << "# Diffusion Calculation Results (Flow-Through)\n";
                    // ... (header info as before) ...
                    outfile << "VolumeFraction: " << std::fixed << std::setprecision(9) << actual_vf << "\n";
                    for (const auto& pair : tortuosity_map_results) {
                        outfile << "Tortuosity_" << direction_to_string_upper(pair.first) << ": "
                                << std::fixed << std::setprecision(9) << pair.second << "\n";
                    }
                    outfile.close();
                } else { amrex::Warning("Could not open output file: " + output_filepath.string()); }
            }
        }

        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber()); // New
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << std::endl << "Total run time (seconds) = " << stop_time << std::endl;
        }

    } // End AMReX scope
    amrex::Finalize();
    HYPRE_Finalize(); // Ensure HYPRE is finalized after AMReX
    return 0;
}
