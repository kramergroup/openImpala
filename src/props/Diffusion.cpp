/** REV calculation / Full Domain Diffusion
 *
 * This programme calculates the effective diffusivity tensor using homogenization,
 * either for the full domain or as part of a Representative Elementary Volume (REV) study.
 * For an REV study, it analyzes multiple random sub-volumes of increasing sizes.
 *
 * Outputs:
 *  - For full domain: D_eff tensor.
 *  - For REV study: CSV file with D_eff tensor components vs. REV size for each sample.
 */

#include "../io/TiffReader.H"
#include "../io/DatReader.H"
#include "../io/HDF5Reader.H"
// #include "../io/RawReader.H" // Uncomment if RawReader is used

#include "EffectiveDiffusivityHypre.H" // For homogenization
#include "TortuosityHypre.H"          // For flow-through (if still supported as an option)
#include "VolumeFraction.H"
#include "Tortuosity.H" // For OpenImpala::Direction

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H> // For UtilCreateDirectory
#include <AMReX_PlotFileUtil.H>

#include <sstream>
#include <iostream>
#include <fstream> // For CSV output
#include <string>
#include <vector>
#include <algorithm> // For std::min, std::max
#include <random>    // For REV study seeding
#include <iomanip>   // For std::fixed, std::setprecision

// Forward declaration if calculate_Deff_tensor_homogenization is in this file
// Or include a header where it's declared.
// For now, assuming it's accessible.
namespace { // Anonymous namespace for helpers in this file
    void calculate_Deff_tensor_homogenization(
        amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
        const amrex::MultiFab& mf_chi_x,
        const amrex::MultiFab& mf_chi_y,
        const amrex::MultiFab& mf_chi_z,
        const amrex::iMultiFab& active_mask,
        const amrex::Geometry& geom,
        int verbose_level);

    // Helper from your tEffectiveDiffusivity.cpp
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
        amrex::Abort("Invalid solver string: '" + solver_str + "'. Supported: Jacobi, GMRES, FlexGMRES, PCG, BiCGSTAB, SMG, PFMG");
        return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES; // Should not reach here
    }
} // end anonymous namespace

// Definition of calculate_Deff_tensor_homogenization (copied from your working Diffusion.cpp)
namespace {
void calculate_Deff_tensor_homogenization(
    amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
    const amrex::MultiFab& mf_chi_x_in,
    const amrex::MultiFab& mf_chi_y_in,
    const amrex::MultiFab& mf_chi_z_in,
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
                                                           mf_chi_x_in.const_array(mfi);

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
    for (int i_ = 0; i_ < AMREX_SPACEDIM; ++i_) { // Use different loop var name
        for (int j_ = 0; j_ < AMREX_SPACEDIM; ++j_) {
            amrex::ParallelDescriptor::ReduceRealSum(sum_integrand_tensor_comp[i_][j_]);
        }
    }
    amrex::Long N_total_cells_in_REV = geom.Domain().numPts();
    if (N_total_cells_in_REV > 0) {
        for (int l_idx = 0; l_idx < AMREX_SPACEDIM; ++l_idx) {
            for (int m_idx = 0; m_idx < AMREX_SPACEDIM; ++m_idx) {
                Deff_tensor[l_idx][m_idx] = sum_integrand_tensor_comp[l_idx][m_idx] / static_cast<amrex::Real>(N_total_cells_in_REV);
            }
        }
    } else {
         if (amrex::ParallelDescriptor::IOProcessor() && verbose_level > 0) {
            amrex::Warning("Total cells in REV is zero, D_eff cannot be calculated.");
         }
    }
    if (verbose_level > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  [calc_Deff] Raw summed D_xx: " << sum_integrand_tensor_comp[0][0]
                        << ", N_total_cells: " << N_total_cells_in_REV << std::endl;
    }
}
} // end anonymous namespace


int main (int argc, char* argv[])
{
    // Initialize HYPRE First
    int hypre_ierr = HYPRE_Init();
    if (hypre_ierr != 0) {
        fprintf(stderr, "FATAL ERROR: HYPRE_Init() failed with code %d\n", hypre_ierr);
        return 1;
    }

    amrex::Initialize(argc, argv);
    { // AMReX Scope
        amrex::Real master_strt_time = amrex::second();

        // --- Main Parameters ---
        std::string main_filename;
        std::string main_data_path = "./data/"; // Default
        std::string main_results_path = "./results/"; // Default
        std::string main_hdf5_dataset;
        amrex::Real main_threshold_val = 0.5;
        int main_phase_id = 1;
        std::string main_solver_str = "FlexGMRES";
        int main_box_size = 32;
        int main_verbose = 1;
        int main_write_plotfile = 0;
        std::string main_calculation_method = "homogenization"; // "homogenization" or "flow_through"

        // --- REV Study Parameters ---
        bool rev_do_study = false;
        int rev_num_samples = 3;
        std::string rev_sizes_str = "32 64 96"; // Space-separated string of sizes
        int rev_phase_id = 1; // Could differ from main_phase_id if needed
        std::string rev_solver_str = "FlexGMRES"; // Could differ
        std::string rev_results_file = "rev_study_Deff.csv";
        int rev_write_plotfiles = 0; // Plotfiles for each REV chi_k (0=no, 1=yes)
        int rev_verbose = 1; // Verbosity for REV study part

        {
            amrex::ParmParse pp; // Default (command line / first inputs file)
            pp.get("filename", main_filename);
            pp.query("data_path", main_data_path);
            pp.query("results_path", main_results_path);
            pp.query("hdf5_dataset", main_hdf5_dataset); // Optional, only for HDF5
            pp.query("threshold_val", main_threshold_val);
            pp.query("phase_id", main_phase_id);
            pp.query("solver_type", main_solver_str);
            pp.query("box_size", main_box_size);
            pp.query("verbose", main_verbose);
            pp.query("write_plotfile", main_write_plotfile);
            pp.query("calculation_method", main_calculation_method);


            amrex::ParmParse ppr("rev"); // Parameters prefixed with "rev."
            ppr.query("do_study", rev_do_study);
            ppr.query("num_samples", rev_num_samples);
            ppr.query("sizes", rev_sizes_str); // e.g., rev.sizes = "32 64 128"
            ppr.query("phase_id", rev_phase_id); // Overrides main_phase_id for REV if specified
            ppr.query("solver_type", rev_solver_str); // Overrides main_solver_type for REV
            ppr.query("results_file", rev_results_file);
            ppr.query("write_plotfiles", rev_write_plotfiles);
            ppr.query("verbose", rev_verbose);
        }
        
        // Create main results directory
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(main_results_path, 0755);
        }
        amrex::ParallelDescriptor::Barrier();


        // --- Load Full Domain Data (Once) ---
        amrex::Geometry geom_full;
        amrex::BoxArray ba_full;
        amrex::DistributionMapping dm_full;
        amrex::iMultiFab mf_phase_full; // For the thresholded full input image
        amrex::Box domain_box_full;

        try {
            std::string full_input_path = main_data_path + main_filename;
            if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << "Reading full domain data from: " << full_input_path << std::endl;

            // Reader selection logic (simplified from prototype)
            std::string ext = main_filename.substr(main_filename.find_last_of(".") + 1);
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            std::unique_ptr<OpenImpala::ImageReader> reader_full;
            if (ext == "tif" || ext == "tiff") {
                reader_full = std::make_unique<OpenImpala::TiffReader>(full_input_path);
            } else if (ext == "dat") {
                reader_full = std::make_unique<OpenImpala::DatReader>(full_input_path); // Ensure DatReader has a simple constructor or adjust
            } else if (ext == "h5" || ext == "hdf5") {
                reader_full = std::make_unique<OpenImpala::HDF5Reader>(full_input_path, main_hdf5_dataset);
            } // Add other readers like TiffStackReader if distinct, or RawReader
            else {
                throw std::runtime_error("Unsupported file extension: " + ext);
            }

            if (!reader_full || !reader_full->isRead()) {
                throw std::runtime_error("Failed to initialize or read metadata with selected reader.");
            }
            domain_box_full = reader_full->box();

            amrex::RealBox rb_full({AMREX_D_DECL(0.0,0.0,0.0)},
                                   {AMREX_D_DECL(amrex::Real(domain_box_full.length(0)),
                                                 amrex::Real(domain_box_full.length(1)),
                                                 amrex::Real(domain_box_full.length(2)))});
            // For full domain, periodicity depends on the main_calculation_method
            amrex::Array<int,AMREX_SPACEDIM> is_periodic_full;
            if (main_calculation_method == "homogenization") {
                is_periodic_full = {AMREX_D_DECL(1,1,1)};
            } else { // flow_through or other
                is_periodic_full = {AMREX_D_DECL(0,0,0)};
            }
            geom_full.define(domain_box_full, &rb_full, 0, is_periodic_full.data());

            ba_full.define(geom_full.Domain());
            ba_full.maxSize(main_box_size);
            dm_full.define(ba_full);

            // mf_phase_full needs 1 ghost cell for EffDiffHypre, maybe 2 for safety/other uses
            mf_phase_full.define(ba_full, dm_full, 1, 1); 
            
            amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
            reader_full->threshold(main_threshold_val, rev_phase_id /* Using rev.phase_id for consistency if REV is done */,
                                  (rev_phase_id == 0 ? 1 : 0), mf_temp_no_ghost);

            amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0, 0, 1, 0);
            mf_phase_full.FillBoundary(geom_full.periodicity()); // Fill based on geom_full's periodicity

        } catch (const std::exception& e) {
            amrex::Print() << "Error loading full domain data: " << e.what() << std::endl;
            amrex::Abort("Full domain data loading failed.");
        }

        // --- REV Study ---
        if (rev_do_study) {
            if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Starting REV Study (Homogenization Method) ---\n";
                amrex::Print() << "  Number of samples: " << rev_num_samples << std::endl;
                amrex::Print() << "  REV sizes:         " << rev_sizes_str << std::endl;
                amrex::Print() << "  Phase ID for REV:  " << rev_phase_id << std::endl;
                amrex::Print() << "  Solver for REV:    " << rev_solver_str << std::endl;
            }

            std::vector<int> rev_actual_sizes;
            std::stringstream ss_rev_sizes(rev_sizes_str);
            int size_val;
            while (ss_rev_sizes >> size_val) {
                rev_actual_sizes.push_back(size_val);
            }
            if (rev_actual_sizes.empty()) {
                amrex::Warning("REV sizes string is empty or invalid. Skipping REV study.");
                rev_do_study = false; // Turn it off
            }

            std::ofstream rev_csv_file;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                std::string full_rev_results_path = main_results_path + rev_results_file;
                rev_csv_file.open(full_rev_results_path);
                rev_csv_file << "SampleNo,SeedX,SeedY,SeedZ,REV_Size,D_xx,D_yy,D_zz,D_xy,D_xz,D_yz\n"; // Header
            }

            std::mt19937 gen(amrex::ParallelDescriptor::MyProc()); // Seed with MPI rank for different sequences per rank if desired, or use a fixed/random_device seed.
            
            int max_possible_rev_size = 0;
            if (!rev_actual_sizes.empty()) max_possible_rev_size = *std::max_element(rev_actual_sizes.begin(), rev_actual_sizes.end());


            for (int s_idx = 0; s_idx < rev_num_samples; ++s_idx) {
                // Generate random seed, ensuring the largest REV can be somewhat centered
                // This sampling range tries to keep the REV away from edges by half its max size.
                int sx_min = max_possible_rev_size / 2;
                int sx_max = domain_box_full.length(0) - max_possible_rev_size / 2 -1;
                int sy_min = max_possible_rev_size / 2;
                int sy_max = domain_box_full.length(1) - max_possible_rev_size / 2 -1;
                int sz_min = max_possible_rev_size / 2;
                int sz_max = domain_box_full.length(2) - max_possible_rev_size / 2 -1;

                // If domain is too small for this centering, adjust sampling range
                if (sx_min >= sx_max) { sx_min = 0; sx_max = std::max(0, domain_box_full.length(0) - 1); }
                if (sy_min >= sy_max) { sy_min = 0; sy_max = std::max(0, domain_box_full.length(1) - 1); }
                if (sz_min >= sz_max) { sz_min = 0; sz_max = std::max(0, domain_box_full.length(2) - 1); }

                std::uniform_int_distribution<> distr_x(sx_min, sx_max);
                std::uniform_int_distribution<> distr_y(sy_min, sy_max);
                std::uniform_int_distribution<> distr_z(sz_min, sz_max);

                amrex::IntVect seed_center(distr_x(gen), distr_y(gen), distr_z(gen));
                 // Translate seed_center to global coordinates of domain_box_full
                seed_center += domain_box_full.smallEnd();


                if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                     amrex::Print() << "REV Sample " << s_idx + 1 << "/" << rev_num_samples
                                    << ", Seed Center: " << seed_center << std::endl;
                }


                for (int current_rev_size : rev_actual_sizes) {
                    if (rev_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "  Processing REV Size: " << current_rev_size << std::endl;
                    }

                    amrex::IntVect rev_lo = seed_center - amrex::IntVect(current_rev_size/2);
                    amrex::IntVect rev_hi = rev_lo + amrex::IntVect(current_rev_size -1);
                    amrex::Box bx_rev(rev_lo, rev_hi);

                    // Boundary correction (shift box if it goes out of full domain bounds)
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
                        if (bx_rev.smallEnd(d) < domain_box_full.smallEnd(d)) {
                            int shift = domain_box_full.smallEnd(d) - bx_rev.smallEnd(d);
                            bx_rev.shift(d, shift);
                        }
                        if (bx_rev.bigEnd(d) > domain_box_full.bigEnd(d)) {
                            int shift = domain_box_full.bigEnd(d) - bx_rev.bigEnd(d);
                            bx_rev.shift(d, shift);
                        }
                        // Final check: if REV is larger than domain in a dim, cap it.
                        if (bx_rev.length(d) > domain_box_full.length(d)) {
                           bx_rev.setSmall(d, domain_box_full.smallEnd(d));
                           bx_rev.setBig(d, domain_box_full.bigEnd(d));
                        }
                    }
                     if (rev_verbose >=2 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "    Adjusted REV Box: " << bx_rev << std::endl;
                    }
                    if (bx_rev.isEmpty() || bx_rev.longside() == 0) {
                        if (rev_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Warning("Skipping empty or invalid REV box after adjustments.");
                        continue;
                    }


                    // AMReX Setup for this REV sub-domain
                    amrex::Geometry geom_rev;
                    amrex::RealBox rb_rev({AMREX_D_DECL(0.0,0.0,0.0)},
                                          {AMREX_D_DECL(amrex::Real(bx_rev.length(0)),
                                                        amrex::Real(bx_rev.length(1)),
                                                        amrex::Real(bx_rev.length(2)))});
                    amrex::Array<int,AMREX_SPACEDIM> is_periodic_rev = {AMREX_D_DECL(1,1,1)}; // Homogenization is periodic
                    geom_rev.define(bx_rev, &rb_rev, 0, is_periodic_rev.data());

                    amrex::BoxArray ba_rev(geom_rev.Domain()); // ba_rev defined on the (potentially shifted) bx_rev
                    ba_rev.maxSize(main_box_size);
                    amrex::DistributionMapping dm_rev(ba_rev);

                    amrex::iMultiFab mf_phase_rev(ba_rev, dm_rev, 1, 1); // 1 ghost for EffDiffHypre
                    
                    // Efficiently populate mf_phase_rev from mf_phase_full
                    // Create a temporary MultiFab on a BoxArray consisting only of bx_rev, with 0 ghost cells.
                    amrex::BoxArray ba_rev_for_copy(bx_rev); // Single box
                    amrex::DistributionMapping dm_rev_for_copy(ba_rev_for_copy); // Simple DM for single box
                    amrex::iMultiFab mf_temp_copy(ba_rev_for_copy, dm_rev_for_copy, 1, 0);
                    
                    // Copy from mf_phase_full (src_comp 0) into mf_temp_copy (dst_comp 0), for 1 component,
                    // 0 src ghost cells (valid data only), for the region bx_rev.
                    amrex::Copy(mf_temp_copy, mf_phase_full, 0, 0, 1, 0, bx_rev);

                    // Now copy from the no-ghost temporary mf to mf_phase_rev (which has ghost cells)
                    amrex::Copy(mf_phase_rev, mf_temp_copy, 0, 0, 1, 0);
                    mf_phase_rev.FillBoundary(geom_rev.periodicity());

                    // --- Solve for chi_k on this REV ---
                    amrex::MultiFab mf_chi_x_rev(ba_rev, dm_rev, 1, 1);
                    amrex::MultiFab mf_chi_y_rev(ba_rev, dm_rev, 1, 1);
                    amrex::MultiFab mf_chi_z_rev;
                    if (AMREX_SPACEDIM==3) mf_chi_z_rev.define(ba_rev, dm_rev, 1, 1);

                    bool all_chi_rev_converged = true;
                    std::vector<OpenImpala::Direction> rev_solve_dirs = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
                    if (AMREX_SPACEDIM == 3) rev_solve_dirs.push_back(OpenImpala::Direction::Z);

                    OpenImpala::EffectiveDiffusivityHypre::SolverType rev_solver_type_enum = stringToSolverType(rev_solver_str);

                    for (const auto& dir_k_solve : rev_solve_dirs) {
                        std::string chi_plot_subdir = main_results_path + "/REV_Sample" + std::to_string(s_idx) + "_Size" + std::to_string(current_rev_size) + "/";
                        if (rev_write_plotfiles != 0 && amrex::ParallelDescriptor::IOProcessor()) {
                             amrex::UtilCreateDirectory(chi_plot_subdir, 0755);
                        }
                        amrex::ParallelDescriptor::Barrier();


                        std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> rev_chi_solver;
                        try {
                             rev_chi_solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                                geom_rev, ba_rev, dm_rev, mf_phase_rev,
                                rev_phase_id, dir_k_solve, rev_solver_type_enum, 
                                chi_plot_subdir, // Pass specific subdir for plotfiles
                                rev_verbose, (rev_write_plotfiles !=0)
                            );
                            if (!rev_chi_solver->solve()) {
                                all_chi_rev_converged = false;
                                if (rev_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) 
                                    amrex::Warning("REV chi solve failed for sample " + std::to_string(s_idx) + 
                                                   ", size " + std::to_string(current_rev_size) + 
                                                   ", dir " + std::to_string(static_cast<int>(dir_k_solve)));
                                break; // Stop solving other chi for this REV if one fails
                            }
                            if (dir_k_solve == OpenImpala::Direction::X) rev_chi_solver->getChiSolution(mf_chi_x_rev);
                            else if (dir_k_solve == OpenImpala::Direction::Y) rev_chi_solver->getChiSolution(mf_chi_y_rev);
                            else if (AMREX_SPACEDIM==3 && dir_k_solve == OpenImpala::Direction::Z) rev_chi_solver->getChiSolution(mf_chi_z_rev);

                        } catch (const std::exception& e) {
                            all_chi_rev_converged = false;
                             if (rev_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) 
                                amrex::Print() << " Exception in REV chi solve: " << e.what() << std::endl;
                            break;
                        }
                    }

                    amrex::Real Deff_tensor_rev[AMREX_SPACEDIM][AMREX_SPACEDIM];
                    for(int r=0; r<AMREX_SPACEDIM; ++r) for(int c=0; c<AMREX_SPACEDIM; ++c) Deff_tensor_rev[r][c] = std::numeric_limits<amrex::Real>::quiet_NaN();

                    if (all_chi_rev_converged) {
                        // Create 0-ghost active mask for this REV
                        amrex::iMultiFab active_mask_rev(ba_rev, dm_rev, 1, 0);
                        #ifdef AMREX_USE_OMP
                        #pragma omp parallel if(amrex::Gpu::notInLaunchRegion())
                        #endif
                        for (amrex::MFIter mfi(active_mask_rev, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
                            const amrex::Box& tilebox = mfi.tilebox();
                            auto const& mask_arr = active_mask_rev.array(mfi);
                            auto const& phase_arr = mf_phase_rev.const_array(mfi); // From copied & ghost-filled sub-domain
                            amrex::ParallelFor(tilebox, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                                mask_arr(i,j,k) = (phase_arr(i,j,k) == rev_phase_id) ? 1 : 0;
                            });
                        }
                        calculate_Deff_tensor_homogenization(Deff_tensor_rev,
                                                             mf_chi_x_rev, mf_chi_y_rev, mf_chi_z_rev,
                                                             active_mask_rev, geom_rev, rev_verbose);
                    }
                    
                    // Write to CSV
                    if (amrex::ParallelDescriptor::IOProcessor()) {
                        rev_csv_file << s_idx << ","
                                     << seed_center[0] << "," << seed_center[1] << "," << seed_center[2] << ","
                                     << current_rev_size << ","
                                     << std::fixed << std::setprecision(8)
                                     << Deff_tensor_rev[0][0] << "," << Deff_tensor_rev[1][1] << ","
                                     << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[2][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                     << Deff_tensor_rev[0][1] << ","
                                     << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[0][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                     << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[1][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << "\n";
                    }
                } // End loop over REV sizes
            } // End loop over REV samples

            if (amrex::ParallelDescriptor::IOProcessor()) {
                rev_csv_file.close();
                amrex::Print() << "--- REV Study Finished. Results in: "
                               << main_results_path + rev_results_file << " ---\n";
            }
            amrex::ParallelDescriptor::Barrier();

        } // End if (rev_do_study)

        // --- Full Domain Calculation (Optional: based on main_calculation_method) ---
        // This part would be similar to your existing Diffusion.cpp main logic
        // using main_*, geom_full, ba_full, dm_full, mf_phase_full
        if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Full Domain Calculation (" << main_calculation_method << ") ---\n";
        }
        if (main_calculation_method == "homogenization") {
            // ... (Call EffectiveDiffusivityHypre for chi_x, chi_y, chi_z on full domain) ...
            // ... (Call calculate_Deff_tensor_homogenization on full domain) ...
            // ... (Print/save D_eff_full tensor) ...
            amrex::Print() << "Full domain homogenization calculation placeholder." << std::endl;
        } else if (main_calculation_method == "flow_through") {
            // ... (Instantiate TortuosityHypre and calculate for X, Y, Z or specified direction) ...
            // ... (Print/save tortuosity / Deff_scalar) ...
            amrex::Print() << "Full domain flow-through calculation placeholder." << std::endl;
        } else {
            if (main_verbose >=0 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Warning("Unknown main_calculation_method: " + main_calculation_method + ". Skipping full domain calculation.");
            }
        }


        amrex::Real master_stop_time = amrex::second() - master_strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(master_stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << std::endl << "Total program run time (seconds) = " << master_stop_time << std::endl;
        }

    } // End AMReX Scope
    amrex::Finalize();

    hypre_ierr = HYPRE_Finalize();
    if (hypre_ierr != 0) {
        fprintf(stderr, "ERROR: HYPRE_Finalize() failed with code %d\n", hypre_ierr);
        return 1;
    }
    return 0;
}

// Make sure to place the definition of calculate_Deff_tensor_homogenization
// either before main or ensure its prototype is declared before main if defined later.
// I've moved its definition into the anonymous namespace above main.
