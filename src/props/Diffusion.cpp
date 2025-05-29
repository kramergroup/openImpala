/** REV calculation / Full Domain Diffusion
 *
 * This programme calculates the effective diffusivity tensor using homogenization,
 * either for the full domain or as part of a Representative Elementary Volume (REV) study.
 * For an REV study, it analyzes multiple random sub-volumes of increasing sizes.
 *
 * Based on the "Diffusion_Variant_A.cpp" reader handling style.
 */

#include "../io/TiffReader.H"
#include "../io/DatReader.H"   // Assuming these are your existing headers
#include "../io/HDF5Reader.H"
// #include "../io/RawReader.H" // Uncomment if used

#include "EffectiveDiffusivityHypre.H"
#include "TortuosityHypre.H" // For potential flow-through on full domain
#include "VolumeFraction.H"
#include "Tortuosity.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>
#include <AMReX_Utility.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H> // For amrex::Copy

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <filesystem> // For path joining

// Anonymous namespace for helpers
namespace {
    // ... (stringToSolverType, calculate_Deff_tensor_homogenization definitions - keep these)
    OpenImpala::EffectiveDiffusivityHypre::SolverType stringToSolverType(const std::string& solver_str) {
        std::string lower_solver_str = solver_str;
        std::transform(lower_solver_str.begin(), lower_solver_str.end(), lower_solver_str.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (lower_solver_str == "jacobi") return OpenImpala::EffectiveDiffusivityHypre::SolverType::Jacobi;
        if (lower_solver_str == "gmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES;
        if (lower_solver_str == "flexgmres") return OpenImpala::EffectiveDiffusivityHypre::SolverType::FlexGMRES;
        // ... add other solver types from your helper ...
        amrex::Abort("Invalid solver string: '" + solver_str + "'.");
        return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES;
    }

    // calculate_Deff_tensor_homogenization definition (as provided before, it's correct)
void calculate_Deff_tensor_homogenization(
    amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
    const amrex::MultiFab& mf_chi_x_in,
    const amrex::MultiFab& mf_chi_y_in,
    const amrex::MultiFab& mf_chi_z_in,
    const amrex::iMultiFab& active_mask, 
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
    for (int i_ = 0; i_ < AMREX_SPACEDIM; ++i_) { 
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
    HYPRE_Init();
    amrex::Initialize(argc, argv);
    {
        amrex::Real master_strt_time = amrex::second();

        // --- Main Parameters (as before) ---
        std::string main_filename;
        std::string main_data_path_str = "./data/";
        std::string main_results_path_str = "./results/";
        std::string main_hdf5_dataset;
        amrex::Real main_threshold_val = 0.5;
        int main_phase_id_analysis = 1; // Phase to analyze (for VF, Tort, Deff)
        std::string main_solver_str = "FlexGMRES";
        int main_box_size = 32;
        int main_verbose = 1;
        int main_write_plotfile_full = 0;
        std::string main_calculation_method = "homogenization";

        // --- REV Study Parameters (as before) ---
        bool rev_do_study = false;
        int rev_num_samples = 3;
        std::string rev_sizes_str = "32 64 96";
        // int rev_phase_id_analysis = 1; // Use main_phase_id_analysis for REV for consistency
        std::string rev_solver_str = "FlexGMRES";
        std::string rev_results_filename = "rev_study_Deff.csv";
        int rev_write_plotfiles = 0;
        int rev_verbose_level = 1;


        { // ParmParse Scope
            amrex::ParmParse pp;
            pp.get("filename", main_filename);
            pp.query("data_path", main_data_path_str);
            pp.query("results_path", main_results_path_str);
            pp.query("hdf5_dataset", main_hdf5_dataset);
            pp.query("threshold_val", main_threshold_val);
            pp.query("phase_id", main_phase_id_analysis); // Phase to analyze
            pp.query("solver_type", main_solver_str);
            pp.query("box_size", main_box_size);
            pp.query("verbose", main_verbose);
            pp.query("write_plotfile", main_write_plotfile_full);
            pp.query("calculation_method", main_calculation_method);

            amrex::ParmParse ppr("rev");
            ppr.query("do_study", rev_do_study);
            ppr.query("num_samples", rev_num_samples);
            ppr.query("sizes", rev_sizes_str);
            // ppr.query("phase_id", rev_phase_id_analysis); // Let's use main_phase_id_analysis for REV
            ppr.query("solver_type", rev_solver_str);
            ppr.query("results_file", rev_results_filename);
            ppr.query("write_plotfiles", rev_write_plotfiles);
            ppr.query("verbose", rev_verbose_level);
        }

        std::filesystem::path main_data_path(main_data_path_str);
        std::filesystem::path main_results_path(main_results_path_str);
        // ... (Handle '~' for paths if needed) ...
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::UtilCreateDirectory(main_results_path.string(), 0755);
        }
        amrex::ParallelDescriptor::Barrier();
        std::filesystem::path full_input_path = main_data_path / main_filename;


        // --- Load Full Domain Data (Once) ---
        amrex::Geometry geom_full;
        amrex::BoxArray ba_full;
        amrex::DistributionMapping dm_full;
        amrex::iMultiFab mf_phase_full; // Thresholded full domain phase data
        amrex::Box domain_box_full;

        try {
            if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << "Reading full domain data from: " << full_input_path.string() << std::endl;

            std::string ext;
            if (full_input_path.has_extension()) {
                ext = full_input_path.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            } else {
                 throw std::runtime_error("File has no extension: " + full_input_path.string());
            }

            // Instantiate reader locally within if/else blocks
            if (ext == ".tif" || ext == ".tiff") {
                OpenImpala::TiffReader reader(full_input_path.string());
                if (!reader.isRead()) throw std::runtime_error("TiffReader failed to read metadata.");
                domain_box_full = reader.box();
                // Define geom, ba, dm, mf_phase_full for the full domain
                ba_full.define(domain_box_full);
                ba_full.maxSize(main_box_size);
                dm_full.define(ba_full);
                mf_phase_full.define(ba_full, dm_full, 1, 1); // 1 ghost for EffDiff, could be 2
                // Temporary MF for thresholding
                amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
                reader.threshold(main_threshold_val, main_phase_id_analysis, (main_phase_id_analysis == 0 ? 1 : 0), mf_temp_no_ghost);
                amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0,0,1,0);

            } else if (ext == ".dat") {
                OpenImpala::DatReader reader(full_input_path.string());
                // DatReader's threshold needs amrex::Real, its internal DataType raw_threshold is different.
                // We'll assume DatReader::threshold(amrex::Real, ...) is implemented.
                // Or, if DatReader only has threshold(DatReader::DataType, ...), then:
                // OpenImpala::DatReader::DataType dat_thresh = static_cast<OpenImpala::DatReader::DataType>(main_threshold_val);

                if (!reader.isRead()) throw std::runtime_error("DatReader failed to read metadata.");
                domain_box_full = reader.box();
                ba_full.define(domain_box_full);
                ba_full.maxSize(main_box_size);
                dm_full.define(ba_full);
                mf_phase_full.define(ba_full, dm_full, 1, 1);
                amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
                // Assuming DatReader has a threshold method that takes amrex::Real or can be adapted
                reader.threshold(main_threshold_val, main_phase_id_analysis, (main_phase_id_analysis == 0 ? 1 : 0), mf_temp_no_ghost);
                amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0,0,1,0);

            } else if (ext == ".h5" || ext == ".hdf5") {
                OpenImpala::HDF5Reader reader(full_input_path.string(), main_hdf5_dataset);
                if (!reader.isRead()) throw std::runtime_error("HDF5Reader failed to read metadata.");
                domain_box_full = reader.box();
                ba_full.define(domain_box_full);
                ba_full.maxSize(main_box_size);
                dm_full.define(ba_full);
                mf_phase_full.define(ba_full, dm_full, 1, 1);
                amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
                reader.threshold(main_threshold_val, main_phase_id_analysis, (main_phase_id_analysis == 0 ? 1 : 0), mf_temp_no_ghost);
                amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0,0,1,0);
            }
            // Add RawReader, TiffStackReader cases here if needed
            else {
                throw std::runtime_error("Unsupported file extension for full domain load: " + ext);
            }

            // Common setup for geom_full after domain_box_full is known
            amrex::RealBox rb_full({AMREX_D_DECL(0.0,0.0,0.0)},
                                   {AMREX_D_DECL(amrex::Real(domain_box_full.length(0)),
                                                 amrex::Real(domain_box_full.length(1)),
                                                 amrex::Real(domain_box_full.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic_full;
            if (main_calculation_method == "homogenization" || rev_do_study) { // REV study needs periodic full domain for easy copying
                is_periodic_full = {AMREX_D_DECL(1,1,1)};
            } else {
                is_periodic_full = {AMREX_D_DECL(0,0,0)};
            }
            geom_full.define(domain_box_full, &rb_full, 0, is_periodic_full.data());
            mf_phase_full.FillBoundary(geom_full.periodicity());

        } catch (const std::exception& e) {
            amrex::Print() << "Error loading full domain data: " << e.what() << std::endl;
            amrex::Abort("Full domain data loading failed.");
        }


        // --- REV Study ---
        if (rev_do_study) {
            // ... (REV study logic as provided in the previous comprehensive solution,
            //      using mf_phase_full for data, and parameters like
            //      rev_num_samples, rev_sizes_str, main_phase_id_analysis, rev_solver_str, etc.)
            //      The key change is that you already have mf_phase_full loaded.
            //      The copy into mf_phase_rev from mf_phase_full is the crucial part.

            if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Starting REV Study (Homogenization Method) ---\n";
                // ... print other rev params ...
            }

            std::vector<int> rev_actual_sizes;
            std::stringstream ss_rev_sizes(rev_sizes_str);
            int size_val;
            while (ss_rev_sizes >> size_val) rev_actual_sizes.push_back(size_val);

            if (rev_actual_sizes.empty()) {
                amrex::Warning("REV sizes string is empty. Skipping REV study.");
            } else {
                std::ofstream rev_csv_file;
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    std::string full_rev_csv_path = main_results_path.string() + "/" + rev_results_filename;
                    rev_csv_file.open(full_rev_csv_path);
                    rev_csv_file << "SampleNo,SeedX,SeedY,SeedZ,REV_Size,ActualSizeX,ActualSizeY,ActualSizeZ,D_xx,D_yy,D_zz,D_xy,D_xz,D_yz\n";
                }

                std::mt19937 gen(amrex::ParallelDescriptor::MyProc() + 12345); // Basic seed
                int max_rev_dim = domain_box_full.longside(); // Simplistic max for sampling
                if (!rev_actual_sizes.empty()) max_rev_dim = *std::max_element(rev_actual_sizes.begin(), rev_actual_sizes.end());


                for (int s_idx = 0; s_idx < rev_num_samples; ++s_idx) {
                    // Simplified Seed Generation - ensure it's within full domain for simplicity
                    amrex::IntVect seed_center;
                    for(int d=0; d<AMREX_SPACEDIM; ++d) {
                        // Ensure seed allows for at least smallest REV size
                        int min_coord = domain_box_full.smallEnd(d) + rev_actual_sizes[0]/2;
                        int max_coord = domain_box_full.bigEnd(d)   - rev_actual_sizes[0]/2;
                        if (min_coord > max_coord) min_coord = max_coord = domain_box_full.smallEnd(d) + domain_box_full.length(d)/2; // Fallback if domain too small
                        std::uniform_int_distribution<> distr(min_coord, max_coord);
                        seed_center[d] = distr(gen);
                    }


                    if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "REV Sample " << s_idx + 1 << "/" << rev_num_samples
                                       << ", Seed Center (global): " << seed_center << std::endl;
                    }

                    for (int current_rev_size_target : rev_actual_sizes) {
                        if (rev_verbose_level >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                            amrex::Print() << "  Target REV Size: " << current_rev_size_target << std::endl;
                        }

                        amrex::IntVect rev_half_size(AMREX_D_DECL(current_rev_size_target/2, current_rev_size_target/2, current_rev_size_target/2));
                        amrex::IntVect rev_lo = seed_center - rev_half_size;
                        amrex::IntVect rev_hi = rev_lo + amrex::IntVect(current_rev_size_target -1);
                        amrex::Box bx_rev(rev_lo, rev_hi);
                        
                        // Intersect with the full domain to ensure REV is within bounds
                        bx_rev &= domain_box_full; 

                        if (bx_rev.isEmpty() || bx_rev.longside() < 8) { // Skip very small or empty REVs
                            if (rev_verbose_level >=1 && amrex::ParallelDescriptor::IOProcessor()) 
                                amrex::Warning("Skipping REV for sample " + std::to_string(s_idx) + " size " + std::to_string(current_rev_size_target) + " due to small/empty box after boundary intersection: " + bx_rev.toString() );
                            continue;
                        }
                         if (rev_verbose_level >=2 && amrex::ParallelDescriptor::IOProcessor()) {
                            amrex::Print() << "    Actual REV Box (global coords): " << bx_rev << std::endl;
                        }

                        amrex::Geometry geom_rev;
                        amrex::RealBox rb_rev({AMREX_D_DECL(0.0,0.0,0.0)},
                                              {AMREX_D_DECL(amrex::Real(bx_rev.length(0)), amrex::Real(bx_rev.length(1)), amrex::Real(bx_rev.length(2)))});
                        amrex::Array<int,AMREX_SPACEDIM> is_periodic_rev = {AMREX_D_DECL(1,1,1)};
                        geom_rev.define(bx_rev, &rb_rev, 0, is_periodic_rev.data()); // bx_rev is global here

                        amrex::BoxArray ba_rev(geom_rev.Domain()); // Uses global bx_rev
                        ba_rev.maxSize(main_box_size);
                        amrex::DistributionMapping dm_rev(ba_rev);

                        amrex::iMultiFab mf_phase_rev(ba_rev, dm_rev, 1, 1);
                        
                        // Copy data from mf_phase_full (which is on ba_full, dm_full)
                        // into mf_phase_rev (which is on ba_rev, dm_rev).
                        // mf_phase_rev's BoxArray is simply bx_rev.
                        // AMReX Copy will handle the logic of finding the overlapping data.
                        mf_phase_rev.setVal(0); // Initialize to ensure no old data
                        amrex::Copy(mf_phase_rev, mf_phase_full, 0, 0, 1, 0);
                        mf_phase_rev.FillBoundary(geom_rev.periodicity());


                        amrex::MultiFab mf_chi_x_rev(ba_rev, dm_rev, 1, 1);
                        amrex::MultiFab mf_chi_y_rev(ba_rev, dm_rev, 1, 1);
                        amrex::MultiFab mf_chi_z_rev;
                        if (AMREX_SPACEDIM==3) mf_chi_z_rev.define(ba_rev, dm_rev, 1, 1);

                        bool all_chi_rev_converged = true;
                        std::vector<OpenImpala::Direction> rev_solve_dirs = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
                        if (AMREX_SPACEDIM == 3) rev_solve_dirs.push_back(OpenImpala::Direction::Z);
                        OpenImpala::EffectiveDiffusivityHypre::SolverType rev_solver_type_enum = stringToSolverType(rev_solver_str);

                        for (const auto& dir_k_solve : rev_solve_dirs) {
                            std::string chi_plot_rev_subdir = main_results_path.string() + "/REV_S" + std::to_string(s_idx) + "_Size" + std::to_string(bx_rev.longside()) + "_Dir" + std::to_string(static_cast<int>(dir_k_solve)) + "/";
                            if (rev_write_plotfiles != 0 && amrex::ParallelDescriptor::IOProcessor()) {
                                 amrex::UtilCreateDirectory(chi_plot_rev_subdir, 0755);
                            }
                             amrex::ParallelDescriptor::Barrier(); // Ensure dir created before use by other ranks

                            std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> rev_chi_solver;
                            try {
                                 rev_chi_solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                                    geom_rev, ba_rev, dm_rev, mf_phase_rev,
                                    main_phase_id_analysis, dir_k_solve, rev_solver_type_enum,
                                    chi_plot_rev_subdir, rev_verbose_level, (rev_write_plotfiles !=0)
                                );
                                if (!rev_chi_solver->solve()) {
                                    all_chi_rev_converged = false; break;
                                }
                                if (dir_k_solve == OpenImpala::Direction::X) rev_chi_solver->getChiSolution(mf_chi_x_rev);
                                else if (dir_k_solve == OpenImpala::Direction::Y) rev_chi_solver->getChiSolution(mf_chi_y_rev);
                                else if (AMREX_SPACEDIM==3 && dir_k_solve == OpenImpala::Direction::Z) rev_chi_solver->getChiSolution(mf_chi_z_rev);
                            } catch (const std::exception& e) {
                                all_chi_rev_converged = false; break;
                            }
                        }

                        amrex::Real Deff_tensor_rev[AMREX_SPACEDIM][AMREX_SPACEDIM];
                        for(int r=0; r<AMREX_SPACEDIM; ++r) for(int c=0; c<AMREX_SPACEDIM; ++c) Deff_tensor_rev[r][c] = std::numeric_limits<amrex::Real>::quiet_NaN();

                        if (all_chi_rev_converged) {
                            amrex::iMultiFab active_mask_rev(ba_rev, dm_rev, 1, 0);
                            #ifdef AMREX_USE_OMP
                            #pragma omp parallel if(amrex::Gpu::notInLaunchRegion())
                            #endif
                            for (amrex::MFIter mfi(active_mask_rev, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
                                const amrex::Box& tb = mfi.tilebox(); // Use tilebox for loop bounds
                                auto const& mask_arr = active_mask_rev.array(mfi);
                                auto const& phase_arr = mf_phase_rev.const_array(mfi);
                                amrex::LoopOnCpu(tb, [=] (int i, int j, int k) { // Use LoopOnCpu
                                    mask_arr(i,j,k) = (phase_arr(i,j,k) == main_phase_id_analysis) ? 1 : 0;
                                });
                            }
                            calculate_Deff_tensor_homogenization(Deff_tensor_rev,
                                                                 mf_chi_x_rev, mf_chi_y_rev, mf_chi_z_rev,
                                                                 active_mask_rev, geom_rev, rev_verbose_level);
                        }
                        if (amrex::ParallelDescriptor::IOProcessor()) {
                            rev_csv_file << s_idx << ","
                                         << seed_center[0] << "," << seed_center[1] << "," << seed_center[2] << ","
                                         << current_rev_size_target << "," // Target size
                                         << bx_rev.length(0) << "," << bx_rev.length(1) << "," << bx_rev.length(2) << "," // Actual size
                                         << std::fixed << std::setprecision(8)
                                         << Deff_tensor_rev[0][0] << "," << Deff_tensor_rev[1][1] << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[2][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                         << Deff_tensor_rev[0][1] << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[0][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[1][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << "\n";
                        }
                    } // End REV sizes
                } // End REV samples
                if (amrex::ParallelDescriptor::IOProcessor()) rev_csv_file.close();
            } // End if !rev_actual_sizes.empty()
        } // End if rev_do_study


        // --- Full Domain Calculation (Placeholder - fill this in from your previous Diffusion.cpp) ---
        if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Full Domain Calculation (" << main_calculation_method << ") using phase " << main_phase_id_analysis << " ---\n";
        }
        if (main_calculation_method == "homogenization") {
             if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor())
                amrex::Print() << "\n--- Effective Diffusivity via Homogenization (Full Domain) ---\n";

            amrex::MultiFab mf_chi_x_full(ba_full, dm_full, 1, 1);
            amrex::MultiFab mf_chi_y_full(ba_full, dm_full, 1, 1);
            amrex::MultiFab mf_chi_z_full; 
            if (AMREX_SPACEDIM == 3) mf_chi_z_full.define(ba_full, dm_full, 1, 1);
            bool all_chi_full_converged = true;

            std::vector<OpenImpala::Direction> dirs_to_solve_chi_full = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
            if (AMREX_SPACEDIM == 3) dirs_to_solve_chi_full.push_back(OpenImpala::Direction::Z);

            auto solver_type_effdiff_full = stringToSolverType(main_solver_str); // Use main_solver_str

            for (const auto& dir_k : dirs_to_solve_chi_full) {
                std::string dir_k_str = (dir_k == OpenImpala::Direction::X) ? "X" : (dir_k == OpenImpala::Direction::Y) ? "Y" : "Z";
                 if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor())
                    amrex::Print() << "\n--- Solving for Full Domain chi_" << dir_k_str << " ---\n";
                
                std::string full_chi_plot_dir = main_results_path.string() + "/FullDomain_chi_" + dir_k_str + "/";
                if (main_write_plotfile_full !=0 && amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::UtilCreateDirectory(full_chi_plot_dir, 0755);
                }
                amrex::ParallelDescriptor::Barrier();


                OpenImpala::EffectiveDiffusivityHypre solver_chi_k_full(
                    geom_full, ba_full, dm_full, mf_phase_full, // Use full domain objects
                    main_phase_id_analysis, dir_k, solver_type_effdiff_full, 
                    full_chi_plot_dir, // Pass specific dir
                    main_verbose, (main_write_plotfile_full != 0)
                );
                if (!solver_chi_k_full.solve()) {
                    all_chi_full_converged = false;
                    // Handle non-convergence if needed (e.g., log, set mf_chi to 0)
                    break; 
                }
                if (dir_k == OpenImpala::Direction::X) solver_chi_k_full.getChiSolution(mf_chi_x_full);
                else if (dir_k == OpenImpala::Direction::Y) solver_chi_k_full.getChiSolution(mf_chi_y_full);
                else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) solver_chi_k_full.getChiSolution(mf_chi_z_full);
            }

            if (all_chi_full_converged) {
                amrex::Real Deff_tensor_full[AMREX_SPACEDIM][AMREX_SPACEDIM];
                amrex::iMultiFab active_mask_full(ba_full, dm_full, 1, 0);
                #ifdef AMREX_USE_OMP
                #pragma omp parallel if(amrex::Gpu::notInLaunchRegion())
                #endif
                for (amrex::MFIter mfi(active_mask_full, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
                    const amrex::Box& tb = mfi.tilebox();
                    auto const& mask_arr = active_mask_full.array(mfi);
                    auto const& phase_arr = mf_phase_full.const_array(mfi);
                    amrex::LoopOnCpu(tb, [=] (int i, int j, int k) {
                        mask_arr(i,j,k) = (phase_arr(i,j,k) == main_phase_id_analysis) ? 1 : 0;
                    });
                }
                calculate_Deff_tensor_homogenization(
                    Deff_tensor_full, mf_chi_x_full, mf_chi_y_full, mf_chi_z_full,
                    active_mask_full, geom_full, main_verbose);

                if (amrex::ParallelDescriptor::IOProcessor()) {
                    amrex::Print() << "Full Domain Effective Diffusivity Tensor D_eff / D_material:\n";
                    // ... (Print tensor) ...
                     for (int r = 0; r < AMREX_SPACEDIM; ++r) {
                        amrex::Print() << "  [";
                        for (int c = 0; c < AMREX_SPACEDIM; ++c) {
                            amrex::Print() << std::scientific << std::setprecision(8) << Deff_tensor_full[r][c]
                                           << (c == AMREX_SPACEDIM - 1 ? "" : ", ");
                        }
                        amrex::Print() << "]\n";
                    }
                    // ... (Write to main output file) ...
                }
            } else {
                 if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Full domain D_eff calculation skipped due to chi_k non-convergence.\n";
            }

        } else if (main_calculation_method == "flow_through") {
            // ... (Full domain flow-through logic using TortuosityHypre, mf_phase_full, geom_full etc.)
            amrex::Print() << "Full domain flow-through calculation (placeholder)." << std::endl;
        }


        amrex::Real master_stop_time = amrex::second() - master_strt_time;
        // ... (Reduce and print time) ...
        amrex::ParallelDescriptor::ReduceRealMax(master_stop_time,amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor())
            amrex::Print() << std::endl << "Total run time (seconds) = " << master_stop_time << std::endl;

    }
    amrex::Finalize();
    HYPRE_Finalize();
    return 0;
}
