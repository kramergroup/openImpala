/** REV calculation / Full Domain Diffusion
 *
 * This programme calculates the effective diffusivity tensor using homogenization,
 * either for the full domain or as part of a Representative Elementary Volume (REV) study.
 * For an REV study, it analyzes multiple random sub-volumes of increasing sizes.
 *
 */

#include "../io/TiffReader.H"
#include "../io/DatReader.H"
#include "../io/HDF5Reader.H"
// #include "../io/RawReader.H" // Uncomment if used

#include "EffectiveDiffusivityHypre.H"
#include "TortuosityHypre.H"
#include "VolumeFraction.H"
#include "Tortuosity.H" // For OpenImpala::Direction enum

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
#include <AMReX_MultiFabUtil.H>

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <iomanip>
#include <filesystem>
#include <limits> // Required for std::numeric_limits

// Anonymous namespace for helpers
namespace {
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
        return OpenImpala::EffectiveDiffusivityHypre::SolverType::GMRES; // Should not reach
    }

void calculate_Deff_tensor_homogenization(
    amrex::Real Deff_tensor[AMREX_SPACEDIM][AMREX_SPACEDIM],
    const amrex::MultiFab& mf_chi_x_in,
    const amrex::MultiFab& mf_chi_y_in,
    const amrex::MultiFab& mf_chi_z_in,
    const amrex::iMultiFab& active_mask, 
    const amrex::Geometry& geom,
    int verbose_level)
{
    BL_PROFILE("calculate_Deff_tensor_homogenization_main_driver"); 
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            Deff_tensor[i][j] = 0.0;
        }
    }
    AMREX_ASSERT(mf_chi_x_in.nGrow() >= 1); 
    AMREX_ASSERT(mf_chi_y_in.nGrow() >= 1);
    if (AMREX_SPACEDIM == 3) {
        AMREX_ASSERT(mf_chi_z_in.isDefined() && mf_chi_z_in.nGrow() >= 1); 
    }
    AMREX_ASSERT(active_mask.nGrow() == 0); 

    const amrex::Real* dx_arr = geom.CellSize();
    amrex::Real inv_2dx[AMREX_SPACEDIM];
    for (int i=0; i<AMREX_SPACEDIM; ++i) {
        inv_2dx[i] = 1.0 / (2.0 * dx_arr[i]);
    }

    amrex::Real sum_integrand_tensor_comp_local[AMREX_SPACEDIM][AMREX_SPACEDIM];
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        for (int j = 0; j < AMREX_SPACEDIM; ++j) {
            sum_integrand_tensor_comp_local[i][j] = 0.0;
        }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel reduction(+:sum_integrand_tensor_comp_local)
#endif
    for (amrex::MFIter mfi(active_mask, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox(); 
        amrex::Array4<const int> const mask_arr = active_mask.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_x_arr = mf_chi_x_in.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_y_arr = mf_chi_y_in.const_array(mfi);
        amrex::Array4<const amrex::Real> const chi_z_arr = (AMREX_SPACEDIM == 3 && mf_chi_z_in.isDefined()) ? 
                                                           mf_chi_z_in.const_array(mfi) :
                                                           mf_chi_x_in.const_array(mfi); 

        amrex::LoopOnCpu(bx, [=, &sum_integrand_tensor_comp_local] (int i, int j, int k) noexcept
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
                sum_integrand_tensor_comp_local[0][0] += (1.0 - grad_chi_x[0]); 
                sum_integrand_tensor_comp_local[0][1] += (    - grad_chi_y[0]); 
                sum_integrand_tensor_comp_local[1][0] += (    - grad_chi_x[1]); 
                sum_integrand_tensor_comp_local[1][1] += (1.0 - grad_chi_y[1]); 

                if (AMREX_SPACEDIM == 3) {
                    sum_integrand_tensor_comp_local[0][2] += (    - grad_chi_z[0]); 
                    sum_integrand_tensor_comp_local[2][0] += (    - grad_chi_x[2]); 
                    sum_integrand_tensor_comp_local[1][2] += (    - grad_chi_z[1]); 
                    sum_integrand_tensor_comp_local[2][1] += (    - grad_chi_y[2]); 
                    sum_integrand_tensor_comp_local[2][2] += (1.0 - grad_chi_z[2]); 
                }
            }
        });
    }
    
    for (int r = 0; r < AMREX_SPACEDIM; ++r) {
        for (int c = 0; c < AMREX_SPACEDIM; ++c) {
            amrex::ParallelDescriptor::ReduceRealSum(sum_integrand_tensor_comp_local[r][c]);
        }
    }

    amrex::Long N_total_cells_in_domain = geom.Domain().numPts(); 
    if (N_total_cells_in_domain > 0) {
        for (int l_idx = 0; l_idx < AMREX_SPACEDIM; ++l_idx) {
            for (int m_idx = 0; m_idx < AMREX_SPACEDIM; ++m_idx) {
                Deff_tensor[l_idx][m_idx] = sum_integrand_tensor_comp_local[l_idx][m_idx] / static_cast<amrex::Real>(N_total_cells_in_domain);
            }
        }
    } else {
         if (amrex::ParallelDescriptor::IOProcessor() && verbose_level > 0) {
            amrex::Warning("Total cells in domain/REV is zero, D_eff cannot be calculated.");
         }
    }

     if (verbose_level > 1 && amrex::ParallelDescriptor::IOProcessor()) {
         amrex::Print() << "  [calc_Deff] Raw summed (1-dchi_x_dx): " << sum_integrand_tensor_comp_local[0][0]
                        << ", N_total_cells_in_domain: " << N_total_cells_in_domain << std::endl;
     }
}
} // end anonymous namespace


int main (int argc, char* argv[])
{
    HYPRE_Init(); 
    amrex::Initialize(argc, argv);
    {
        amrex::Real master_strt_time = amrex::second();

        std::string main_filename;
        std::string main_data_path_str = "./data/";    
        std::string main_results_path_str = "./results_diffusion/"; 
        std::string main_hdf5_dataset = "image";      
        amrex::Real main_threshold_val = 0.5;
        int main_phase_id_analysis = 1; 
        std::string main_solver_str = "FlexGMRES";
        int main_box_size = 32;
        int main_verbose = 1;
        int main_write_plotfile_full = 0; 
        std::string main_calculation_method = "homogenization"; 
        std::string output_filename = "results.txt"; // Added this line


        bool rev_do_study = false;
        int rev_num_samples = 3;
        std::string rev_sizes_str = "32 64 96"; 
        std::string rev_solver_str = "FlexGMRES";
        std::string rev_results_filename = "rev_study_Deff.csv";
        int rev_write_plotfiles = 0; 
        int rev_verbose_level = 1;   

        { 
            amrex::ParmParse pp; 
            pp.get("filename", main_filename);
            pp.query("data_path", main_data_path_str);
            pp.query("results_path", main_results_path_str);
            pp.query("hdf5_dataset", main_hdf5_dataset);
            pp.query("threshold_val", main_threshold_val);
            pp.query("phase_id", main_phase_id_analysis); 
            pp.query("solver_type", main_solver_str); 
            pp.query("box_size", main_box_size);
            pp.query("verbose", main_verbose);
            pp.query("write_plotfile", main_write_plotfile_full);
            pp.query("calculation_method", main_calculation_method);
            pp.query("output_filename", output_filename); // Added this line

            amrex::ParmParse ppr("rev"); 
            ppr.query("do_study", rev_do_study);
            ppr.query("num_samples", rev_num_samples);
            ppr.query("sizes", rev_sizes_str);
            ppr.query("solver_type", rev_solver_str); 
            ppr.query("results_file", rev_results_filename);
            ppr.query("write_plotfiles", rev_write_plotfiles);
            ppr.query("verbose", rev_verbose_level);
        }

        std::filesystem::path main_data_path(main_data_path_str);
        std::filesystem::path main_results_path(main_results_path_str);
        
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (!std::filesystem::exists(main_results_path)) {
                std::filesystem::create_directories(main_results_path);
                 if (main_verbose >=1 ) amrex::Print() << "Created results directory: " << main_results_path.string() << std::endl;
            }
        }
        amrex::ParallelDescriptor::Barrier(); 
        std::filesystem::path full_input_path = main_data_path / main_filename;


        amrex::Geometry geom_full;
        amrex::BoxArray ba_full;
        amrex::DistributionMapping dm_full;
        amrex::iMultiFab mf_phase_full; 
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
            
            int reader_phase_active = main_phase_id_analysis;
            int reader_phase_inactive = (main_phase_id_analysis == 0 ? 1 : 0);

            if (ext == ".tif" || ext == ".tiff") {
                OpenImpala::TiffReader reader(full_input_path.string());
                if (!reader.isRead()) throw std::runtime_error("TiffReader failed to read metadata.");
                domain_box_full = reader.box();
                ba_full.define(domain_box_full);
                ba_full.maxSize(main_box_size);
                dm_full.define(ba_full);
                mf_phase_full.define(ba_full, dm_full, 1, 1); 
                amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
                reader.threshold(main_threshold_val, reader_phase_active, reader_phase_inactive, mf_temp_no_ghost);
                amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0,0,1,0);

            } else if (ext == ".dat") {
                OpenImpala::DatReader reader(full_input_path.string());
                 if (!reader.isRead()) throw std::runtime_error("DatReader failed to read metadata.");
                domain_box_full = reader.box();
                ba_full.define(domain_box_full);
                ba_full.maxSize(main_box_size);
                dm_full.define(ba_full);
                mf_phase_full.define(ba_full, dm_full, 1, 1);
                amrex::iMultiFab mf_temp_no_ghost(ba_full, dm_full, 1, 0);
                reader.threshold(main_threshold_val, reader_phase_active, reader_phase_inactive, mf_temp_no_ghost);
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
                reader.threshold(main_threshold_val, reader_phase_active, reader_phase_inactive, mf_temp_no_ghost);
                amrex::Copy(mf_phase_full, mf_temp_no_ghost, 0,0,1,0);
            }
            else {
                throw std::runtime_error("Unsupported file extension for full domain load: " + ext);
            }

            amrex::RealBox rb_full({AMREX_D_DECL(0.0,0.0,0.0)},
                                   {AMREX_D_DECL(amrex::Real(domain_box_full.length(0)),
                                                 amrex::Real(domain_box_full.length(1)),
                                                 amrex::Real(domain_box_full.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic_full;
            is_periodic_full = {AMREX_D_DECL(1,1,1)}; 
            geom_full.define(domain_box_full, &rb_full, 0, is_periodic_full.data());
            mf_phase_full.FillBoundary(geom_full.periodicity());

        } catch (const std::exception& e) {
            amrex::Print() << "Error loading full domain data: " << e.what() << std::endl;
            amrex::Abort("Full domain data loading failed.");
        }


        if (rev_do_study) {
            if (main_verbose >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "\n--- Starting REV Study (Homogenization Method) for Phase ID " << main_phase_id_analysis << " ---\n";
                amrex::Print() << "  Number of samples per size: " << rev_num_samples << std::endl;
                amrex::Print() << "  Target REV sizes: " << rev_sizes_str << std::endl;
                amrex::Print() << "  REV Solver: " << rev_solver_str << std::endl;
                amrex::Print() << "  REV Plotfiles: " << (rev_write_plotfiles ? "Yes" : "No") << std::endl;
            }

            std::vector<int> rev_actual_sizes_vec; 
            std::stringstream ss_rev_sizes(rev_sizes_str);
            int size_val_loop; 
            while (ss_rev_sizes >> size_val_loop) rev_actual_sizes_vec.push_back(size_val_loop);

            if (rev_actual_sizes_vec.empty()) {
                amrex::Warning("REV sizes string is empty or invalid. Skipping REV study.");
            } else {
                std::ofstream rev_csv_file;
                std::filesystem::path full_rev_csv_path = main_results_path / rev_results_filename;
                if (amrex::ParallelDescriptor::IOProcessor()) {
                    rev_csv_file.open(full_rev_csv_path.string());
                    rev_csv_file << "SampleNo,SeedX,SeedY,SeedZ,REV_Size_Target,ActualSizeX,ActualSizeY,ActualSizeZ,D_xx,D_yy,D_zz,D_xy,D_xz,D_yz\n";
                }

                std::mt19937 gen(amrex::ParallelDescriptor::MyProc() + 12345 + rev_num_samples); 

                for (int s_idx = 0; s_idx < rev_num_samples; ++s_idx) {
                    for (int current_rev_size_target : rev_actual_sizes_vec) {
                        amrex::IntVect seed_lo_global; 
                        for(int d=0; d<AMREX_SPACEDIM; ++d) {
                            int min_coord = domain_box_full.smallEnd(d);
                            int max_coord = domain_box_full.bigEnd(d) - (current_rev_size_target -1) ;
                            if (min_coord > max_coord || current_rev_size_target > domain_box_full.length(d)) { 
                                seed_lo_global[d] = domain_box_full.smallEnd(d); 
                            } else {
                                std::uniform_int_distribution<> distr(min_coord, max_coord);
                                seed_lo_global[d] = distr(gen);
                            }
                        }
                        
                        amrex::Box bx_rev_global_const = amrex::Box(seed_lo_global, seed_lo_global + amrex::IntVect(current_rev_size_target - 1));
                        amrex::Box bx_rev_global = bx_rev_global_const; 
                        bx_rev_global &= domain_box_full; 

                        if (bx_rev_global.isEmpty() || bx_rev_global.longside() < 8) { 
                            if (rev_verbose_level >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                                std::stringstream ss; ss << bx_rev_global;
                                amrex::Warning("Skipping REV for sample " + std::to_string(s_idx+1) + 
                                               " target size " + std::to_string(current_rev_size_target) + 
                                               " due to small/empty box after intersection: " + ss.str());
                            }
                            continue;
                        }
                         if (rev_verbose_level >=1 && amrex::ParallelDescriptor::IOProcessor()) {
                            amrex::Print() << " REV Sample " << s_idx + 1 << ", Target Size " << current_rev_size_target 
                                           << ", Seed Lo (global): " << seed_lo_global 
                                           << ", Actual REV Box (global): " << bx_rev_global << std::endl;
                        }
                        
                        amrex::Box domain_rev_relative_temp = bx_rev_global; 
                        amrex::Box domain_rev_relative = domain_rev_relative_temp; // Make a copy
                        domain_rev_relative.shift(-bx_rev_global.smallEnd()); // Shift the copy
                        
                        amrex::Geometry geom_rev;
                        amrex::RealBox rb_rev({AMREX_D_DECL(0.0,0.0,0.0)},
                                              {AMREX_D_DECL(amrex::Real(domain_rev_relative.length(0)), 
                                                            amrex::Real(domain_rev_relative.length(1)), 
                                                            amrex::Real(domain_rev_relative.length(2)))});
                        amrex::Array<int,AMREX_SPACEDIM> is_periodic_rev = {AMREX_D_DECL(1,1,1)}; 
                        geom_rev.define(domain_rev_relative, &rb_rev, 0, is_periodic_rev.data()); 

                        amrex::BoxArray ba_rev_relative(domain_rev_relative); 
                        ba_rev_relative.maxSize(main_box_size); 
                        amrex::DistributionMapping dm_rev_relative(ba_rev_relative);
                        amrex::iMultiFab mf_phase_rev(ba_rev_relative, dm_rev_relative, 1, 1); 
                        
                        amrex::BoxArray ba_temp_global(bx_rev_global);
                        amrex::DistributionMapping dm_temp_global(ba_temp_global); 
                        amrex::iMultiFab mf_temp_global(ba_temp_global, dm_temp_global, 1, 0); 
                        mf_temp_global.ParallelCopy(mf_phase_full, 0, 0, 1, amrex::IntVect::TheZeroVector(), amrex::IntVect::TheZeroVector(), geom_full.periodicity());
                        
                        mf_phase_rev.setVal(0); 
                        for(amrex::MFIter mfi_dest(mf_phase_rev); mfi_dest.isValid(); ++mfi_dest) {
                            amrex::IArrayBox& dest_fab = mf_phase_rev[mfi_dest];
                            const amrex::Box& dest_fab_box_local = dest_fab.box(); 

                            amrex::Box temp_dest_box_global = dest_fab_box_local; // Non-const copy
                            amrex::Box required_src_box_global = temp_dest_box_global.shift(bx_rev_global.smallEnd());
                            
                            for (amrex::MFIter mfi_src(mf_temp_global); mfi_src.isValid(); ++mfi_src) { // Should be 1 FAB for np=1
                                const amrex::IArrayBox& src_fab = mf_temp_global[mfi_src];
                                const amrex::Box& src_fab_box_global_const = src_fab.box(); 
                                amrex::Box src_fab_box_global = src_fab_box_global_const; // Non-const copy
                                
                                amrex::Box copy_region_global = required_src_box_global & src_fab_box_global;

                                if (!copy_region_global.isEmpty()) {
                                    amrex::Box temp_copy_region_dest_local = copy_region_global; // Non-const copy
                                    amrex::Box copy_region_dest_local = temp_copy_region_dest_local.shift(-bx_rev_global.smallEnd());
                                    dest_fab.copy(src_fab, copy_region_global, 0, copy_region_dest_local, 0, 1);
                                }
                            }
                        }
                        mf_phase_rev.FillBoundary(geom_rev.periodicity()); 


                        amrex::MultiFab mf_chi_x_rev(ba_rev_relative, dm_rev_relative, 1, 1);
                        amrex::MultiFab mf_chi_y_rev(ba_rev_relative, dm_rev_relative, 1, 1);
                        amrex::MultiFab mf_chi_z_rev;
                        if (AMREX_SPACEDIM==3) mf_chi_z_rev.define(ba_rev_relative, dm_rev_relative, 1, 1);

                        bool all_chi_rev_converged = true;
                        std::vector<OpenImpala::Direction> rev_solve_dirs = {OpenImpala::Direction::X, OpenImpala::Direction::Y};
                        if (AMREX_SPACEDIM == 3) rev_solve_dirs.push_back(OpenImpala::Direction::Z);
                        OpenImpala::EffectiveDiffusivityHypre::SolverType rev_solver_type_enum = stringToSolverType(rev_solver_str);

                        for (const auto& dir_k_solve : rev_solve_dirs) {
                            std::string chi_plot_rev_subdir_str = "REV_Sample" + std::to_string(s_idx+1) + 
                                                                  "_Size" + std::to_string(bx_rev_global.length(0)) + 
                                                                  "_Dir" + std::to_string(static_cast<int>(dir_k_solve));
                            std::filesystem::path chi_plot_rev_full_path = main_results_path / chi_plot_rev_subdir_str;
                            
                            if (rev_write_plotfiles != 0 && amrex::ParallelDescriptor::IOProcessor()) {
                                 amrex::UtilCreateDirectory(chi_plot_rev_full_path.string(), 0755);
                            }
                             amrex::ParallelDescriptor::Barrier(); 

                            std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> rev_chi_solver;
                            try {
                                 rev_chi_solver = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                                    geom_rev, ba_rev_relative, dm_rev_relative, mf_phase_rev, // use ba_rev_relative and dm_rev_relative
                                    main_phase_id_analysis, dir_k_solve, rev_solver_type_enum,
                                    chi_plot_rev_full_path.string(), rev_verbose_level, (rev_write_plotfiles !=0)
                                );
                                if (!rev_chi_solver->solve()) {
                                    all_chi_rev_converged = false; 
                                    if(rev_verbose_level >= 1 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    REV Chi solve FAILED for dir " << static_cast<int>(dir_k_solve) << std::endl;
                                    break;
                                }
                                if (dir_k_solve == OpenImpala::Direction::X) rev_chi_solver->getChiSolution(mf_chi_x_rev);
                                else if (dir_k_solve == OpenImpala::Direction::Y) rev_chi_solver->getChiSolution(mf_chi_y_rev);
                                else if (AMREX_SPACEDIM==3 && dir_k_solve == OpenImpala::Direction::Z) rev_chi_solver->getChiSolution(mf_chi_z_rev);
                            } catch (const std::exception& e) {
                                if(rev_verbose_level >= 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    REV Chi solve EXCEPTION for dir " << static_cast<int>(dir_k_solve) << ": " << e.what() << std::endl;
                                all_chi_rev_converged = false; break;
                            }
                        }

                        amrex::Real Deff_tensor_rev[AMREX_SPACEDIM][AMREX_SPACEDIM];
                        for(int r=0; r<AMREX_SPACEDIM; ++r) for(int c=0; c<AMREX_SPACEDIM; ++c) Deff_tensor_rev[r][c] = std::numeric_limits<amrex::Real>::quiet_NaN();

                        if (all_chi_rev_converged) {
                            amrex::iMultiFab active_mask_rev(ba_rev_relative, dm_rev_relative, 1, 0); // use ba_rev_relative
                            #ifdef AMREX_USE_OMP
                            #pragma omp parallel if(amrex::Gpu::notInLaunchRegion())
                            #endif
                            for (amrex::MFIter mfi(active_mask_rev, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi){
                                const amrex::Box& tb = mfi.tilebox(); 
                                auto const& mask_arr = active_mask_rev.array(mfi);
                                auto const& phase_arr = mf_phase_rev.const_array(mfi); 
                                amrex::LoopOnCpu(tb, [=] (int i, int j, int k) { 
                                    mask_arr(i,j,k,0) = (phase_arr(i,j,k,0) == main_phase_id_analysis) ? 1 : 0;
                                });
                            }
                            calculate_Deff_tensor_homogenization(Deff_tensor_rev,
                                                                 mf_chi_x_rev, mf_chi_y_rev, mf_chi_z_rev,
                                                                 active_mask_rev, geom_rev, rev_verbose_level);
                        }
                        if (amrex::ParallelDescriptor::IOProcessor()) {
                            rev_csv_file << s_idx+1 << ","
                                         << seed_lo_global[0] << "," << seed_lo_global[1] << "," 
                                         << (AMREX_SPACEDIM == 3 ? seed_lo_global[2] : 0) << ","
                                         << current_rev_size_target << "," 
                                         << bx_rev_global.length(0) << "," << bx_rev_global.length(1) << "," 
                                         << (AMREX_SPACEDIM == 3 ? bx_rev_global.length(2) : 1) << "," 
                                         << std::fixed << std::setprecision(8)
                                         << Deff_tensor_rev[0][0] << "," << Deff_tensor_rev[1][1] << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[2][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                         << Deff_tensor_rev[0][1] << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[0][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << ","
                                         << (AMREX_SPACEDIM == 3 ? Deff_tensor_rev[1][2] : std::numeric_limits<amrex::Real>::quiet_NaN()) << "\n";
                            rev_csv_file.flush(); 
                        }
                    } 
                } 
                if (amrex::ParallelDescriptor::IOProcessor()) rev_csv_file.close();
            } 
        } 


        if (!rev_do_study || main_calculation_method != "skip_if_rev") { 
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

                auto solver_type_effdiff_full = stringToSolverType(main_solver_str);

                for (const auto& dir_k : dirs_to_solve_chi_full) {
                    std::string dir_k_str = (dir_k == OpenImpala::Direction::X) ? "X" : (dir_k == OpenImpala::Direction::Y) ? "Y" : "Z";
                    if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor())
                        amrex::Print() << "\n--- Solving for Full Domain chi_" << dir_k_str << " ---\n";
                    
                    std::filesystem::path full_chi_plot_dir_path = main_results_path / ("FullDomain_chi_" + dir_k_str);
                    if (main_write_plotfile_full !=0 && amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::UtilCreateDirectory(full_chi_plot_dir_path.string(), 0755);
                    }
                    amrex::ParallelDescriptor::Barrier();

                    std::unique_ptr<OpenImpala::EffectiveDiffusivityHypre> solver_chi_k_full;
                     try {
                        solver_chi_k_full = std::make_unique<OpenImpala::EffectiveDiffusivityHypre>(
                            geom_full, ba_full, dm_full, mf_phase_full, 
                            main_phase_id_analysis, dir_k, solver_type_effdiff_full, 
                            full_chi_plot_dir_path.string(), 
                            main_verbose, (main_write_plotfile_full != 0)
                        );
                        if (!solver_chi_k_full->solve()) {
                            all_chi_full_converged = false;
                            break; 
                        }
                        if (dir_k == OpenImpala::Direction::X) solver_chi_k_full->getChiSolution(mf_chi_x_full);
                        else if (dir_k == OpenImpala::Direction::Y) solver_chi_k_full->getChiSolution(mf_chi_y_full);
                        else if (AMREX_SPACEDIM == 3 && dir_k == OpenImpala::Direction::Z) solver_chi_k_full->getChiSolution(mf_chi_z_full);
                    } catch (const std::exception& e_full) {
                         if(main_verbose >= 0 && amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "    Full Domain Chi solve EXCEPTION for dir " << static_cast<int>(dir_k) << ": " << e_full.what() << std::endl;
                        all_chi_full_converged = false; break;
                    }
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
                            mask_arr(i,j,k,0) = (phase_arr(i,j,k,0) == main_phase_id_analysis) ? 1 : 0;
                        });
                    }
                    calculate_Deff_tensor_homogenization(
                        Deff_tensor_full, mf_chi_x_full, mf_chi_y_full, mf_chi_z_full,
                        active_mask_full, geom_full, main_verbose);

                    if (amrex::ParallelDescriptor::IOProcessor()) {
                        amrex::Print() << "Full Domain Effective Diffusivity Tensor D_eff / D_material:\n";
                        for (int r_print = 0; r_print < AMREX_SPACEDIM; ++r_print) { 
                            amrex::Print() << "  [";
                            for (int c_print = 0; c_print < AMREX_SPACEDIM; ++c_print) { 
                                amrex::Print() << std::scientific << std::setprecision(8) << Deff_tensor_full[r_print][c_print]
                                               << (c_print == AMREX_SPACEDIM - 1 ? "" : ", ");
                            }
                            amrex::Print() << "]\n";
                        }
                    }
                } else {
                    if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "Full domain D_eff calculation skipped due to chi_k non-convergence.\n";
                }

           } else if (main_calculation_method == "flow_through") {
    // ===================================================================================
    // --- Tortuosity via Flow-Through (Laplacian Solve) - Implemented Logic ---
    // This block calculates tortuosity by solving a potential equation (Laplace's)
    // with Dirichlet boundary conditions on inlet/outlet faces and Neumann (no-flux)
    // conditions on the side walls. It uses the TortuosityHypre class.
    // ===================================================================================

    if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "\n--- Full Domain Calculation: Tortuosity via Flow-Through ---\n";
    }

    // --- Get Tortuosity-Specific Parameters ---
    // Set default boundary condition values for the potential field
    amrex::Real vlo = -1.0;
    amrex::Real vhi = 1.0;

    // Allow users to override these defaults from a [tortuosity] block in the inputs file
    amrex::ParmParse pp_tort("tortuosity");
    pp_tort.query("vlo", vlo);
    pp_tort.query("vhi", vhi);

    // --- Calculate Volume Fraction (Prerequisite for Tortuosity) ---
    if (main_verbose > 0) amrex::Print() << "Calculating Volume Fraction for Phase ID: " << main_phase_id_analysis << "\n";
    OpenImpala::VolumeFraction vf_calc(mf_phase_full, main_phase_id_analysis);

    // FIX 1: The 'value()' method was called without arguments.
    // The correct function signature is 'void value(long long&, long long&, bool)',
    // which requires variables to be passed by reference to store the results.
    long long phase_voxels = 0;
    long long total_voxels = 0;
    vf_calc.value(phase_voxels, total_voxels, false); // Call with the required arguments
    amrex::Real volume_fraction = (total_voxels > 0) ? (static_cast<amrex::Real>(phase_voxels) / static_cast<amrex::Real>(total_voxels)) : 0.0;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "  Volume Fraction = " << std::fixed << std::setprecision(8) << volume_fraction << "\n";
    }

    // --- Parse Directions to Run ---
    std::map<std::string, amrex::Real> tortuosity_results;
    std::string direction_str;
    amrex::ParmParse pp; // Top-level ParmParse to get "direction"
    pp.get("direction", direction_str);
    std::string upper_direction_str = direction_str;
    std::transform(upper_direction_str.begin(), upper_direction_str.end(), upper_direction_str.begin(), ::toupper);

    std::vector<OpenImpala::Direction> directions_to_run;
    if (upper_direction_str.find("ALL") != std::string::npos) {
        directions_to_run = {OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
    } else {
        std::stringstream ss(upper_direction_str);
        std::string single_dir;
        while (ss >> single_dir) {
            if (single_dir == "X") directions_to_run.push_back(OpenImpala::Direction::X);
            else if (single_dir == "Y") directions_to_run.push_back(OpenImpala::Direction::Y);
            else if (single_dir == "Z") directions_to_run.push_back(OpenImpala::Direction::Z);
        }
    }

    if (directions_to_run.empty()) {
        amrex::Warning("No valid directions specified in 'direction' input. Skipping tortuosity calculation.");
    }

    // --- Main Calculation Loop for Each Direction ---
    for (const auto& dir : directions_to_run) {
        std::string dir_char = (dir == OpenImpala::Direction::X) ? "X" : (dir == OpenImpala::Direction::Y) ? "Y" : "Z";
        if (main_verbose >= 1 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Solving for Tortuosity in Direction: " << dir_char << " ---\n";
        }

        // FIX 2: The 'stringToSolverType' function returns an 'EffectiveDiffusivityHypre::SolverType',
        // but the TortuosityHypre constructor requires a 'TortuosityHypre::SolverType'.
        // We must cast the enum to the correct type.
        auto solver_type_enum_effdiff = stringToSolverType(main_solver_str);
        auto solver_type_enum = static_cast<OpenImpala::TortuosityHypre::SolverType>(solver_type_enum_effdiff);

        // Create the main solver object for tortuosity
        OpenImpala::TortuosityHypre tort_solver(
            geom_full,
            ba_full,
            dm_full,
            mf_phase_full,
            volume_fraction,
            main_phase_id_analysis,
            dir,
            solver_type_enum, // Pass the correctly typed enum
            main_results_path_str,
            vlo,
            vhi,
            main_verbose,
            (main_write_plotfile_full != 0)
        );

        amrex::Real tau = tort_solver.value();

        tortuosity_results["Tortuosity_" + dir_char] = tau;

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  >>> Calculated Tortuosity (" << dir_char << "): " << std::fixed << std::setprecision(8) << tau << " <<<\n";
        }
    }

    // --- Write Final Summary Results to File ---
    if (amrex::ParallelDescriptor::IOProcessor()) {
        // FIX 3: 'output_filename' was not declared. I added its declaration
        // and ParmParse query at the top of the main() function.
        std::string output_filename = "results.txt"; // Default value
        pp.query("output_filename", output_filename); // Read from inputs

        std::filesystem::path output_filepath = std::filesystem::path(main_results_path_str) / output_filename;
        amrex::Print() << "\nWriting final results to: " << output_filepath << "\n";

        std::ofstream outfile(output_filepath);
        if (outfile.is_open()) {
            outfile << "# Tortuosity Calculation Results (Flow-Through Method)\n";
            outfile << "# Input File: " << main_filename << "\n";
            outfile << "# Analysis Phase ID: " << main_phase_id_analysis << "\n";
            outfile << "# -----------------------------\n";
            outfile << "VolumeFraction: " << std::fixed << std::setprecision(9) << volume_fraction << "\n";
            for (const auto& pair : tortuosity_results) {
                outfile << pair.first << ": " << std::fixed << std::setprecision(9) << pair.second << "\n";
            }
            outfile.close();
        } else {
            amrex::Warning("Could not open output file for writing: " + output_filepath.string());
        }
    }
}
        }


        amrex::Real master_stop_time = amrex::second() - master_strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(master_stop_time,amrex::ParallelDescriptor::IOProcessorNumber());
        if (amrex::ParallelDescriptor::IOProcessor())
            amrex::Print() << std::endl << "Total run time (seconds) = " << master_stop_time << std::endl;

    }
    amrex::Finalize();
    HYPRE_Finalize(); 
    return 0;
}




