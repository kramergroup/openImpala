#include <cmath>
#include <cstdlib> // Prefer over <stdlib.h>
#include <filesystem> // Requires C++17
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory> // For std::unique_ptr
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <algorithm> // Needed for std::transform

// AMReX includes
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H> // Potentially needed by TortuosityHypre
#include <AMReX_iMultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_RealBox.H>
#include <AMReX_Utility.H> // For amrex::UtilCreateDirectory

// OpenImpala includes (assuming namespace OpenImpala)
#include "../io/DatReader.H"      // Assuming DatReader.H includes needed types like RawDataType, parseRawDataType
#include "../io/HDF5Reader.H"
#include "../io/TiffReader.H"
// #include "../io/TiffStackReader.H" // REMOVED - Deprecated
#include "TortuosityHypre.H"      // Assuming TortuosityHypre.H includes nested SolverType enum
#include "VolumeFraction.H"

// Assume these enums exist in OpenImpala namespace
namespace OpenImpala {
    enum class Direction { X = 0, Y = 1, Z = 2 };
    // Namespace TortuosityHypre { // Assuming SolverType is nested
    //     enum class SolverType { FlexGMRES /*, other solvers */ };
    // }
} // namespace OpenImpala

namespace // Anonymous namespace for helpers
{
    // Helper to convert string ("X", "Y", "Z") to Direction enum
    // Returns Direction::X if input is invalid, prints warning.
    OpenImpala::Direction string_to_direction(const std::string& s) {
        std::string upper_s = s;
        std::transform(upper_s.begin(), upper_s.end(), upper_s.begin(), ::toupper);
        if (upper_s == "X") { return OpenImpala::Direction::X; }
        if (upper_s == "Y") { return OpenImpala::Direction::Y; }
        if (upper_s == "Z") { return OpenImpala::Direction::Z; }
        amrex::Warning("Invalid direction string '" + s + "', defaulting to X.");
        return OpenImpala::Direction::X;
    }

    // Helper to get string representation of Direction
    std::string direction_to_string(OpenImpala::Direction dir) {
        switch (dir) {
            case OpenImpala::Direction::X: return "X";
            case OpenImpala::Direction::Y: return "Y";
            case OpenImpala::Direction::Z: return "Z";
            default: return "Unknown";
        }
    }

    // Helper to convert string to SolverType (Example, needs actual enum)
    OpenImpala::TortuosityHypre::SolverType string_to_solver(const std::string& s) {
        std::string upper_s = s;
        std::transform(upper_s.begin(), upper_s.end(), upper_s.begin(), ::toupper);
        if (upper_s == "FLEXGMRES") { return OpenImpala::TortuosityHypre::SolverType::FlexGMRES; }
        // Add other solver types here...
        amrex::Warning("Invalid solver type string '" + s + "', defaulting to FlexGMRES.");
        return OpenImpala::TortuosityHypre::SolverType::FlexGMRES;
    }

} // namespace

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    { // Start AMReX scope
        amrex::Real strt_time = amrex::second();

        // --- Parameters ---
        amrex::ParmParse pp; // Default ParmParse constructor

        // Input File Parameters
        std::string filename;
        pp.get("filename", filename); // Input image filename (required)

        std::string data_path_str = "."; // Default to current directory
        pp.query("data_path", data_path_str); // Optional path prefix for input file

        std::string hdf5_dataset; // Optional: Path within HDF5 file
        pp.query("hdf5_dataset", hdf5_dataset);

        // int tiff_stack_size = 0; // REMOVED - TiffStackReader deprecated
        // pp.query("tiff_stack_size", tiff_stack_size); // REMOVED

        // RAW/DAT Reader Specific (Required if using these types)
        int raw_width = 0, raw_height = 0, raw_depth = 0;
        std::string raw_datatype_str;
        pp.query("raw_width", raw_width);
        pp.query("raw_height", raw_height);
        pp.query("raw_depth", raw_depth);
        pp.query("raw_datatype", raw_datatype_str); // e.g., "uint8", "float32"

        // Analysis Parameters
        int phase_id = 1; // Phase ID to calculate VF and Tortuosity for (default: 1)
        pp.query("phase_id", phase_id);

        double threshold_value = 127.5; // Default threshold (suitable for uint8 0-255)
        pp.query("threshold_value", threshold_value);

        std::string direction_str = "All"; // Direction(s) to compute: "X", "Y", "Z", "All"
        pp.query("direction", direction_str);

        // Solver Parameters
        std::string solver_type_str = "FlexGMRES";
        pp.query("solver_type", solver_type_str);
        auto solver_type = string_to_solver(solver_type_str);

        double hypre_eps = 1e-9; // Solver tolerance
        pp.query("hypre_eps", hypre_eps);
        int hypre_maxiter = 200; // Solver max iterations
        pp.query("hypre_maxiter", hypre_maxiter);

        // Grid Parameters
        int box_size = 32;
        pp.query("box_size", box_size);

        // Output Parameters
        std::string results_dir_str = "DiffusionResults";
        pp.query("results_dir", results_dir_str);
        std::string output_filename = "diffusion_results.txt";
        pp.query("output_filename", output_filename);
        int write_plotfile = 0; // Flag to write Hypre plotfiles (0=no, 1=yes)
        pp.query("write_plotfile", write_plotfile);

        // Control Parameters
        int verbose = 1;
        pp.query("verbose", verbose);

        // --- Path Handling ---
        std::filesystem::path data_path = data_path_str;
        std::filesystem::path results_dir = results_dir_str;

        // Handle '~' expansion for results directory
        if (!results_dir.empty() && results_dir.string().front() == '~') {
            const char* homeDir = getenv("HOME");
            if (homeDir == nullptr) {
                amrex::Warning("Could not get HOME directory; cannot expand '~' in results_dir.");
                // Decide default behavior: use relative path or abort? Using relative for now.
                results_dir = results_dir.string().substr(1); // Remove leading '~/' if present
                 if (!results_dir.empty() && (results_dir.string().front() == '/' || results_dir.string().front() == '\\')) {
                    results_dir = results_dir.string().substr(1); // Remove leading slash if present after removing ~
                }
            } else {
                std::string subpath = results_dir.string().substr(1);
                if (!subpath.empty() && (subpath.front() == '/' || subpath.front() == '\\')) {
                     subpath = subpath.substr(1); // Remove leading slash if present after removing ~
                 }
                results_dir = std::filesystem::path(homeDir) / subpath;
            }
        }
        // Handle '~' for data path similarly if needed
         if (!data_path.empty() && data_path.string().front() == '~') {
             const char* homeDir = getenv("HOME");
             if (homeDir == nullptr) {
                 amrex::Warning("Could not get HOME directory; cannot expand '~' in data_path.");
                 data_path = data_path.string().substr(1); // Remove leading '~/' if present
                 if (!data_path.empty() && (data_path.string().front() == '/' || data_path.string().front() == '\\')) {
                     data_path = data_path.string().substr(1); // Remove leading slash if present after removing ~
                 }
             } else {
                std::string subpath = data_path.string().substr(1);
                if (!subpath.empty() && (subpath.front() == '/' || subpath.front() == '\\')) {
                     subpath = subpath.substr(1); // Remove leading slash if present after removing ~
                 }
                 data_path = std::filesystem::path(homeDir) / subpath;
             }
         }


        std::filesystem::path full_input_path = data_path / filename;

        // Create results directory if it doesn't exist
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (!std::filesystem::exists(results_dir)) {
                if (verbose > 0) amrex::Print() << " Creating results directory: " << results_dir << "\n";
                if (!amrex::UtilCreateDirectory(results_dir.string(), 0755)) {
                    amrex::Warning("Could not create results directory: " + results_dir.string());
                    // Continue, but output file writing might fail
                }
            }
        }
        amrex::ParallelDescriptor::Barrier(); // Ensure directory exists before proceeding


        if (verbose > 0 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- Diffusion Calculation Setup ---\n";
            amrex::Print() << " Input File:         " << full_input_path << "\n";
            if (!hdf5_dataset.empty()) amrex::Print() << " HDF5 Dataset:       " << hdf5_dataset << "\n";
            // if (tiff_stack_size > 0) amrex::Print() << " TIFF Stack Size:    " << tiff_stack_size << "\n"; // REMOVED
            amrex::Print() << " Analysis Phase ID:  " << phase_id << "\n";
            amrex::Print() << " Threshold Value:    " << threshold_value << "\n";
            amrex::Print() << " Direction(s):       " << direction_str << "\n";
            amrex::Print() << " Results Directory:  " << results_dir << "\n";
            amrex::Print() << " Output Filename:    " << output_filename << "\n";
            amrex::Print() << " Max Grid Size:      " << box_size << "\n";
            amrex::Print() << " Solver:             " << solver_type_str << "\n";
            amrex::Print() << " Solver Tol:         " << hypre_eps << "\n";
            amrex::Print() << " Solver MaxIter:     " << hypre_maxiter << "\n";
            amrex::Print() << "-----------------------------------\n\n";
        }

        // --- File Reading and AMReX Grid Setup ---
        amrex::Geometry geom;
        amrex::BoxArray ba;
        amrex::DistributionMapping dm;
        amrex::iMultiFab mf_phase;
        bool reader_success = false;
        amrex::Box domain_box;

        try {
            std::string ext;
            if (filename.find('.') != std::string::npos) {
                ext = filename.substr(filename.find_last_of('.'));
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            }

            if (ext == ".tif" || ext == ".tiff") {
                if (verbose > 0) amrex::Print() << " Reading TIFF file: " << full_input_path << "\n";
                // Assumes TiffReader can handle multi-directory TIFFs for stacks
                OpenImpala::TiffReader reader(full_input_path.string());
                if (!reader.isRead()) throw std::runtime_error("TiffReader failed to read file.");
                domain_box = reader.box();
                reader_success = true;
                // Define mf_phase now *before* reader goes out of scope if needed
                 ba.define(domain_box);
                 ba.maxSize(box_size);
                 dm.define(ba);
                 mf_phase.define(ba, dm, 1, 1); // 1 ghost cell for solver
                 reader.threshold(threshold_value, phase_id, (phase_id == 0 ? 1 : 0), mf_phase); // Assign phase_id to > T

            } else if (ext == ".dat") { // Separated RAW for clarity if needed later
                 if (verbose > 0) amrex::Print() << " Reading DAT file: " << full_input_path << "\n";
                 // For DAT, dimensions MUST be provided via file header (handled by reader)
                 // Type is also fixed by DataType definition
                 OpenImpala::DatReader reader(full_input_path.string());
                 if (!reader.isRead()) throw std::runtime_error("DatReader failed to read file.");
                 domain_box = reader.box();
                 reader_success = true;
                 // Define and threshold mf_phase
                 ba.define(domain_box);
                 ba.maxSize(box_size);
                 dm.define(ba);
                 mf_phase.define(ba, dm, 1, 1);
                 // Assuming DatReader's threshold uses its internal DataType
                 reader.threshold(static_cast<OpenImpala::DatReader::DataType>(threshold_value), phase_id, (phase_id == 0 ? 1 : 0), mf_phase);

            } else if (ext == ".raw") { // Example: Handle raw if needed distinctly
                 if (verbose > 0) amrex::Print() << " Reading RAW file: " << full_input_path << "\n";
                 if (raw_width <= 0 || raw_height <= 0 || raw_depth <= 0 || raw_datatype_str.empty()) {
                     throw std::runtime_error("raw_width, raw_height, raw_depth, and raw_datatype must be specified in inputs for RAW files.");
                 }
                 // TODO: Implement or use appropriate RawReader assuming it takes dims/type
                 // OpenImpala::RawDataType raw_type = OpenImpala::parseRawDataType(raw_datatype_str); // Assuming helper exists
                 // OpenImpala::RawReader reader(full_input_path.string(), raw_width, raw_height, raw_depth, raw_type);
                 // if (!reader.isRead()) throw std::runtime_error("RawReader failed to read file.");
                 // domain_box = reader.box();
                 // reader_success = true;
                 // ... setup ba, dm, mf_phase ...
                 // reader.threshold(threshold_value, phase_id, (phase_id == 0 ? 1 : 0), mf_phase);
                 throw std::runtime_error("RAW file reading not fully implemented in this example.");


            } else if (ext == ".h5" || ext == ".hdf5") {
                 if (verbose > 0) amrex::Print() << " Reading HDF5 file: " << full_input_path << ", Dataset: " << hdf5_dataset << "\n";
                 if (hdf5_dataset.empty()) {
                     throw std::runtime_error("hdf5_dataset must be specified in inputs for HDF5 files.");
                 }
                 OpenImpala::HDF5Reader reader(full_input_path.string(), hdf5_dataset);
                 if (!reader.isRead()) throw std::runtime_error("HDF5Reader failed to read file.");
                 domain_box = reader.box();
                 reader_success = true;
                 // Define and threshold mf_phase
                 ba.define(domain_box);
                 ba.maxSize(box_size);
                 dm.define(ba);
                 mf_phase.define(ba, dm, 1, 1);
                 reader.threshold(threshold_value, phase_id, (phase_id == 0 ? 1 : 0), mf_phase);

            // REMOVED TiffStackReader block
            // else if (tiff_stack_size > 0) { ... }

            } else {
                 amrex::Abort("File format not recognized or supported: " + filename);
            }

            if (!reader_success || domain_box.isEmpty()) {
                 amrex::Abort("Failed to read valid domain box from input file.");
            }

            // --- Setup Geometry and Fill Ghost Cells ---
            // Use standard physical box based on index space (assumes voxel size 1)
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int, AMREX_SPACEDIM> is_periodic{0, 0, 0}; // Assuming non-periodic
            geom.define(domain_box, &rb, 0, is_periodic.data());

            // Fill ghost cells for the phase field (needed for solver stencils)
            mf_phase.FillBoundary(geom.periodicity());

        } catch (const std::exception& e) {
            amrex::Abort("Error during file reading or grid setup: " + std::string(e.what()));
        }

        // --- Calculate Volume Fraction ---
        if (verbose > 0) amrex::Print() << " Calculating Volume Fraction for Phase ID: " << phase_id << "\n";
        OpenImpala::VolumeFraction vf(mf_phase, phase_id, 0); // Assume phase data in component 0
        amrex::Real actual_vf = vf.value(false); // Global calculation
        amrex::Print() << "  Volume Fraction = " << std::fixed << std::setprecision(6) << actual_vf << "\n";

        // --- Calculate Tortuosity ---
        std::vector<OpenImpala::Direction> directions_to_compute;
        std::string upper_direction_str = direction_str;
        std::transform(upper_direction_str.begin(), upper_direction_str.end(), upper_direction_str.begin(), ::toupper);

        if (upper_direction_str == "ALL") {
            directions_to_compute = {OpenImpala::Direction::X, OpenImpala::Direction::Y, OpenImpala::Direction::Z};
        } else {
            // Basic parsing for space-separated directions like "X Y"
             std::stringstream ss(direction_str);
             std::string single_dir_str;
             while (ss >> single_dir_str) {
                 directions_to_compute.push_back(string_to_direction(single_dir_str));
             }
             if (directions_to_compute.empty() && !direction_str.empty()) { // Handle single direction case without spaces, check not empty
                 directions_to_compute.push_back(string_to_direction(direction_str));
             }
             // Remove duplicates if any (optional)
        }

        std::map<OpenImpala::Direction, amrex::Real> tortuosity_results;

        for (const auto& dir : directions_to_compute) {
            if (verbose > 0) amrex::Print() << "\n Calculating Tortuosity for Direction: " << direction_to_string(dir) << "\n";

            try {
                // Assuming TortuosityHypre constructor takes these arguments
                // May need to adjust based on actual TortuosityHypre definition
                OpenImpala::TortuosityHypre tortuosity_solver(
                    geom, ba, dm, mf_phase,
                    actual_vf,      // Pass calculated volume fraction
                    phase_id,       // Phase of interest
                    dir,            // Direction to compute
                    solver_type,
                    results_dir.string(), // Pass results dir for potential plotfiles
                    verbose,
                    write_plotfile, // Control plotfile writing
                    hypre_eps,      // Pass solver tolerance
                    hypre_maxiter   // Pass solver max iterations
                );

                amrex::Real tau_value = tortuosity_solver.value(); // Calculate tortuosity
                tortuosity_results[dir] = tau_value;
                amrex::Print() << "  Tortuosity (" << direction_to_string(dir) << ") = " << std::fixed << std::setprecision(6) << tau_value << "\n";

            } catch (const std::exception& e) {
                 amrex::Warning("Error calculating tortuosity for direction " + direction_to_string(dir) + ": " + std::string(e.what()));
                 tortuosity_results[dir] = -1.0; // Indicate failure
            }
        }

        // --- Write Results to File ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
             std::filesystem::path output_filepath = results_dir / output_filename;
             if (verbose > 0) amrex::Print() << "\n Writing results to: " << output_filepath << "\n";
             std::ofstream outfile(output_filepath);
             if (outfile.is_open()) {
                 outfile << "# Diffusion Calculation Results\n";
                 outfile << "# Input File: " << full_input_path.string() << "\n";
                 outfile << "# Analysis Phase ID: " << phase_id << "\n";
                 outfile << "# Threshold Value: " << threshold_value << "\n";
                 outfile << "# Solver: " << solver_type_str << "\n";
                 outfile << "# Solver Tolerance: " << hypre_eps << "\n";
                 outfile << "# Solver Max Iter: " << hypre_maxiter << "\n";
                 outfile << "# -----------------------------\n";
                 outfile << "VolumeFraction: " << std::fixed << std::setprecision(9) << actual_vf << "\n";
                 for (const auto& pair : tortuosity_results) {
                     outfile << "Tortuosity_" << direction_to_string(pair.first) << ": " << std::fixed << std::setprecision(9) << pair.second << "\n";
                 }
                 outfile.close();
             } else {
                 amrex::Warning("Could not open output file for writing: " + output_filepath.string());
             }
         }


        // --- Final Timing ---
        amrex::Real stop_time = amrex::second() - strt_time;
        amrex::ParallelDescriptor::ReduceRealMax(stop_time, amrex::ParallelDescriptor::IOProcessorNumber());
        amrex::Print() << std::endl << "Total run time (seconds) = " << stop_time << std::endl;

    } // End AMReX scope block

    amrex::Finalize();
    return 0;
}
