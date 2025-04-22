#include "HDF5Reader.H" // Assuming this defines OpenImpala::HDF5Reader

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check (needed for debug)
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr, std::make_unique
#include <sstream>   // For std::stringstream

#include <AMReX.H>
#include <AMReX_ParmParse.H> // For reading input files
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H> // For amrex::UtilCreateDirectory, amrex::Concatenate
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, MyProc (needed for debug)
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor
#include <AMReX_MFIter.H>    // <<< Needed for MFIter >>>


// Output directory relative to executable location
const std::string test_output_dir = "tHDF5Reader_output";
// Define Box size for breaking down domain (can also be read from inputs)
const int BOX_SIZE = 32;

int main (int argc, char* argv[])
{
    // *** NOTE: amrex::Initialize now uses argc, argv correctly ***
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        std::string hdf5_filename;
        std::string hdf5_dataset;
        bool write_plotfile = false;
        int box_size = BOX_SIZE;
        double threshold_value = 1.0;


        // +++ MANUAL READ DEBUGGING (Optional - can be removed if ParmParse works) +++
        amrex::ParallelDescriptor::Barrier(); // Sync before IO
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\nDEBUG: === Manually reading ./inputs line by line: ===\n";
            std::ifstream manual_reader("./inputs"); // Check the symlink directly
            if (manual_reader) {
                std::string line;
                int line_num = 1;
                while (std::getline(manual_reader, line)) {
                    amrex::Print() << "DEBUG: Manual Line " << line_num++ << ": [" << line << "]\n";
                }
                manual_reader.close();
                amrex::Print() << "DEBUG: === End of manual read. ===\n";
            } else {
                // This check might fail if run *before* the make test target creates ./inputs
                // It's less useful now we know Initialize needs the command-line arg.
                amrex::Print() << "DEBUG: Manual read check - could not open ./inputs (symlink might not exist yet).\n";
            }
        }
        amrex::ParallelDescriptor::Barrier(); // Sync after IO
        // +++ END MANUAL READ DEBUGGING +++


        // +++ DEBUGGING ACCESSIBILITY (Less relevant now, but harmless) +++
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\nDEBUG: === Entering ParmParse Section ===\n";
            amrex::Print() << "DEBUG: Checking for './inputs' file accessibility...\n";
            std::ifstream check_inputs("./inputs"); // Check symlink
            if (check_inputs) {
                 amrex::Print() << "DEBUG: SUCCESS - './inputs' file exists and is readable (ifstream check).\n";
                 check_inputs.close();
            } else {
                 amrex::Print() << "DEBUG: FAILED - './inputs' file NOT found or readable (ifstream check)!\n";
            }
             amrex::Print() << "DEBUG: About to create ParmParse object (no prefix) - relies on Initialize having read the input file from command line.\n";
        }
        amrex::ParallelDescriptor::Barrier();
        // +++ END DEBUGGING +++

        { // Use ParmParse to query the database populated by Initialize
            amrex::ParmParse pp; // No prefix - queries global database

            // +++ Check if ParmParse contains the key +++
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: ParmParse object created. Checking pp.contains(\"filename\")...\n";
                 if (pp.contains("filename")) {
                     amrex::Print() << "DEBUG: SUCCESS - pp.contains(\"filename\") is TRUE.\n";
                 } else {
                     amrex::Print() << "DEBUG: FAILED - pp.contains(\"filename\") is FALSE! (Did Initialize read the input file?)\n";
                 }
                 amrex::Print() << "DEBUG: Now attempting pp.get(\"filename\", ...)\n";
            }
            amrex::ParallelDescriptor::Barrier();
            // +++ End contains check +++

            // This should now work if make test passes ./inputs argument
            pp.get("filename", hdf5_filename);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: SUCCESS - pp.get(\"filename\") returned: " << hdf5_filename << "\n";
                 amrex::Print() << "DEBUG: Attempting pp.get(\"hdf5_dataset\", ...)\n";
            }

            pp.get("hdf5_dataset", hdf5_dataset);

             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: SUCCESS - pp.get(\"hdf5_dataset\") returned: " << hdf5_dataset << "\n";
                 amrex::Print() << "DEBUG: Attempting optional queries...\n";
             }

            pp.query("write_plotfile", write_plotfile);
            pp.query("box_size", box_size);
            pp.query("threshold_value", threshold_value);

             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: Optional queries finished.\n";
                 amrex::Print() << "DEBUG: === Exiting ParmParse Scope ===\n\n";
             }
        } // End of ParmParse scope

        // Check the HDF5 file using the path read
        {
            std::ifstream test_ifs(hdf5_filename);
            if (!test_ifs) {
                 amrex::Error("Error: Cannot open input HDF5 file specified by 'filename': " + hdf5_filename + "\n"
                                "       Value came from input file '" + amrex::ParmParse::getInputsfilename() + "' or command line.");
                 // Note: amrex::ParmParse::getInputsfilename() returns the name of the file read by Initialize
            } else {
                 if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: Successfully opened HDF5 file '" << hdf5_filename << "' for check.\n";
            }
        }

        // --- The rest of the code remains the same ---

        // Only IOProcessor should print parameter summary
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Starting tHDF5Reader Test (Compiled: " << __DATE__ << " " << __TIME__ << ")\n";
            amrex::Print() << "Input HDF5 file (from inputs): " << hdf5_filename << "\n";
            amrex::Print() << "Input dataset path (from inputs): " << hdf5_dataset << "\n";
            amrex::Print() << "Threshold value: " << threshold_value << "\n";
            amrex::Print() << "Box size: " << box_size << "\n";
            amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";
        }

        // Test HDF5Reader
        std::unique_ptr<OpenImpala::HDF5Reader> reader_ptr;
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;

        try {
             if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: Creating HDF5Reader object...\n";
            reader_ptr = std::make_unique<OpenImpala::HDF5Reader>(hdf5_filename, hdf5_dataset);
             if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: HDF5Reader object created.\n";
        } catch (const std::exception& e) {
            amrex::Error("Error creating HDF5Reader: " + std::string(e.what()));
        }

        if (!reader_ptr || !reader_ptr->isRead()) {
             amrex::Error("HDF5Reader object creation or file read failed after construction.");
        } else {
             if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: HDF5Reader reports isRead() is true.\n";
        }

        // Check Dimensions & Metadata
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Checking dimensions and metadata...\n";
        }
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Read dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
            amrex::Print() << "  (Skipping metadata checks...).\n";
        }

        if (actual_width != expected_width || actual_height != expected_height || actual_depth != expected_depth) {
            std::stringstream ss;
            ss << "FAIL: Read dimensions (" << actual_width << "x" << actual_height << "x" << actual_depth
               << ") do not match expected dimensions (" << expected_width << "x" << expected_height << "x" << expected_depth << ").";
            amrex::Error(ss.str());
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Dimension check passed.\n";
        }

        // Setup AMReX Data Structures
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
         if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: Setting up Geometry and BoxArray...\n";
        {
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(amrex::Real(domain_box.length(0)),
                                            amrex::Real(domain_box.length(1)),
                                            amrex::Real(domain_box.length(2)))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{0, 0, 0};
            geom.define(domain_box, &rb, amrex::CoordSys::cartesian, is_periodic.data());
        }
        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size);
        amrex::DistributionMapping dm(ba);
        amrex::iMultiFab mf(ba, dm, 1, 0);
        mf.setVal(0);
         if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: AMReX data structures created.\n";

        // Test Thresholding
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        }
        try {
            reader_ptr->threshold(threshold_value, mf);
             if (amrex::ParallelDescriptor::IOProcessor()) amrex::Print() << "DEBUG: Threshold operation completed.\n";
        } catch (const std::exception& e) {
            amrex::Error("Error during threshold operation: " + std::string(e.what()));
        }

        // Check Threshold Result
        int min_val = mf.min(0, 0, true);
        int max_val = mf.max(0, 0, true);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Threshold result min value (global): " << min_val << "\n";
            amrex::Print() << "  Threshold result max value (global): " << max_val << "\n";
            if (min_val != 0 || max_val != 1) {
                 amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                                << ") not the expected 0/1.\n";
            } else {
                 amrex::Print() << "  Threshold value range looks plausible.\n";
            }
        }

        // Optional: Write Plotfile
        if (write_plotfile) {
            // ... (Plotfile writing code remains the same) ...
             std::string plotfilename;
             if (amrex::ParallelDescriptor::IOProcessor()) { /* ... create dir ... */ }
             amrex::ParallelDescriptor::Barrier();
             { /* ... construct filename ... */ }
             if (amrex::ParallelDescriptor::IOProcessor()) { /* ... print writing message ... */ }
             amrex::MultiFab mfv(ba, dm, 1, 0);
             #ifdef AMREX_USE_OMP
             #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
             #endif
             for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) { /* ... copy mf to mfv ... */ }
             amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);
             if (amrex::ParallelDescriptor::IOProcessor()) { /* ... print written message ... */ }
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "tHDF5Reader Test Completed Successfully.\n";
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Return 0 indicates success
}
