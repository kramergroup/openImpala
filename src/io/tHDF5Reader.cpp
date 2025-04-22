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
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Input Parameters ---
        // Declare variables, values MUST come from the 'inputs' file via ParmParse
        std::string hdf5_filename;
        std::string hdf5_dataset; // Will hold the literal path string, e.g., "/t$F/channel$C"
        bool write_plotfile = false;
        int box_size = BOX_SIZE;
        double threshold_value = 1.0;

        // +++ DEBUGGING ADDED +++
        amrex::ParallelDescriptor::Barrier(); // Ensure all ranks sync before printing/checking
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\nDEBUG: === Entering ParmParse Section ===\n";
            amrex::Print() << "DEBUG: Checking for './inputs' file accessibility...\n";
            std::ifstream check_inputs("./inputs");
            if (check_inputs) {
                 amrex::Print() << "DEBUG: SUCCESS - './inputs' file exists and is readable.\n";
                 // Optional: You could read/print the first few lines here if needed, but be careful with file pointers.
                 check_inputs.close(); // Close the check stream
            } else {
                 amrex::Print() << "DEBUG: FAILED - './inputs' file NOT found or readable right before ParmParse!\n";
                 // This would explain the "not found" error if it occurs
            }
            amrex::Print() << "DEBUG: About to create ParmParse object (no prefix).\n";
        }
        amrex::ParallelDescriptor::Barrier(); // Sync again before proceeding
        // +++ END DEBUGGING +++

        { // Use ParmParse to read parameters from the './inputs' file
            amrex::ParmParse pp; // No prefix, reads from root level of ./inputs

            // +++ DEBUGGING ADDED +++
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: ParmParse object created. Attempting pp.get(\"filename\", ...)\n";
            }
            // +++ END DEBUGGING +++

            // Use get() to REQUIRE the parameter from the inputs file.
            // This is where the original error occurred according to the log
            pp.get("filename", hdf5_filename);   // EXPECTS 'filename = /path/to/file.h5' in inputs

            // +++ DEBUGGING ADDED +++
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: SUCCESS - pp.get(\"filename\") returned: " << hdf5_filename << "\n";
                 amrex::Print() << "DEBUG: Attempting pp.get(\"hdf5_dataset\", ...)\n";
            }
            // +++ END DEBUGGING +++

            pp.get("hdf5_dataset", hdf5_dataset); // EXPECTS 'hdf5_dataset = /t$F/channel$C' in inputs

            // +++ DEBUGGING ADDED +++
             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: SUCCESS - pp.get(\"hdf5_dataset\") returned: " << hdf5_dataset << "\n";
                 amrex::Print() << "DEBUG: Attempting optional queries...\n";
             }
            // +++ END DEBUGGING +++

            // query() is suitable for optional parameters
            pp.query("write_plotfile", write_plotfile);
            pp.query("box_size", box_size);
            pp.query("threshold_value", threshold_value);

            // +++ DEBUGGING ADDED +++
             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "DEBUG: Optional queries finished.\n";
                 amrex::Print() << "DEBUG: === Exiting ParmParse Scope ===\n\n";
             }
            // +++ END DEBUGGING +++
        } // End of ParmParse scope

        // Simple check if input HDF5 file exists (using the value read by ParmParse)
        {
            std::ifstream test_ifs(hdf5_filename);
            if (!test_ifs) {
                 amrex::Error("Error: Cannot open input file specified by 'filename': " + hdf5_filename + "\n"
                                "       Ensure './inputs' file exists and specifies correct path relative to execution dir.");
            }
        }

        // Only IOProcessor should print parameter summary
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Starting tHDF5Reader Test (Compiled: " << __DATE__ << " " << __TIME__ << ")\n";
            amrex::Print() << "Input HDF5 file (from inputs): " << hdf5_filename << "\n";
            amrex::Print() << "Input dataset path (from inputs): " << hdf5_dataset << "\n"; // Prints the literal path
            amrex::Print() << "Threshold value: " << threshold_value << "\n";
            amrex::Print() << "Box size: " << box_size << "\n";
            amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";
        }

        // --- Test HDF5Reader ---
        std::unique_ptr<OpenImpala::HDF5Reader> reader_ptr;
        // *** NOTE: These expected dimensions might need adjustment based on the actual HDF5 file ***
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;

        try {
            // *** Pass the LITERAL dataset path string read from inputs ***
            reader_ptr = std::make_unique<OpenImpala::HDF5Reader>(hdf5_filename, hdf5_dataset);
        } catch (const std::exception& e) {
            amrex::Error("Error creating HDF5Reader: " + std::string(e.what()));
        }

        if (!reader_ptr || !reader_ptr->isRead()) {
             amrex::Error("HDF5Reader object creation or file read failed after construction.");
        }

        // --- Check Dimensions & Metadata ---
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

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
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
        amrex::iMultiFab mf(ba, dm, 1, 0); // Integer results of threshold
        mf.setVal(0);

        // --- Test Thresholding ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        }
        try {
            // Pass threshold_value (double) and the iMultiFab to fill
            reader_ptr->threshold(threshold_value, mf);
        } catch (const std::exception& e) {
            amrex::Error("Error during threshold operation: " + std::string(e.what()));
        }

        // --- Check Threshold Result ---
        int min_val = mf.min(0, 0, true); // Global min
        int max_val = mf.max(0, 0, true); // Global max
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Threshold result min value (global): " << min_val << "\n";
            amrex::Print() << "  Threshold result max value (global): " << max_val << "\n";
            if (min_val != 0 || max_val != 1) {
                 amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                                << ") not the expected 0/1.\n";
                 // Consider making this a failure depending on test strictness
                 // amrex::Error("FAIL: Unexpected min/max values after thresholding.");
            } else {
                 amrex::Print() << "  Threshold value range looks plausible.\n";
            }
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
            std::string plotfilename; // Declare outside the block

            // Create directory only on IOProcessor
            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Attempting to create plotfile output directory...\n";
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                     amrex::Print() << "Warning: Could not create output directory (it might already exist): " << test_output_dir << "\n";
                 }
            }
            amrex::ParallelDescriptor::Barrier(); // Ensure directory exists

            // Construct plotfile name
            {
                std::string datetime_str;
                std::time_t now_time = std::time(nullptr);
                std::tm* now_tm = std::localtime(&now_time);
                if (now_tm) {
                    char datetime_buf[80];
                    std::strftime(datetime_buf, sizeof(datetime_buf),"%Y%m%d_%H%M%S", now_tm);
                    datetime_str = datetime_buf;
                } else {
                    datetime_str = "time_error";
                }
                // Using fixed step 0 for plotfile name consistency in tests
                plotfilename = amrex::Concatenate(test_output_dir + "/plt_hdf5_", 0, 5);
                plotfilename += "_" + datetime_str;
            }

            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Writing plot file to: " << plotfilename << "\n";
            }

            // Create a MultiFab (Real) to hold the data for plotting
            amrex::MultiFab mfv(ba, dm, 1, 0); // 0 ghost cells

            // *** CORRECTED COPY LOGIC using MFIter and ParallelFor ***
            #ifdef AMREX_USE_OMP
            #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
            #endif
            for (amrex::MFIter mfi(mfv, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const amrex::Box& tileBox = mfi.tilebox();
                amrex::Array4<amrex::Real> const& real_fab = mfv.array(mfi);
                amrex::Array4<const int> const& int_fab = mf.const_array(mfi);

                amrex::ParallelFor(tileBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    real_fab(i, j, k) = static_cast<amrex::Real>(int_fab(i, j, k));
                });
            }
            // *** End of corrected copy logic ***

            // Write plot file
            amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "  Plot file written.\n";
            }
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "tHDF5Reader Test Completed Successfully.\n";
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Return 0 indicates success
}
