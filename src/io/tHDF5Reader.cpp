#include "HDF5Reader.H" // Assuming this defines OpenImpala::HDF5Reader

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
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
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, MyProc
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor
#include <AMReX_MFIter.H>    // Needed for MFIter


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
        std::string hdf5_filename;
        std::string hdf5_dataset;
        bool write_plotfile = false;
        int box_size = BOX_SIZE;
        double threshold_value = 1.0;


        // --- ParmParse ---
        { // Use ParmParse to query the database populated by Initialize
            amrex::ParmParse pp; // No prefix - queries global database

            // These should now work if make test passes ./inputs argument
            pp.get("filename", hdf5_filename);
            pp.get("hdf5_dataset", hdf5_dataset);

            // Optional parameters
            pp.query("write_plotfile", write_plotfile);
            pp.query("box_size", box_size);
            pp.query("threshold_value", threshold_value);

        } // End of ParmParse scope

        // Check the HDF5 file using the path read
        {
            std::ifstream test_ifs(hdf5_filename);
            if (!test_ifs) {
                amrex::Error("Error: Cannot open input HDF5 file specified by 'filename': " + hdf5_filename);
            }
            // (Removed DEBUG print for successful open)
        }

        // --- Parameter Summary ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Starting tHDF5Reader Test (Compiled: " << __DATE__ << " " << __TIME__ << ")\n";
            amrex::Print() << "Input HDF5 file (from inputs): " << hdf5_filename << "\n";
            amrex::Print() << "Input dataset path (from inputs): " << hdf5_dataset << "\n";
            amrex::Print() << "Threshold value: " << threshold_value << "\n";
            amrex::Print() << "Box size: " << box_size << "\n";
            amrex::Print() << "Write plot file: " << (write_plotfile ? "Yes" : "No") << "\n";
        }

        // --- Test HDF5Reader ---
        std::unique_ptr<OpenImpala::HDF5Reader> reader_ptr;
        // Expected dimensions (adjust if your sample data is different)
        int expected_width = 100;
        int expected_height = 100;
        int expected_depth = 100;

        try {
            // (Removed DEBUG print for object creation start)
            reader_ptr = std::make_unique<OpenImpala::HDF5Reader>(hdf5_filename, hdf5_dataset);
            // (Removed DEBUG print for object creation end)
        } catch (const std::exception& e) {
            amrex::Error("Error creating HDF5Reader: " + std::string(e.what()));
        }

        if (!reader_ptr || !reader_ptr->isRead()) {
            amrex::Error("HDF5Reader object creation or file read failed after construction.");
        }
        // (Removed DEBUG print for isRead() check)


        // --- Check Dimensions & Metadata ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Checking dimensions and metadata...\n";
        }
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Read dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
            // (Removed skipping metadata checks message)
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
        // (Removed DEBUG print for geometry/boxarray setup)
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
        mf.setVal(0); // Initialize to 0
        // (Removed DEBUG print for data structure creation end)

        // --- Test Thresholding ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Performing threshold > " << threshold_value << "...\n";
        }
        try {
            reader_ptr->threshold(threshold_value, mf);
            // (Removed DEBUG print for threshold completion)
        } catch (const std::exception& e) {
            amrex::Error("Error during threshold operation: " + std::string(e.what()));
        }

        // --- Check Threshold Result ---
        int min_val = mf.min(0, 0, true); // Use mf.min/max on iMultiFab directly
        int max_val = mf.max(0, 0, true);
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Threshold result min value (global): " << min_val << "\n";
            amrex::Print() << "  Threshold result max value (global): " << max_val << "\n";
            // Expecting 0 and 1 for the default threshold overload
            if (min_val != 0 || max_val != 1) {
                amrex::Print() << "Warning: Thresholded data min/max (" << min_val << "/" << max_val
                               << ") not the expected 0/1.\n";
            } else {
                amrex::Print() << "  Threshold value range looks plausible.\n";
            }
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
             std::string plotfilename;
             const std::string plotfile_prefix = "plt_thresh";
             if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                     amrex::CreateDirectoryFailed(test_output_dir);
                 }
             }
             amrex::ParallelDescriptor::Barrier(); // Ensure directory exists before proceeding

             // Construct filename (e.g., plt_thresh_00000) - simpler example
             plotfilename = amrex::Concatenate(test_output_dir + "/" + plotfile_prefix, 5, '0');

             if (amrex::ParallelDescriptor::IOProcessor()) {
                  amrex::Print() << "Writing plotfile: " << plotfilename << "\n";
             }
             // Need to copy iMultiFab to MultiFab for plotting
             amrex::MultiFab mfv(ba, dm, 1, 0);
             amrex::Copy(mfv, mf, 0, 0, 1, 0); // Copy component 0 from mf to mfv

             amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);

             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Plotfile writing complete.\n";
             }
         }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "tHDF5Reader Test Completed Successfully.\n";
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Return 0 indicates success
}
