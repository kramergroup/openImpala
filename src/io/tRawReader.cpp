#include "RawReader.H" // Assuming defines OpenImpala::RawReader and OpenImpala::RawDataType

#include <cstdlib>
#include <string>
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <fstream>   // For std::ifstream check
#include <iomanip>   // For std::setw, std::setfill
#include <ctime>
#include <memory>    // For std::unique_ptr
#include <map>       // For mapping string to enum
#include <limits>    // For numeric_limits // Ensure this is included

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
#include <AMReX_Utility.H>      // For amrex::UtilCreateDirectory, amrex::Concatenate
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier
#include <AMReX_GpuLaunch.H> // Provides amrex::ParallelFor


// --- Test Configuration ---
// Output directory relative to executable location
const std::string test_output_dir = "tRawReader_output";
// Default Box size for breaking down domain (can be overridden by inputs)
const int DEFAULT_BOX_SIZE = 32;
// Default threshold value (can be overridden by inputs)
const double DEFAULT_THRESHOLD_VALUE = 127.5;


// Helper to convert string to RawDataType enum
OpenImpala::RawDataType StringToRawDataType(const std::string& type_str) {
    // Create a map for easy lookup (case-insensitive might be better)
    static const std::map<std::string, OpenImpala::RawDataType> type_map = {
        {"UINT8", OpenImpala::RawDataType::UINT8},
        {"INT8", OpenImpala::RawDataType::INT8},
        {"UINT16_LE", OpenImpala::RawDataType::UINT16_LE},
        {"UINT16_BE", OpenImpala::RawDataType::UINT16_BE},
        {"INT16_LE", OpenImpala::RawDataType::INT16_LE},
        {"INT16_BE", OpenImpala::RawDataType::INT16_BE},
        {"UINT32_LE", OpenImpala::RawDataType::UINT32_LE},
        {"UINT32_BE", OpenImpala::RawDataType::UINT32_BE},
        {"INT32_LE", OpenImpala::RawDataType::INT32_LE},
        {"INT32_BE", OpenImpala::RawDataType::INT32_BE},
        {"FLOAT32_LE", OpenImpala::RawDataType::FLOAT32_LE},
        {"FLOAT32_BE", OpenImpala::RawDataType::FLOAT32_BE},
        {"FLOAT64_LE", OpenImpala::RawDataType::FLOAT64_LE},
        {"FLOAT64_BE", OpenImpala::RawDataType::FLOAT64_BE}
        // Add more if needed
    };

    auto it = type_map.find(type_str);
    if (it != type_map.end()) {
        return it->second;
    } else {
        // Return UNKNOWN, let the main code handle the error/abort
        amrex::Print() << "Warning: [StringToRawDataType] Unrecognized RawDataType string: "
                       << type_str << ". Returning UNKNOWN.\n";
        return OpenImpala::RawDataType::UNKNOWN;
    }
}

//================================================================
// Main Function
//================================================================
int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // Use a block for AMReX object lifetimes
    {
        // --- Parameters to be read from input file ---
        std::string raw_filename; // Mandatory
        int width = 0;            // Mandatory
        int height = 0;           // Mandatory
        int depth = 0;            // Mandatory
        std::string datatype_str; // Mandatory

        // Optional parameters with defaults
        double threshold_val = DEFAULT_THRESHOLD_VALUE;
        bool write_plotfile = false;
        int box_size = DEFAULT_BOX_SIZE;

        // --- ParmParse to read parameters ---
        // Assumes an input file (e.g., ./inputs) is provided via command line
        // or default AMReX behavior.
        {
            amrex::ParmParse pp; // No prefix - queries global database

            // Use get() for mandatory parameters - will abort if not found
            pp.get("rawfile", raw_filename);
            pp.get("width", width);
            pp.get("height", height);
            pp.get("depth", depth);
            pp.get("datatype", datatype_str); // Read type as string

            // Use query() for optional parameters - keeps default if not found
            pp.query("threshold", threshold_val);
            pp.query("write_plotfile", write_plotfile);
            pp.query("box_size", box_size);
        }

        // --- Validate Mandatory Parameters ---
        OpenImpala::RawDataType data_type_enum = StringToRawDataType(datatype_str);
        if (data_type_enum == OpenImpala::RawDataType::UNKNOWN) {
            amrex::Abort("Error: Invalid 'datatype' string specified in input file: " + datatype_str + "\n"
                         "  Supported types: UINT8, INT8, UINT16_LE, UINT16_BE, INT16_LE, ..., FLOAT64_BE");
        }
        if (width <= 0 || height <= 0 || depth <= 0) {
            amrex::Abort("Error: Invalid dimensions specified in input file (W=" + std::to_string(width)
                         + ", H=" + std::to_string(height) + ", D=" + std::to_string(depth) + "). Must be positive.");
        }
        if (box_size <= 0) {
             amrex::Abort("Error: Invalid 'box_size' specified (" + std::to_string(box_size) + "). Must be positive.");
        }


        // --- Check if input file exists before attempting to read ---
        // (Useful especially if path itself comes from input)
        {
            std::ifstream test_ifs(raw_filename);
            if (!test_ifs) {
                amrex::Abort("Error: Cannot open input rawfile specified: " + raw_filename);
            }
        }

        // --- Parameter Summary (on IOProcessor) ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- tRawReader Test Configuration ---\n";
            amrex::Print() << "  Input RAW file:    " << raw_filename << "\n";
            amrex::Print() << "  Input Dimensions:  " << width << "x" << height << "x" << depth << "\n";
            amrex::Print() << "  Input DataType:    " << datatype_str << " (Enum: " << static_cast<int>(data_type_enum) << ")\n";
            amrex::Print() << "  Threshold value:   " << threshold_val << "\n";
            amrex::Print() << "  Box Size:          " << box_size << "\n";
            amrex::Print() << "  Write plot file:   " << (write_plotfile ? "Yes" : "No") << "\n";
            amrex::Print() << "--------------------------------------\n\n";
        }

        // --- Test RawReader ---
        std::unique_ptr<OpenImpala::RawReader> reader_ptr;
        try {
            // Use the constructor that takes metadata and throws on failure
            reader_ptr = std::make_unique<OpenImpala::RawReader>(raw_filename, width, height, depth, data_type_enum);

        } catch (const std::exception& e) {
            // Catch errors during construction (e.g., file size mismatch, allocation fail)
            amrex::Abort("Error creating RawReader: " + std::string(e.what()));
        }

        // --- Check Dimensions & Type Post-Read (should match input if successful) ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Checking dimensions and data type reported by reader...\n";
        }
        int actual_width = reader_ptr->width();
        int actual_height = reader_ptr->height();
        int actual_depth = reader_ptr->depth();
        OpenImpala::RawDataType actual_type = reader_ptr->getDataType();

        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Reader dimensions: " << actual_width << "x" << actual_height << "x" << actual_depth << "\n";
             amrex::Print() << "  Reader DataType:   " << static_cast<int>(actual_type) << "\n";
        }

        // These checks primarily verify the reader stored the input metadata correctly
        if (actual_width != width || actual_height != height || actual_depth != depth) {
            amrex::Abort("FAIL: Reader dimensions do not match input dimensions provided to constructor.");
        }
         if (actual_type != data_type_enum) {
            amrex::Abort("FAIL: Reader data type does not match input data type provided to constructor.");
         }
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "  Dimensions and data type confirmed.\n";
        }
        // TODO: Add tests for RawReader getValue(i,j,k) against known data points if available.
        // TODO: Add separate tests for error conditions (bad file path, size mismatch, unsupported type).

        // --- Setup AMReX Data Structures ---
        amrex::Geometry geom;
        amrex::Box domain_box = reader_ptr->box();
        if (domain_box.isEmpty()) {
             amrex::Abort("FAIL: Reader returned an empty box after successful read.");
        }
        {
            // Use physical domain size equal to index space for simplicity here
            amrex::RealBox rb({AMREX_D_DECL(0.0, 0.0, 0.0)},
                              {AMREX_D_DECL(static_cast<amrex::Real>(actual_width),
                                            static_cast<amrex::Real>(actual_height),
                                            static_cast<amrex::Real>(actual_depth))});
            amrex::Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)}; // Non-periodic
            // Setup static geometry data
            amrex::Geometry::Setup(&rb, 0, is_periodic.data());
            // Define the BoundingBox object
            geom.define(domain_box);
        }

        amrex::BoxArray ba(domain_box);
        ba.maxSize(box_size); // Use box_size read from inputs
        amrex::DistributionMapping dm(ba);
        amrex::iMultiFab mf(ba, dm, 1, 0); // 1 component, 0 ghost cells
        mf.setVal(-1); // Initialize to -1 to check if all cells get set by threshold

        // --- Test Thresholding ---
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Performing threshold > " << threshold_val << "...\n";
        }
        try {
            // Use threshold(double, iMultiFab) overload for 1/0 output
            reader_ptr->threshold(threshold_val, mf);
            // TODO: Test threshold overload with custom true/false values
        } catch (const std::exception& e) {
            // Catch potential errors from threshold method itself (e.g., Abort inside)
            amrex::Abort("Error during threshold operation: " + std::string(e.what()));
        }

        // --- Check Threshold Result ---
        // min/max are collective operations
        int min_val = mf.min(0);
        int max_val = mf.max(0);

        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "  Threshold result min value (global): " << min_val << "\n";
             amrex::Print() << "  Threshold result max value (global): " << max_val << "\n";

             // Check if result is strictly binary {0, 1} and both are present
             if (min_val == 0 && max_val == 1) {
                 amrex::Print() << "  Threshold value range looks plausible (0 and 1 found).\n";
             } else if (min_val == 0 && max_val == 0) {
                 amrex::Print() << "Warning: Thresholded data is uniformly 0. Check threshold/data.\n";
             } else if (min_val == 1 && max_val == 1) {
                 amrex::Print() << "Warning: Thresholded data is uniformly 1. Check threshold/data.\n";
             } else {
                 // This case includes min_val being -1 (if some cells weren't set) or other unexpected values
                 amrex::Abort("FAIL: Thresholded data min/max (" + std::to_string(min_val) + "/"
                              + std::to_string(max_val) + ") not {0, 1}. Check threshold value ("
                              + std::to_string(threshold_val) + ") or sample data range/content.");
             }
        }

        // --- Optional: Write Plotfile ---
        if (write_plotfile) {
             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "Writing plot file...\n";
             }

             // Create output directory relative to executable location
             if (amrex::ParallelDescriptor::IOProcessor()) {
                 if (!amrex::UtilCreateDirectory(test_output_dir, 0755)) {
                     // Use CreateDirectoryFailed to potentially abort or handle error
                     amrex::CreateDirectoryFailed(test_output_dir);
                 }
             }
             // Ensure all processors wait until directory is created
             amrex::ParallelDescriptor::Barrier();

             // Get datetime string for unique filename
             std::string datetime_str = amrex::UniqueString(); // Use AMReX helper
             std::string plotfilename = test_output_dir + "/rawreadertest_" + datetime_str;

             // Copy integer data to float MultiFab for plotting
             amrex::MultiFab mfv(ba, dm, 1, 0);
             // Use amrex::Copy which handles parallelism internally
             amrex::Copy(mfv, mf, 0, 0, 1, 0);

             // Write the plotfile
             amrex::WriteSingleLevelPlotfile(plotfilename, mfv, {"phase_threshold"}, geom, 0.0, 0);

             if (amrex::ParallelDescriptor::IOProcessor()) {
                 amrex::Print() << "  Plot file written to: " << plotfilename << "\n";
             }
        }

        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "\n--- tRawReader Test Completed Successfully ---\n";
        }

    } // End AMReX scope block

    amrex::Finalize();
    return 0; // Return 0 indicates success
}
