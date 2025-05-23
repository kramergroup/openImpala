/**
 * @file TiffReader.cpp
 * @brief Implementation of the TiffReader class for reading TIFF image files
 * and sequences, including thresholding capabilities. Handles various
 * TIFF formats and integrates with AMReX MultiFabs.
 */

#include <omp.h> // For OpenMP directives

// --- Other standard and library includes ---
#include "TiffReader.H"
#include <tiffio.h>  // libtiff C API Header
#include <memory>    // For std::unique_ptr
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <iomanip>
// #include <map> // Only if attributes are re-enabled

// --- AMReX Includes ---
#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
// #include <AMReX_GpuContainers.H> // Optional
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Utility.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_MFIter.H>
#include <AMReX_Array4.H>
#include <AMReX_Loop.H>


namespace OpenImpala {

//================================================================
// Anonymous namespace for internal helpers
//================================================================
namespace {

// RAII wrapper for TIFF* handle
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double interpretBytesAsDouble(const unsigned char* byte_ptr,
                              uint16_t bits_per_sample,
                              uint16_t sample_format)
{
    if (bits_per_sample < 8 || (bits_per_sample % 8 != 0) ) return 0.0;
    size_t bytes_per_sample_val = bits_per_sample / 8;
    double value = 0.0;
    switch (sample_format) {
        case SAMPLEFORMAT_UINT:
            if (bytes_per_sample_val == 1) { uint8_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
            else if (bytes_per_sample_val == 2) { uint16_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
            else if (bytes_per_sample_val == 4) { uint32_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
            else if (bytes_per_sample_val == 8) { uint64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
            break;
        case SAMPLEFORMAT_INT:
             if (bytes_per_sample_val == 1) { int8_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
             else if (bytes_per_sample_val == 2) { int16_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
             else if (bytes_per_sample_val == 4) { int32_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
             else if (bytes_per_sample_val == 8) { int64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
            break;
        case SAMPLEFORMAT_IEEEFP:
             if (bytes_per_sample_val == 4) { static_assert(sizeof(float) == 4, "Float size mismatch"); float val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); }
             else if (bytes_per_sample_val == 8) { static_assert(sizeof(double) == 8, "Double size mismatch"); double val_d; std::memcpy(&val_d, byte_ptr, sizeof(val_d)); value = val_d; }
            break;
        default: value = 0.0; break;
    }
    return value;
}

std::string generateFilename(const std::string& base, int index, int digits, const std::string& suffix) {
     std::ostringstream ss;
     ss << base << std::setw(digits) << std::setfill('0') << index << suffix;
     return ss.str();
}

} // end anonymous namespace

//================================================================
// Constructors
//================================================================
TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0),
    m_bits_per_sample(0), m_sample_format(SAMPLEFORMAT_UINT), m_samples_per_pixel(1),
    m_fill_order(FILLORDER_MSB2LSB),
    m_is_read(false), m_is_sequence(false), m_start_index(0), m_digits(1)
{}

TiffReader::TiffReader(const std::string& filename) : TiffReader() {
    m_filename = filename; // Store filename before calling readFile, as readFile might use it (e.g. in error messages on rank 0)
    if (!readFile(filename)) {
        throw std::runtime_error("TiffReader(filename): Failed to read metadata from file: " + filename);
    }
}

TiffReader::TiffReader(
    const std::string& base_pattern, int num_files,
    int start_index, int digits, const std::string& suffix) : TiffReader() {
    // Store these before calling readFileSequence, as it will use them
    m_base_pattern = base_pattern; m_start_index = start_index; m_digits = digits; m_suffix = suffix;
    if (!readFileSequence(base_pattern, num_files, start_index, digits, suffix)) {
         throw std::runtime_error("TiffReader(sequence): Failed to read metadata for sequence: " + base_pattern);
    }
}

//================================================================
// Metadata Getters
//================================================================
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) { return amrex::Box(); }
    return amrex::Box(amrex::IntVect::TheZeroVector(),
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

//================================================================
// readFile Method (Single Stack File)
//================================================================
bool TiffReader::readFile(const std::string& filename) {
    // Ensure member filename is set correctly before this method is called by constructor, or set it here.
    // If called directly, update member variables.
    m_is_sequence = false; m_filename = filename; m_base_pattern = "";
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=SAMPLEFORMAT_UINT, spp_r0=1, fill_order_r0 = FILLORDER_MSB2LSB;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (filename.empty()) { amrex::Abort("[TiffReader::readFile] Filename cannot be empty."); }
        TiffPtr tif(TIFFOpen(filename.c_str(), "r"), TiffCloser());
        if (!tif) { amrex::Abort("[TiffReader::readFile] Failed to open TIFF file: " + filename); }

        uint32_t w32=0, h32=0; uint16_t planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
            amrex::Abort("[TiffReader::readFile] Failed to get image dimensions from: " + filename);
        }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);
        
        if (amrex::Verbose() >= 2 && amrex::ParallelDescriptor::IOProcessor()) { // Ensure IOProc check for safety if Verbose isn't global
            amrex::Print() << "TiffReader DEBUG (readFile): FillOrder tag read from file '" << filename << "' is: " << fill_order_r0
                           << " (1=MSB2LSB, 2=LSB2MSB, Standard TIFF Default=" << FILLORDER_MSB2LSB << ")\n";
        }

        width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32);
        bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG || spp_r0 != 1) {
            std::stringstream ss;
            ss << "[TiffReader::readFile] Invalid/unsupported TIFF: " << filename << " (W=" << width_r0 
               << ", H=" << height_r0 << ", BPS=" << bps_r0 << ", Planar=" << planar << ", SPP=" << spp_r0 << ").";
            amrex::Abort(ss.str());
        }
        depth_r0 = 0;
        if (!TIFFSetDirectory(tif.get(), 0)) { amrex::Abort("[TiffReader::readFile] Failed to set initial directory (0) in: " + filename); }
        do { depth_r0++; } while (TIFFReadDirectory(tif.get()));
        if (depth_r0 == 0) { amrex::Abort("[TiffReader::readFile] No directories (depth=0) in: " + filename); }
    }

    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    std::vector<int> idata = {width_r0, height_r0, depth_r0, static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0), 0 /*is_sequence=false*/, static_cast<int>(fill_order_r0)};
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), root, amrex::ParallelDescriptor::Communicator());
    m_width=idata[0]; m_height=idata[1]; m_depth=idata[2]; m_bits_per_sample=(uint16_t)idata[3]; m_sample_format=(uint16_t)idata[4]; m_samples_per_pixel=(uint16_t)idata[5]; m_is_sequence=(bool)idata[6]; m_fill_order=(uint16_t)idata[7];

    // Corrected string broadcast for m_filename
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_filename.length()); // m_filename is already set on IOProc
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        m_filename.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    }

    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0) {
        amrex::Abort("TiffReader::readFile: Invalid metadata received after broadcast.");
    }
    m_is_read = true;
    return true;
}

//================================================================
// readFileSequence Method
//================================================================
bool TiffReader::readFileSequence(
    const std::string& base_pattern, int num_files,
    int start_index, int digits, const std::string& suffix) {
    // Ensure member variables are set correctly if this is called directly
    m_is_sequence = true; m_base_pattern = base_pattern; m_start_index = start_index; m_digits = digits; m_suffix = suffix; m_filename = "";
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=SAMPLEFORMAT_UINT, spp_r0=1, fill_order_r0 = FILLORDER_MSB2LSB;
    std::string first_filename_r0 = "";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (num_files <= 0 || digits <= 0 || base_pattern.empty()) { amrex::Abort("[TiffReader::readFileSequence] Invalid sequence params."); }
        depth_r0 = num_files; // For sequences, depth is num_files
        first_filename_r0 = generateFilename(base_pattern, start_index, digits, suffix);
        TiffPtr tif(TIFFOpen(first_filename_r0.c_str(), "r"), TiffCloser());
        if (!tif) { amrex::Abort("[TiffReader::readFileSequence] Failed to open first sequence file: " + first_filename_r0); }
        
        uint32_t w32=0, h32=0; uint16_t planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
             amrex::Abort("[TiffReader::readFileSequence] Failed to get image dimensions from: " + first_filename_r0);
        }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);

        if (amrex::Verbose() >= 2 && amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "TiffReader DEBUG (readFileSequence): FillOrder tag read from file '" << first_filename_r0 << "' is: " << fill_order_r0
                           << " (1=MSB2LSB, 2=LSB2MSB, Standard TIFF Default=" << FILLORDER_MSB2LSB << ")\n";
        }

        width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32);
        bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG || spp_r0 != 1) {
             std::stringstream ss;
             ss << "[TiffReader::readFileSequence] Invalid/unsupported TIFF: " << first_filename_r0 << " (W=" << width_r0 
                << ", H=" << height_r0 << ", BPS=" << bps_r0 << ", Planar=" << planar << ", SPP=" << spp_r0 << ").";
             amrex::Abort(ss.str());
        }
        if (TIFFReadDirectory(tif.get())) { amrex::Warning("[TiffReader::readFileSequence] First sequence file has >1 directory. Using first only for metadata.");}
    }
    
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    // Broadcast m_start_index, m_digits separately or ensure they are set from constructor before this Bcast
    std::vector<int> idata = {width_r0, height_r0, depth_r0, static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0), 1 /*is_sequence=true*/, m_start_index, m_digits, static_cast<int>(fill_order_r0)};
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), root, amrex::ParallelDescriptor::Communicator());
    m_width=idata[0]; m_height=idata[1]; m_depth=idata[2]; m_bits_per_sample=(uint16_t)idata[3]; m_sample_format=(uint16_t)idata[4]; m_samples_per_pixel=(uint16_t)idata[5]; m_is_sequence=(bool)idata[6]; m_start_index=idata[7]; m_digits=idata[8]; m_fill_order=(uint16_t)idata[9];
    
    // Corrected string broadcast for m_base_pattern
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_base_pattern.length()); // m_base_pattern is set on IOProc
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        m_base_pattern.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    }

    // Corrected string broadcast for m_suffix
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_suffix.length()); // m_suffix is set on IOProc
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root, amrex::ParallelDescriptor::Communicator());
        m_suffix.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root, amrex::ParallelDescriptor::Communicator());
    }

    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || (m_is_sequence && m_base_pattern.empty())) {
        amrex::Abort("TiffReader::readFileSequence: Invalid metadata after broadcast.");
    }
    m_is_read = true;
    return true;
}

//================================================================
// readDistributedIntoFab Method - Main Data Reading Logic
//================================================================
void TiffReader::readDistributedIntoFab(
    amrex::iMultiFab& dest_mf, int value_if_true,
    int value_if_false, double raw_threshold) const {
    if (!m_is_read) { amrex::Abort("[TiffReader::readDistributedIntoFab] Metadata not processed."); }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.boxArray().minimalBox()==this->box(), "Dest MF BoxArray domain mismatch.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nComp()==1, "Dest MF must have 1 component.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nGrow()==0, "Dest MF must have 0 ghost cells.");

    const int bits_per_sample_val = m_bits_per_sample;
    const size_t bytes_per_sample_calc = (bits_per_sample_val >= 8) ? (bits_per_sample_val / 8) : 1;
    const size_t bytes_per_pixel = bytes_per_sample_calc * m_samples_per_pixel; // Assumes SPP=1
    if (bits_per_sample_val == 0 ) { amrex::Abort("[TiffReader] Bits per sample is zero!"); }

    const int image_width_const = m_width; 
    const int image_height_const = m_height;
    const int image_depth_const = m_depth;
    const uint16_t fill_order_local = m_fill_order;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        std::vector<unsigned char> temp_buffer;
        TiffPtr thread_local_tif_handle = nullptr; 

        for (amrex::MFIter mfi(dest_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi);
            const amrex::Box& tile_box = mfi.tilebox();
            const int k_min_fab = tile_box.smallEnd(2);
            const int k_max_fab = tile_box.bigEnd(2);

            for (int k_loop_idx = k_min_fab; k_loop_idx <= k_max_fab; ++k_loop_idx) {
                tsize_t actual_bytes_per_scanline_for_1bit = 0;
                TIFF* current_tif_ptr = nullptr;

                if (m_is_sequence) {
                    std::string current_filename = generateFilename(m_base_pattern, m_start_index + k_loop_idx, m_digits, m_suffix);
                    thread_local_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!thread_local_tif_handle) { amrex::Abort("[TiffReader] Seq: Failed to open: " + current_filename); }
                    current_tif_ptr = thread_local_tif_handle.get();
                } else { // Stack file
                    thread_local_tif_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
                    if (!thread_local_tif_handle) { amrex::Abort("[TiffReader] Stack: Failed to open: " + m_filename + " in thread."); }
                    current_tif_ptr = thread_local_tif_handle.get();
                    if (TIFFCurrentDirectory(current_tif_ptr) != static_cast<tdir_t>(k_loop_idx)) {
                        if (!TIFFSetDirectory(current_tif_ptr, static_cast<tdir_t>(k_loop_idx))) {
                            amrex::Abort("[TiffReader] Stack: Failed to set dir " + std::to_string(k_loop_idx));
                        }
                    }
                }
                
                if (bits_per_sample_val == 1) {
                    actual_bytes_per_scanline_for_1bit = TIFFScanlineSize(current_tif_ptr);
                    if (amrex::Verbose() >= 2 && omp_get_thread_num() == 0 && mfi.LocalTileIndex() == 0 && amrex::ParallelDescriptor::IOProcessor()) {
                         amrex::Print() << "DEBUG_SCANLINE_SIZE: Z-slice " << k_loop_idx
                                   << ", TIFFScanlineSize() reports: " << actual_bytes_per_scanline_for_1bit
                                   << " bytes/scanline. (ImgWidth=" << image_width_const
                                   << ", TheoryMinBytes: " << (image_width_const + 7) / 8 << ")\n";
                    }
                     AMREX_ALWAYS_ASSERT_WITH_MESSAGE(actual_bytes_per_scanline_for_1bit > 0 || image_width_const == 0, 
                                                 "actual_bytes_per_scanline_for_1bit is zero for 1-bit image with non-zero width!");
                }

                bool is_tiled = TIFFIsTiled(current_tif_ptr);

                if (is_tiled) {
                    uint32_t tile_w=0, tile_h=0; tsize_t tile_buf_size=0;
                    TIFFGetField(current_tif_ptr, TIFFTAG_TILEWIDTH, &tile_w);
                    TIFFGetField(current_tif_ptr, TIFFTAG_TILELENGTH, &tile_h);
                    tile_buf_size = TIFFTileSize(current_tif_ptr);
                    if(tile_w==0||tile_h==0||tile_buf_size<=0) amrex::Abort("Invalid tile params.");
                    if(temp_buffer.size()< (size_t)tile_buf_size) temp_buffer.resize(tile_buf_size);
                    const int tile_width_int = (int)tile_w; const int tile_height_int = (int)tile_h;
                    int tx_min = tile_box.smallEnd(0)/tile_width_int; int tx_max = tile_box.bigEnd(0)/tile_width_int;
                    int ty_min = tile_box.smallEnd(1)/tile_height_int; int ty_max = tile_box.bigEnd(1)/tile_height_int;

                    for (int ty = ty_min; ty <= ty_max; ++ty) {
                        for (int tx = tx_min; tx <= tx_max; ++tx) {
                            int chk_ox = tx*tile_width_int; int chk_oy = ty*tile_height_int;
                            amrex::Box chk_box(amrex::IntVect(chk_ox,chk_oy,k_loop_idx), amrex::IntVect(chk_ox+tile_width_int-1, chk_oy+tile_height_int-1, k_loop_idx));
                            amrex::Box insect = tile_box & chk_box;
                            if (insect.ok()) {
                                ttile_t tile_idx = TIFFComputeTile(current_tif_ptr, chk_ox, chk_oy, 0, 0);
                                tsize_t bytes_rd = TIFFReadEncodedTile(current_tif_ptr, tile_idx, temp_buffer.data(), tile_buf_size);
                                if(bytes_rd < 0) amrex::Abort("Error reading tile " + std::to_string(tile_idx));
                                amrex::LoopOnCpu(insect, [&](int i, int j, int k_fab){
                                    double val_dbl=0.0; int bit_v=-1; unsigned char p_byte=0; size_t byte_i=0; int bit_i=0; size_t lin_i_chk=0;
                                    if(bits_per_sample_val==1){
                                        int i_chk=i-chk_ox; int j_chk=j-chk_oy;
                                        lin_i_chk = (size_t)j_chk*tile_width_int + i_chk; // Use tile_width_int for tile calculations
                                        byte_i=lin_i_chk/8; bit_i=lin_i_chk%8;
                                        if(byte_i<(size_t)bytes_rd){ p_byte=temp_buffer[byte_i]; bit_v=(fill_order_local==FILLORDER_MSB2LSB)?(p_byte>>(7-bit_i))&1:(p_byte>>bit_i)&1; val_dbl=(double)bit_v; }
                                        else { val_dbl=0.0; bit_v=-2; }
                                    } else {
                                        int i_chk=i-chk_ox; int j_chk=j-chk_oy;
                                        size_t off = ((size_t)j_chk*tile_width_int + i_chk)*bytes_per_pixel; // Use tile_width_int
                                        if(off+bytes_per_sample_calc <= (size_t)bytes_rd){ val_dbl=interpretBytesAsDouble(temp_buffer.data()+off, bits_per_sample_val, m_sample_format); }
                                        else { val_dbl=0.0; }
                                    }
                                    if (amrex::Verbose()>=3 && amrex::ParallelDescriptor::IOProcessor()){ bool bnd=(i==image_width_const-1||j==image_height_const-1||k_fab==image_depth_const-1||i==0||j==0||k_fab==0); if(bnd&&bits_per_sample_val==1) {amrex::Print().SetPrecision(0)<<"TIFF_DBG(Tile): V("<<i<<","<<j<<","<<k_fab<<") T("<<tx<<","<<ty<<") LinChkIdx("<<lin_i_chk<<") ByteIdx("<<byte_i<<") BitInByte("<<bit_i<<") PByte(0x"<<std::hex<<(int)p_byte<<std::dec<<") RawBit("<<bit_v<<") Thr("<<((val_dbl>raw_threshold)?value_if_true:value_if_false)<<")\n";}}
                                    fab_arr(i,j,k_fab) = (val_dbl > raw_threshold) ? value_if_true : value_if_false;
                                });
                            }
                        }
                    }
                } else { // Striped Reading
                    uint32_t rps=0; uint32_t cur_h32=(uint32_t)m_height; tsize_t strip_buf_size=0;
                    TIFFGetFieldDefaulted(current_tif_ptr, TIFFTAG_ROWSPERSTRIP, &rps);
                    strip_buf_size = TIFFStripSize(current_tif_ptr);
                    if(rps==0||rps>cur_h32) rps=cur_h32;
                    if(strip_buf_size<=0) amrex::Abort("Invalid strip buffer size.");
                    if(temp_buffer.size()<(size_t)strip_buf_size) temp_buffer.resize(strip_buf_size);

                    int fab_y_min = tile_box.smallEnd(1); int fab_y_max = tile_box.bigEnd(1);
                    tstrip_t first_strip = TIFFComputeStrip(current_tif_ptr, fab_y_min, 0);
                    tstrip_t last_strip = TIFFComputeStrip(current_tif_ptr, fab_y_max, 0);

                    for (tstrip_t strip_num = first_strip; strip_num <= last_strip; ++strip_num) {
                        uint32_t strip_oy_uint = strip_num * rps;
                        int strip_oy_img = (int)strip_oy_uint;
                        uint32_t rows_this_strip = std::min(rps, cur_h32 - strip_oy_uint);
                        if(rows_this_strip==0) continue;
                        int strip_h_px = (int)rows_this_strip;
                        amrex::Box strip_box_img(amrex::IntVect(0,strip_oy_img,k_loop_idx), amrex::IntVect(image_width_const-1, strip_oy_img+strip_h_px-1, k_loop_idx));
                        amrex::Box insect = tile_box & strip_box_img;
                        if(insect.ok()){
                            tsize_t bytes_rd = TIFFReadEncodedStrip(current_tif_ptr, strip_num, temp_buffer.data(), strip_buf_size);
                            if(bytes_rd<0) amrex::Abort("Error reading strip " + std::to_string(strip_num));
                            amrex::LoopOnCpu(insect, [&](int i, int j, int k_fab){
                                double val_dbl=0.0; int bit_v=-1; unsigned char p_byte=0; size_t byte_i=0; int bit_i=0;
                                if(bits_per_sample_val==1){
                                    int j_in_strip_buf = j - strip_oy_img; 
                                    size_t scanline_start_off = (size_t)j_in_strip_buf * actual_bytes_per_scanline_for_1bit;
                                    size_t byte_off_in_scanline = (size_t)i / 8;
                                    bit_i = i % 8;
                                    byte_i = scanline_start_off + byte_off_in_scanline;
                                    if(byte_i<(size_t)bytes_rd){ p_byte=temp_buffer[byte_i]; bit_v=(fill_order_local==FILLORDER_MSB2LSB)?(p_byte>>(7-bit_i))&1:(p_byte>>bit_i)&1; val_dbl=(double)bit_v; }
                                    else { val_dbl=0.0; bit_v=-2; }
                                } else {
                                    int j_in_strip_buf = j - strip_oy_img;
                                    size_t off = ((size_t)j_in_strip_buf * image_width_const + i) * bytes_per_pixel;
                                    if(off+bytes_per_sample_calc <= (size_t)bytes_rd){ val_dbl=interpretBytesAsDouble(temp_buffer.data()+off, bits_per_sample_val, m_sample_format); }
                                    else { val_dbl=0.0; }
                                }
                                if (amrex::Verbose()>=3 && amrex::ParallelDescriptor::IOProcessor()){ bool bnd=(i==image_width_const-1||j==image_height_const-1||k_fab==image_depth_const-1||i==0||j==0||k_fab==0); if(bnd&&bits_per_sample_val==1) {amrex::Print().SetPrecision(0)<<"TIFF_DBG(Strip): V("<<i<<","<<j<<","<<k_fab<<") Strp("<<strip_num<<") ByteIdxInBuf("<<byte_i<<") BitInByte("<<bit_i<<") PByte(0x"<<std::hex<<(int)p_byte<<std::dec<<") RawBit("<<bit_v<<") Thr("<<((val_dbl > raw_threshold) ? value_if_true : value_if_false)<<")\n";}}
                                fab_arr(i,j,k_fab) = (val_dbl > raw_threshold) ? value_if_true : value_if_false;
                            });
                        }
                    }
                }
                // thread_local_tif_handle (and its TIFF*) will be closed when TiffPtr goes out of scope at the end of this k_loop_idx iteration
            } // End k_loop_idx Z-slices
        } // End MFIter
    } // End OMP parallel region
    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");
}

//================================================================
// Public threshold methods
//================================================================
void TiffReader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const {
    readDistributedIntoFab(mf, value_if_true, value_if_false, raw_threshold);
}

void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    threshold(raw_threshold, 1, 0, mf);
}

} // namespace OpenImpala
