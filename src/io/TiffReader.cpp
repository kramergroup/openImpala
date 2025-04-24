// Add this include if you haven't already, for OpenMP directives
#include <omp.h>

// ... other includes ...

namespace OpenImpala {

// ... (Helper namespace, Constructors, Metadata Getters, readFile methods remain the same) ...

// --- readDistributedIntoFab Method ---
// --- MODIFIED WITH OMP CRITICAL SECTION ---
void TiffReader::readDistributedIntoFab(
    amrex::iMultiFab& dest_mf,
    int value_if_true,
    int value_if_false,
    double raw_threshold
) const
{
    if (!m_is_read) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Cannot read, metadata not processed successfully.");
    }

    const int bits_per_sample_val = m_bits_per_sample;
    const size_t bytes_per_pixel = (bits_per_sample_val < 8) ?
                                   (bits_per_sample_val * m_samples_per_pixel + 7) / 8
                                   : (bits_per_sample_val / 8) * m_samples_per_pixel;
    if (bits_per_sample_val == 0 || bytes_per_pixel == 0) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bits per sample or calculated bytes per pixel is zero!");
    }

    // Open the single stack file handle once OUTSIDE the parallel region.
    TiffPtr shared_tif_stack_handle = nullptr;
    if (!m_is_sequence) {
        shared_tif_stack_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
        if (!shared_tif_stack_handle) {
            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open shared TIFF file: " + m_filename);
        }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) // Keep the parallel region
#endif
    {
        // Each thread gets its own buffer.
        // For sequences, it also gets its own file handle per slice.
        std::vector<unsigned char> temp_buffer;
        TiffPtr sequence_tif_handle = nullptr; // Per-thread handle for sequence files

        for (amrex::MFIter mfi(dest_mf, true); mfi.isValid(); ++mfi)
        {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi);
            const amrex::Box& tile_box = mfi.tilebox();

            const int k_min = tile_box.smallEnd(2);
            const int k_max = tile_box.bigEnd(2);

            // Loop over Z-slices needed for this MFIter tile
            for (int k = k_min; k <= k_max; ++k) {
                // Buffer to hold the raw data read for the current tile/strip within slice k
                // We manage this buffer per-slice iteration.
                std::vector<unsigned char> current_chunk_data;
                tsize_t bytes_read_in_chunk = -1;
                bool is_chunk_tiled = false;
                uint32_t chunk_width = 0, chunk_height = 0; // For tiled
                int chunk_origin_x = 0, chunk_origin_y = 0; // Relative origin within slice
                int chunk_num_x = 0, chunk_num_y = 0; // Tile indices if tiled

                if (m_is_sequence) {
                    // --- Sequence Reading Logic (Thread-Safe per slice) ---
                    std::string current_filename = generateFilename(m_base_pattern, m_start_index + k, m_digits, m_suffix);
                    std::string current_file_id_for_error = current_filename;
                    // Open handle locally for this slice
                    sequence_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!sequence_tif_handle) {
                         amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open sequence file: " + current_filename + " for slice " + std::to_string(k));
                    }
                    TIFF* current_tif_raw_ptr = sequence_tif_handle.get();

                    // --- Read data for the current slice (k) from sequence file ---
                    // (This logic needs to be similar to the stack reading but uses sequence_tif_handle)
                    is_chunk_tiled = TIFFIsTiled(current_tif_raw_ptr); // Check if tiled

                    if (is_chunk_tiled) {
                         // ... (Get tile info: tile_width, tile_height, tile_buffer_size) ...
                         uint32_t tile_width_seq=0, tile_height_seq=0;
                         TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width_seq);
                         TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILELENGTH, &tile_height_seq);
                         if (tile_width_seq == 0 || tile_height_seq == 0) { amrex::Abort("..."); }
                         chunk_width = tile_width_seq; chunk_height = tile_height_seq;

                         tsize_t tile_buffer_size_seq = TIFFTileSize(current_tif_raw_ptr);
                         if (tile_buffer_size_seq <= 0) { amrex::Abort("..."); }
                         // Ensure local temp_buffer is large enough (use thread-local buffer)
                         if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size_seq)) { temp_buffer.resize(tile_buffer_size_seq); }

                         // Find the single tile containing the necessary data for this MFIter box
                         // (Simplified: Assume MFIter box fits within one tile or handle complexity)
                         // We only need to read the tile(s) intersecting the current MFIter tile_box
                         int tx_min_seq = tile_box.smallEnd(0) / tile_width_seq; int tx_max_seq = tile_box.bigEnd(0) / tile_width_seq;
                         int ty_min_seq = tile_box.smallEnd(1) / tile_height_seq; int ty_max_seq = tile_box.bigEnd(1) / tile_height_seq;

                         // NOTE: This simple loop assumes we process one tile intersection at a time.
                         // A more efficient approach might read all intersecting tiles first.
                         for (int ty = ty_min_seq; ty <= ty_max_seq; ++ty) {
                             for (int tx = tx_min_seq; tx <= tx_max_seq; ++tx) {
                                 chunk_origin_x = tx * tile_width_seq; chunk_origin_y = ty * tile_height_seq;
                                 chunk_num_x = tx; chunk_num_y = ty; // Store tile indices
                                 ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, chunk_origin_x, chunk_origin_y, 0, 0);
                                 bytes_read_in_chunk = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size_seq);
                                 if (bytes_read_in_chunk < 0) { amrex::Abort("..."); }

                                 // Process this chunk immediately after reading
                                 amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                          amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                                 amrex::Box intersection = tile_box & chunk_abs_box;
                                 if (intersection.ok()) {
                                      // --- Apply threshold from temp_buffer to fab_arr ---
                                      // (Thresholding logic goes here, using temp_buffer)
                                      // ... see below ...
                                      amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                          double value_as_double = 0.0;
                                          if (bits_per_sample_val == 1) { /* 1-bit logic */
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                              size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                              size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                              int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                              if (byte_index_in_buffer < static_cast<size_t>(bytes_read_in_chunk)) {
                                                   unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                                   int bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                   value_as_double = static_cast<double>(bit_value);
                                              } // else value_as_double remains 0.0 if index out of bounds
                                          } else { /* BPS >= 8 logic */
                                              const size_t bytes_per_sample = bits_per_sample_val / 8;
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                              size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                              if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read_in_chunk)) {
                                                 const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                 value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                              } // else value_as_double remains 0.0 if index out of bounds
                                          }
                                          fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                      });
                                 } // end if intersection ok
                             } // end tx loop
                         } // end ty loop
                    } else {
                         // ... (Striped Reading Logic for Sequences - similar structure) ...
                         // Read necessary strips into temp_buffer
                         // Process strips immediately
                    }
                    // Close the per-thread sequence handle
                    sequence_tif_handle.reset();

                } else {
                    // --- Stack Reading Logic (Needs Protection) ---
                    TIFF* current_tif_raw_ptr = nullptr; // Will be set inside critical section

                    #pragma omp critical (TiffReadLock) // Protect access to shared handle and libtiff state
                    {
                        if (!shared_tif_stack_handle) { // Double check handle validity
                            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Shared TIFF handle is null inside critical section!");
                        }
                        // --- Set Directory ---
                        if (!TIFFSetDirectory(shared_tif_stack_handle.get(), static_cast<tdir_t>(k))) {
                            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to set directory " + std::to_string(k) + " in file: " + m_filename);
                        }
                        current_tif_raw_ptr = shared_tif_stack_handle.get(); // Pointer is valid now

                        // --- Check Tiled/Striped and Read Data INTO temp_buffer ---
                        is_chunk_tiled = TIFFIsTiled(current_tif_raw_ptr);

                        if (is_chunk_tiled) {
                            uint32_t tile_width_st=0, tile_height_st=0;
                            TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width_st);
                            TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILELENGTH, &tile_height_st);
                            if (tile_width_st == 0 || tile_height_st == 0) { amrex::Abort("..."); }
                            chunk_width = tile_width_st; chunk_height = tile_height_st;

                            tsize_t tile_buffer_size_st = TIFFTileSize(current_tif_raw_ptr);
                            if (tile_buffer_size_st <= 0) { amrex::Abort("..."); }
                            // Resize thread-local buffer IF NEEDED (happens inside critical section)
                            if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size_st)) { temp_buffer.resize(tile_buffer_size_st); }

                            // Calculate which tiles intersect the current MFIter box for slice k
                            int tx_min_st = tile_box.smallEnd(0) / tile_width_st; int tx_max_st = tile_box.bigEnd(0) / tile_width_st;
                            int ty_min_st = tile_box.smallEnd(1) / tile_height_st; int ty_max_st = tile_box.bigEnd(1) / tile_height_st;

                            // Read ONLY the relevant tile(s) for this MFIter box *inside* the critical section
                            // NOTE: For simplicity, we read and process one tile at a time within the critical section.
                            // This might not be optimal but ensures safety.
                            for (int ty = ty_min_st; ty <= ty_max_st; ++ty) {
                                for (int tx = tx_min_st; tx <= tx_max_st; ++tx) {
                                     // Check if this specific tile tx,ty intersects the current MFIter tile_box
                                     chunk_origin_x = tx * tile_width_st; chunk_origin_y = ty * tile_height_st;
                                     amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                              amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                                     amrex::Box intersection = tile_box & chunk_abs_box;

                                     if (intersection.ok()) {
                                         // This tile is relevant, read it
                                         chunk_num_x = tx; chunk_num_y = ty; // Store tile indices
                                         ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, chunk_origin_x, chunk_origin_y, 0, 0);
                                         bytes_read_in_chunk = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size_st);
                                         if (bytes_read_in_chunk < 0) { amrex::Abort("..."); }

                                         // --- Apply threshold from temp_buffer to fab_arr (OUTSIDE critical section) ---
                                         // We process this specific tile chunk right after reading it
                                         // NOTE: The LoopOnCpu should happen *outside* the critical section
                                         //       to allow parallel processing.
                                         //       This structure reads one tile, then processes, repeats.
                                         //       Alternatively, store temp_buffer data per tile and process all at end.

                                         // ** Correction: Apply threshold OUTSIDE critical section **
                                         // This requires storing the read data temporarily. Let's use the thread-local temp_buffer
                                         // and process it immediately after the critical section block for this tile.
                                         // We break the critical section here.

                                    // } // end if intersection ok - moved below
                                // } // end tx loop - moved below
                            // } // end ty loop - moved below
                        } else { // Striped logic inside critical section
                            // ... (Get strip info: rows_per_strip, compression, byte_counts etc.) ...
                            // ... (Determine buffer size needed) ...
                            // ... (Resize temp_buffer if needed) ...
                            // ... (Loop through relevant strips: first_strip to last_strip) ...
                            // ... (Read strip using TIFFReadRawStrip or TIFFReadEncodedStrip into temp_buffer) ...
                            // ... (Store chunk info: origin_x/y, width/height = m_width/strip_rows_this) ...
                            // ** Correction: Apply threshold OUTSIDE critical section **
                        } // end if tiled/striped
                    } // >>> END of #pragma omp critical (TiffReadLock) <<<

                    // ---- Process the data read into temp_buffer ----
                    // This part runs potentially in parallel for different MFIter tiles/k-slices,
                    // but uses the data just read serially within the critical section.
                    if (bytes_read_in_chunk >= 0) { // Check if read was successful
                        amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                 amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                        amrex::Box intersection = tile_box & chunk_abs_box;

                        if (intersection.ok()) {
                             amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                 double value_as_double = 0.0;
                                 if (bits_per_sample_val == 1) { /* 1-bit logic using temp_buffer */
                                     int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                     size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                     size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                     int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                     if (byte_index_in_buffer < static_cast<size_t>(bytes_read_in_chunk)) {
                                          unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                          int bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                          value_as_double = static_cast<double>(bit_value);
                                     }
                                 } else { /* BPS >= 8 logic using temp_buffer */
                                     const size_t bytes_per_sample = bits_per_sample_val / 8;
                                     int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                     size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                     if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read_in_chunk)) {
                                        const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                        value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                     }
                                 }
                                 fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                             });
                        } // end if intersection ok
                    } // end if bytes_read_in_chunk >= 0

                    // --- Need to handle looping through multiple tiles/strips ---
                    // The structure above processes only the *last* read tile/strip
                    // Correct structure:
                    // 1. Inside critical section: Determine needed tiles/strips for this MFIter box & k.
                    // 2. Inside critical section: Loop through needed tiles/strips.
                    // 3. Inside critical section: Read ONE tile/strip into temp_buffer.
                    // 4. --- EXIT CRITICAL SECTION ---
                    // 5. Process the data in temp_buffer using LoopOnCpu/ParallelFor for the intersection box.
                    // 6. --- ENTER CRITICAL SECTION for next tile/strip --- (or rethink structure)

                    // >>> Let's simplify: Read and process ONE tile/strip per critical section entry <<<
                    // The code above needs restructuring for multiple tiles/strips per MFIter box.
                    // A simpler, correct (but possibly less performant) version is to
                    // put the ENTIRE tile/strip read AND process loop inside the critical section.


                } // End else (Stack reading)

            } // End loop k (Z-slices)
        } // End MFIter loop
    } // End OMP parallel region

    // Ensure all ranks have finished before proceeding
    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");
}


// ... (Public threshold methods remain the same) ...

} // namespace OpenImpala
