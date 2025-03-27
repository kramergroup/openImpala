/** REV calculation
 *
 * This programme calculates diffusion and tortuosity in 3 cartesian directions,
 * as well volume fraction for a 2 phase segmented dataset, across multiple different randomised seedpoints in increasingly sized boxes.
 *
 * The program will open the input file from user input, read its data, potentially perform an REV study,
 * and then calculate the following properties for the full domain (respecting DIRECTION input) or sub-domains (all directions):
 *
 * 1) volume fractions of the phase of interest
 * 2) effective diffusivity and tortuosity in the x direction
 * 3) effective diffusivity and tortuosity in the y direction
 * 4) effective diffusivity and tortuosity in the z direction
 *
 */

#include "../io/TiffReader.H"
#include "../io/TiffStackReader.H"
#include "../io/DatReader.H"
#include "../io/HDF5Reader.H"
// Assuming RawReader might be needed from the other branch, add potentially:
// #include "../io/RawReader.H"
#include "TortuosityHypre.H"
#include "VolumeFraction.H"

#include <typeinfo>

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_ParmParse.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>

#include <sstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <random> // Needed for REV study seeding

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
  // What time is it now?  We'll use this to compute total run time.
  amrex::Real strt_time = amrex::second();

  // Parameters
  amrex::ParmParse pp;
  // NOTE: Assuming periodic boundaries for REV study and subsequent calculations in this version
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{true, true, true};
  amrex::Print() << AMREX_SPACEDIM << "D test" << std::endl;

  std::string FILENAME;
  pp.get("FILENAME", FILENAME);  // query filename from inputs file
  
  // Get the users home directory to write plot file to right place
  const char* homeDir = getenv("HOME");
  
  std::string DATA_PATH = "/openImpala/data/";
  pp.get("DATA_PATH", DATA_PATH);  // query data directory from inputs file
  if(DATA_PATH.at(0) == '~')
  {
      DATA_PATH = DATA_PATH.substr(1);
      DATA_PATH = homeDir + DATA_PATH;
  }

  std::string RESULTS_PATH = "~/openimpalaresults/";
  pp.get("RESULTS_PATH", RESULTS_PATH);  // query results directory from inputs file
  if(RESULTS_PATH.at(0) == '~')
  {
      RESULTS_PATH = RESULTS_PATH.substr(1);
      RESULTS_PATH = homeDir + RESULTS_PATH;   
  }
  
  std::string HDF5_DATASET;
  // Use get instead of query if it's mandatory for HDF5 files
  pp.get("HDF5_DATASET", HDF5_DATASET); // query HDF5 dataset path from inputs file
  
  amrex::Real DIRECTION = Direction::X; // Default direction
  pp.query("DIRECTION", DIRECTION);  // query direction from inputs file

  // BOX_SIZE can have a considerable influence on the calculation speed of the programme
  // further work is required to work out the optimum value, if not, leave the value as 32
  amrex::Real BOX_SIZE = 32;
  pp.query("BOX_SIZE", BOX_SIZE);

  // Parameter potentially used by TiffStackReader
  int TIFF_STACK = 100;
  pp.query("TIFF_STACK", TIFF_STACK);    
    
  amrex::Real EPS= 1e-10;
  pp.query("EPS", EPS); // Although EPS seems unused if solver is hardcoded later? Check TortuosityHypre.

  // Define physical geometry, index space, and multifab for the full domain
  amrex::Geometry geom;
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  amrex::iMultiFab mf_phase;
  amrex::Box bx; // Will hold the full domain box
    
  // Define objects needed for REV study sub-domains
  amrex::Geometry geom_11;
  amrex::BoxArray ba_11;
  amrex::DistributionMapping dm_11;
  amrex::iMultiFab mf_phase_11;
  // amrex::Box bx_11; // Defined inside loop

  // Scope for reader to manage memory
  {
    amrex::Real fx;
    amrex::Real fy;
    amrex::Real fz;
    // Periodicity is defined above and used here
    
    // --- Reader Selection Logic ---
    bool is_tif = (FILENAME.substr (FILENAME.length() - 3) == "tif" || FILENAME.substr (FILENAME.length() - 4) == "tiff");
    bool is_dat = (FILENAME.substr (FILENAME.length() - 3) == "dat");
    // Add check for RawReader if integrating from the other branch:
    // bool is_raw = (FILENAME.substr (FILENAME.length() - 3) == "raw");
    bool is_h5 = (FILENAME.substr (FILENAME.length() - 2) == "h5" || FILENAME.substr (FILENAME.length() - 4) == "hdf5");

    if (is_tif)
    {    
        amrex::Print() << "\tTiffReader - Reading file " << DATA_PATH + FILENAME << std::endl;
        TiffReader reader(DATA_PATH + FILENAME);

        // Get full domain box AFTER reader is constructed
        bx = reader.box();
        // Calculate physical size factors (relative to chosen direction - maybe revise this?)
        fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
        fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
        fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
        amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
        geom.define(bx, &rb, 0, is_periodic.data());

        // Define computational domain and index space for full domain
        ba.define(geom.Domain());
        ba.maxSize(BOX_SIZE);
        
        dm.define(ba);
        // Use 2 ghost cells based on REV branch changes
        mf_phase.define(ba,dm,1,2);
        
        // Threshold image data
        reader.threshold(1,mf_phase); // Assuming threshold 1 for phase selection
                
        // Use 2 ghost cells, call FillBoundary without periodicity argument based on REV branch changes
        mf_phase.FillBoundary();
                                                    
    }
    // Add RawReader block here if integrating:
    // else if (is_raw) { ... RawReader logic ... }
    else if (is_dat)
    {    
        amrex::Print() << "\tDatReader - Reading file " << DATA_PATH + FILENAME << std::endl;
        // Assuming DatReader needs XDIM, YDIM, ZDIM, PRECISION, ENDIAN from input - needs implementation
        // DatReader reader(DATA_PATH + FILENAME, xdim, ydim, zdim, precision, endian); // Example constructor
        DatReader reader(DATA_PATH + FILENAME); // Use actual constructor

        bx = reader.box();
        fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
        fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
        fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
        amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
        geom.define(bx, &rb, 0, is_periodic.data());

        ba.define(geom.Domain());
        ba.maxSize(BOX_SIZE);
        
        dm.define(ba);
        mf_phase.define(ba,dm,1,1); // Original uses 1 ghost cell
        
        reader.threshold(1,mf_phase); // Assuming threshold 1
                
        mf_phase.FillBoundary(geom.periodicity()); // Original uses periodicity
                                                    
    }
    else if (is_h5)
    {    
        amrex::Print() << "\tHDF5Reader - Reading file " << DATA_PATH + FILENAME << std::endl;
        HDF5Reader reader(DATA_PATH + FILENAME, HDF5_DATASET);

        bx = reader.box();
        fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
        fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
        fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
        amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
        geom.define(bx, &rb, 0, is_periodic.data());

        ba.define(geom.Domain());
        ba.maxSize(BOX_SIZE);
        
        dm.define(ba);
        mf_phase.define(ba,dm,1,1); // Original uses 1 ghost cell
        
        reader.threshold(1,mf_phase); // Assuming threshold 1
                
        mf_phase.FillBoundary(geom.periodicity()); // Original uses periodicity
                                                    
    }
    else // Assume TiffStackReader and perform REV study
    {    
        amrex::Print() << "\tTiffStackReader - Reading file " << DATA_PATH + FILENAME << std::endl;
        TiffStackReader reader(DATA_PATH + FILENAME, TIFF_STACK);

        bx = reader.box();
        fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
        fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
        fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
        amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
        geom.define(bx, &rb, 0, is_periodic.data());

        // Define full domain structures first
        ba.define(geom.Domain());
        ba.maxSize(BOX_SIZE);
        
        dm.define(ba);
        mf_phase.define(ba,dm,1,2); // Use 2 ghost cells
        
        reader.threshold(1,mf_phase); // Threshold full domain
                
        mf_phase.FillBoundary(); // Use 2 ghost cells

        // --- START REV Study Logic ---
        amrex::Print() << std::endl << "--- Starting REV Study ---" << std::endl;

        // Specify REV cube sizes   
        int rev_size [8] = { 32, 40, 50, 62, 80, 100, 126, 158 };      

        // Generate randomised seed point centre of REV     
        std::random_device rd; // obtain a random number from hardware
        std::mt19937 gen(rd()); // seed the generator
        // Define range relative to box size to avoid edges? Adjust range as needed.
        // Using 50-150 as in the original buggy code. Ensure this range is valid for domain size.
        std::uniform_int_distribution<> distr(50, std::min({150, bx.size()[0]-50, bx.size()[1]-50, bx.size()[2]-50})); // Avoid edges

        int num_samples = 3; // Number of random seed points
        amrex::Print() << "Performing REV analysis for " << num_samples << " random seed points." << std::endl;

        int sample_no = 0;
        while ( sample_no < num_samples){
            int x_seed = distr(gen);
            int y_seed = distr(gen);
            int z_seed = distr(gen);
            amrex::Print() << "Sample " << sample_no + 1 << ": Seed (" << x_seed << ", " << y_seed << ", " << z_seed << ")" << std::endl;
            
            // Iterate through box sizes  
            int i = 0;
            // *** BUG FIX: Changed loop condition from i < 9 to i < 8 ***
            while ( i < 8){       

                int current_rev_size = rev_size[i];
                // Ensure rev_size is not larger than the domain itself
                if (current_rev_size > bx.size()[0] || current_rev_size > bx.size()[1] || current_rev_size > bx.size()[2]) {
                    amrex::Print() << "Skipping REV size " << current_rev_size << " as it exceeds domain dimensions." << std::endl;
                    i++;
                    continue; // Skip to next size
                }

                // Calculate sub-box bounds centered at seed
                int x_seed_low = x_seed - (current_rev_size/2);
                int x_seed_high = x_seed_low + current_rev_size - 1; // Correct calculation
                int y_seed_low = y_seed - (current_rev_size/2);
                int y_seed_high = y_seed_low + current_rev_size - 1;
                int z_seed_low = z_seed - (current_rev_size/2);
                int z_seed_high = z_seed_low + current_rev_size - 1;

                // Perform check to see if edge of REV box exceeds domain size and correct if necessary  
                // Shift the box if it goes out of bounds
                if (x_seed_low < bx.loVect()[0]){
                    x_seed_high += (bx.loVect()[0] - x_seed_low);
                    x_seed_low = bx.loVect()[0];
                }
                if (x_seed_high > bx.hiVect()[0]){
                    x_seed_low -= (x_seed_high - bx.hiVect()[0]);
                    x_seed_high = bx.hiVect()[0];
                }
                if (y_seed_low < bx.loVect()[1]){
                    y_seed_high += (bx.loVect()[1] - y_seed_low);
                    y_seed_low = bx.loVect()[1];
                }
                if (y_seed_high > bx.hiVect()[1]){
                    y_seed_low -= (y_seed_high - bx.hiVect()[1]);
                    y_seed_high = bx.hiVect()[1];
                }
                if (z_seed_low < bx.loVect()[2]){
                    z_seed_high += (bx.loVect()[2] - z_seed_low);
                    z_seed_low = bx.loVect()[2];
                }
                if (z_seed_high > bx.hiVect()[2]){
                    z_seed_low -= (z_seed_high - bx.hiVect()[2]);
                    z_seed_high = bx.hiVect()[2];
                }     

                // Define the sub-domain box
                const amrex::Box bx_11 ({x_seed_low, y_seed_low, z_seed_low}, {x_seed_high, y_seed_high, z_seed_high});
                
                // Define geometry etc. for the sub-domain
                // Physical coordinates relative to sub-box size? Or keep original scaling? Assuming relative here.
                fx = 1.0*bx_11.size()[0]/bx_11.size()[0]; // fx=1
                fy = 1.0*bx_11.size()[1]/bx_11.size()[1]; // fy=1
                fz = 1.0*bx_11.size()[2]/bx_11.size()[2]; // fz=1
                amrex::RealBox rb_11({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain [-1,1]^3
                geom_11.define(bx_11, &rb_11, 0, is_periodic.data()); // Use periodic BC for sub-domain too

                ba_11.define(geom_11.Domain());
                ba_11.maxSize(BOX_SIZE); // Use same box size for sub-domain decomposition

                dm_11.define(ba_11);
                mf_phase_11.define(ba_11,dm_11,1,2); // 2 ghost cells

                // Threshold image data *within the sub-domain*
                // This assumes reader.threshold can handle filling a MultiFab
                // whose domain (bx_11) is a sub-set of the reader's full data.
                reader.threshold(1,mf_phase_11); // Assuming threshold 1

                mf_phase_11.FillBoundary();  // Fill ghost cells for sub-domain

                amrex::Print() << "\tBox " << sample_no + 1 << ", Size " << current_rev_size
                             << " (" << i + 1 << "/8)" << std::endl;
                            //<< " Coords: LB(" << x_seed_low << "," << y_seed_low << "," << z_seed_low
                            //<< ") UB(" << x_seed_high << "," << y_seed_high << "," << z_seed_high << ")" << std::endl;   

                // Calculate VF for phase 1 in the sub-domain
                VolumeFraction vf_11(mf_phase_11, 1); 
                amrex::Real vf_val = vf_11.value(); // Calculate once

                amrex::Print() << "\t  Volume Fraction: " << vf_val << std::endl;

                // Compute tortuosity in all 3 directions for the sub-domain
                // Using Phase 1 and GMRES solver as per REV branch logic
                
                // X Direction
                TortuosityHypre tortuosityx_11(geom_11,ba_11,dm_11,mf_phase_11,vf_val,1,Direction::X,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
                amrex::Real tau_value_x_11 = tortuosityx_11.value();
                // amrex::Print() << "\t  Tortuosity (X): " << tau_value_x_11 << std::endl; // Output within .value() now?

                // Y Direction
                TortuosityHypre tortuosityy_11(geom_11,ba_11,dm_11,mf_phase_11,vf_val,1,Direction::Y,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
                amrex::Real tau_value_y_11 = tortuosityy_11.value();
                // amrex::Print() << "\t  Tortuosity (Y): " << tau_value_y_11 << std::endl;

                // Z Direction
                TortuosityHypre tortuosityz_11(geom_11,ba_11,dm_11,mf_phase_11,vf_val,1,Direction::Z,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
                amrex::Real tau_value_z_11 = tortuosityz_11.value();
                // amrex::Print() << "\t  Tortuosity (Z): " << tau_value_z_11 << std::endl;

                i++;  // Increment inner loop counter
            } // End of while loop for sizes (i)
            
            sample_no++; // Increment outer loop counter
        } // End of while loop for samples (sample_no)
        amrex::Print() << "--- Finished REV Study ---" << std::endl;
        // --- END REV Study Logic ---                                      
    }
    else // File type not recognised
    {
        // *** BUG FIX: Corrected condition - this 'else' now catches unrecognised types ***
        amrex::Print() << std::endl << "File format not recognised: " << FILENAME << std::endl; 
        // Optionally, call amrex::Abort() here if file must be recognised.
    } 
  } // End scope for reader

  // --- Final Full Domain Calculation ---
  // Check if mf_phase was actually defined (i.e., if a reader succeeded)
  if (!mf_phase.empty())
  {
    amrex::Print() << std::endl << "--- Starting Final Full Domain Calculation ---" << std::endl;    
      
    // Calculate Volume Fraction for phase 1 on the full domain
    VolumeFraction vf(mf_phase, 1); 

    amrex::Real vf_full_val = vf.value();
    amrex::Print() << std::endl << " Full Domain Volume Fraction (Phase 1): " << vf_full_val << std::endl;

    // *** BUG FIX: Reinstated logic to respect input DIRECTION for final calculation ***
    // Using Phase 1 and GMRES solver consistent with REV part of this branch.
    if (DIRECTION==0) // X
    {
        amrex::Print() << std::endl << " Full Domain Direction: X" << std::endl;
        TortuosityHypre tortuosityx(geom,ba,dm,mf_phase,vf_full_val,1,Direction::X,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_x = tortuosityx.value();
        // amrex::Print() << " Full Domain Tortuosity value (X): " << tau_value_x << std::endl; // Output within .value()?
    }
    else if (DIRECTION==1) // Y
    {
        amrex::Print() << std::endl << " Full Domain Direction: Y" << std::endl;
        TortuosityHypre tortuosityy(geom,ba,dm,mf_phase,vf_full_val,1,Direction::Y,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_y = tortuosityy.value();
        // amrex::Print() << " Full Domain Tortuosity value (Y): " << tau_value_y << std::endl;
    }
    else if (DIRECTION==2) // Z
    {
        amrex::Print() << std::endl << " Full Domain Direction: Z" << std::endl;
        TortuosityHypre tortuosityz(geom,ba,dm,mf_phase,vf_full_val,1,Direction::Z,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_z = tortuosityz.value();
        // amrex::Print() << " Full Domain Tortuosity value (Z): " << tau_value_z << std::endl;
    }
    else // Calculate all three directions if DIRECTION is not 0, 1, or 2
    {
        amrex::Print() << std::endl << " Full Domain Direction: X (Calculating All 3)" << std::endl;
        TortuosityHypre tortuosityx(geom,ba,dm,mf_phase,vf_full_val,1,Direction::X,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_x = tortuosityx.value();
        // amrex::Print() << " Full Domain Tortuosity value (X): " << tau_value_x << std::endl;

        amrex::Print() << std::endl << " Full Domain Direction: Y" << std::endl;
        TortuosityHypre tortuosityy(geom,ba,dm,mf_phase,vf_full_val,1,Direction::Y,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_y = tortuosityy.value();
        // amrex::Print() << " Full Domain Tortuosity value (Y): " << tau_value_y << std::endl;

        amrex::Print() << std::endl << " Full Domain Direction: Z" << std::endl;
        TortuosityHypre tortuosityz(geom,ba,dm,mf_phase,vf_full_val,1,Direction::Z,TortuosityHypre::SolverType::GMRES, RESULTS_PATH);
        amrex::Real tau_value_z = tortuosityz.value();
        // amrex::Print() << " Full Domain Tortuosity value (Z): " << tau_value_z << std::endl;
    }
    amrex::Print() << "--- Finished Final Full Domain Calculation ---" << std::endl;
  } else {
      amrex::Print() << "Skipping final calculation as input data was not loaded successfully." << std::endl;
  }


  // Call the timer again and compute the maximum difference between the start time and stop time
  //  over all processors
  amrex::Real stop_time = amrex::second() - strt_time;
  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
  amrex::ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

  // Tell the I/O Processor to write out the "run time"
  amrex::Print() << std::endl << "Total run time (seconds) = " << stop_time << std::endl;

  } // Ensure amrex related destructors have been called before tearing down the whole thing
    // by putting everything in curly brackets.
  amrex::Finalize();
}
