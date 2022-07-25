/** REV calculation
 *
 * This programme calculates diffusion and tortuosity in 3 cartesian directions,
 * as well volume fraction for a 2 phase segmented dataset, across 8 different randomised seedpoints in increasingly sized boxes.
 *
 * The program will open the tiff file from user input, read its data, and then calculate the
 * following properties:
 *
 *  1) volume fractions of the phase of interest
 *  2) effective diffusivity and tortuosity in the x direction
 *  3) effective diffusivity and tortuosity in the y direction
 *  4) effective diffusivity and tortuosity in the z direction
 *
 */

#include "../io/TiffReader.H"
#include "../io/TiffStackReader.H"
#include "../io/DatReader.H"
#include "../io/HDF5Reader.H"
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

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
  // What time is it now?  We'll use this to compute total run time.
  amrex::Real strt_time = amrex::second();

  // Parameters
  amrex::ParmParse pp;
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
  pp.get("HDF5_DATASET", HDF5_DATASET); // query HDF5 dataset path from inputs file
  
  amrex::Real DIRECTION = Direction::X;
  pp.query("DIRECTION", DIRECTION);  // query direction from inputs file

  // BOX_SIZE can have a considerable influence on the calculation speed of the programme
  // further work is required to work out the optimum value, if not, leave the value as 32
  amrex::Real BOX_SIZE = 32;
  pp.query("BOX_SIZE", BOX_SIZE);

  int TIFF_STACK = 100;
  pp.query("TIFF_STACK", TIFF_STACK);    
    
  amrex::Real EPS= 1e-10;
  pp.query("EPS", EPS);

  // Define physical geometry, index space, and multifab
  amrex::Geometry geom;
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  amrex::iMultiFab mf_phase;
  amrex::Box bx;
    
  amrex::Geometry geom_11;
  amrex::BoxArray ba_11;
  amrex::DistributionMapping dm_11;
  amrex::iMultiFab mf_phase_11;
  amrex::Box bx_11;  
 
  {
    amrex::Real fx;
    amrex::Real fy;
    amrex::Real fz;
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{true, true, true};
    
    // Reading the file
    // The Reader potentially holds significant data in memory (the full voxel set).
    // The code is not parallelised, potentially creating a large memory burden per node.
    // It's best to let the reader go out of scope as soon as it is not needed anymore
    // to free up memory before further computations.
    if (FILENAME.substr (FILENAME.length() - 3) == "tif" || FILENAME.substr (FILENAME.length() - 4) == "tiff")
    {   
    amrex::Print() << "tTiffReader - Reading file " << DATA_PATH + FILENAME << std::endl;
    TiffReader reader(DATA_PATH + FILENAME);

    const amrex::Box bx = reader.box();
    fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    geom.define(bx, &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);
      
    dm.define(ba);
    mf_phase.define(ba,dm,1,2);
      
    // Threshold image data
    reader.threshold(1,mf_phase);
          
    // We have used a fab with one ghost cell to allow for stencil-type operations
    // over the fab. This requires to distribute the ghost cells
    //mf_phase.FillBoundary(geom.periodicity());
    mf_phase.FillBoundary();
                                                             
}
    else if (FILENAME.substr (FILENAME.length() - 3) == "dat")
    {   
    amrex::Print() << "tDatReader - Reading file " << DATA_PATH + FILENAME << std::endl;
    DatReader reader(DATA_PATH + FILENAME);

    const amrex::Box bx = reader.box();
    fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    geom.define(bx, &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);
      
    dm.define(ba);
    mf_phase.define(ba,dm,1,1);
      
    // Threshold image data
    reader.threshold(1,mf_phase);
          
    // We have used a fab with one ghost cell to allow for stencil-type operations
    // over the fab. This requires to distribute the ghost cells
    mf_phase.FillBoundary(geom.periodicity());
                                                             
}
    else if (FILENAME.substr (FILENAME.length() - 2) == "h5" || FILENAME.substr (FILENAME.length() - 4) == "hdf5")
    {   
    amrex::Print() << "tHDF5Reader - Reading file " << DATA_PATH + FILENAME << std::endl;
    HDF5Reader reader(DATA_PATH + FILENAME, HDF5_DATASET);

    const amrex::Box bx = reader.box();
    fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    geom.define(bx, &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);
      
    dm.define(ba);
    mf_phase.define(ba,dm,1,1);
      
    // Threshold image data
    reader.threshold(1,mf_phase);
          
    // We have used a fab with one ghost cell to allow for stencil-type operations
    // over the fab. This requires to distribute the ghost cells
    mf_phase.FillBoundary(geom.periodicity());
                                                             
}
    else if (FILENAME.at(FILENAME.length() - 3) != '.' || FILENAME.at(FILENAME.length() - 4) != '.' || FILENAME.at(FILENAME.length() - 5) != '.')
    {   
    amrex::Print() << "tTiffStackReader - Reading file " << DATA_PATH + FILENAME << std::endl;
    TiffStackReader reader(DATA_PATH + FILENAME, TIFF_STACK);

    const amrex::Box bx = reader.box();
    fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    geom.define(bx, &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);
      
    dm.define(ba);
    mf_phase.define(ba,dm,1,1);
      
    // Threshold image data
    reader.threshold(1,mf_phase);
          
    // We have used a fab with one ghost cell to allow for stencil-type operations
    // over the fab. This requires to distribute the ghost cells
    mf_phase.FillBoundary(geom.periodicity());
    
    // Specify REV cube sizes  
    int rev_size [8] = { 32, 40, 50, 62, 80, 100, 126, 158 };       
    
    

    // Generate randomised seed point centre of REV      
    srand(time(0));
    int x_seed [8] = {rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50};
    int y_seed [8] = {rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50};  
    int z_seed [8] = {rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50, rand() % 100 + 50};
      
    int i = 0;
    while ( i < 2){  
      
        // Perform check to see if edge of REV box exceeds domain size
        // and correct if necessary  
        if ((x_seed[i] - (rev_size[i]/2)) <= 0){
          x_seed[i] = (rev_size[i]/2) + 1;
        }
        if ((x_seed[i] + (rev_size[i]/2) - 1) > bx.size()[0]){
          x_seed[i] = bx.size()[0] - (rev_size[i]/2) + 1;
        }
        if ((y_seed[i] - (rev_size[i]/2)) <= 0){
          y_seed[i] = (rev_size[i]/2) + 1;
        }
        if ((y_seed[i] + (rev_size[i]/2) - 1) > bx.size()[1]){
          y_seed[i] = bx.size()[1] - (rev_size[i]/2) + 1;
        }
        if ((z_seed[i] - (rev_size[i]/2)) <= 0){
          z_seed[i] = (rev_size[i]/2) + 1;
        }
        if ((z_seed[i] + (rev_size[i]/2) - 1) > bx.size()[2]){
          z_seed[i] = bx.size()[2] - (rev_size[i]/2) + 1;
        }
      
        int x_seed_low = x_seed[i] - (rev_size[i]/2);
        int x_seed_high = x_seed[i] + (rev_size[i]/2) - 1;
        int y_seed_low = y_seed[i] - (rev_size[i]/2);
        int y_seed_high = y_seed[i] + (rev_size[i]/2) - 1;
        int z_seed_low = z_seed[i] - (rev_size[i]/2);
        int z_seed_high = z_seed[i] + (rev_size[i]/2) - 1;

        const amrex::Box bx_11 ({x_seed_low,y_seed_low,z_seed_low}, {x_seed_high,y_seed_high,z_seed_high});
        fx = 1.0*bx_11.size()[0]/bx_11.size()[DIRECTION];
        fy = 1.0*bx_11.size()[1]/bx_11.size()[DIRECTION];
        fz = 1.0*bx_11.size()[2]/bx_11.size()[DIRECTION];
        amrex::RealBox rb_11({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
        geom_11.define(bx_11, &rb_11, 0, is_periodic.data());

        // Define computational domain and index space
        ba_11.define(geom_11.Domain());
        ba_11.maxSize(BOX_SIZE);

        dm_11.define(ba_11);
        mf_phase_11.define(ba_11,dm_11,1,2);

        // Threshold image data
        reader.threshold(1,mf_phase_11);

        // We have used a fab with one ghost cell to allow for stencil-type operations
        // over the fab. This requires to distribute the ghost cells
        //mf_phase.FillBoundary(geom.periodicity());
        mf_phase_11.FillBoundary();  
      
        amrex::Print() << std::endl << " Box 1 Size " << i
                   << std::endl << " Coordinates: (" << x_seed_low 
                   << " , " << y_seed_low
                   << " , " << z_seed_low
                   << ")  (" << x_seed_high
                   << " , " << y_seed_high
                   << " , " << z_seed_high << " ) " << std::endl;    
    
        VolumeFraction vf_11(mf_phase_11, 1);

        // Print volume fraction value
        amrex::Print() << std::endl << " Volume Fraction: "
                        << amrex::Real(vf_11.value()) << std::endl;

        amrex::Print() << std::endl << " Direction: X" << std::endl;

        // Compute tortuosity in x direction
        TortuosityHypre tortuosityx_11(geom_11,ba_11,dm_11,mf_phase_11,vf_11.value(),1,Direction::X,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);

        amrex::Real tau_value_x_11 = tortuosityx_11.value();

        amrex::Print() << std::endl << " Direction: Y" << std::endl;

        // Compute tortuosity in y direction
        TortuosityHypre tortuosityy_11(geom_11,ba_11,dm_11,mf_phase_11,vf_11.value(),1,Direction::Y,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);

        amrex::Real tau_value_y_11 = tortuosityy_11.value();

        amrex::Print() << std::endl << " Direction: Z" << std::endl;

        // Compute tortuosity in z direction
        TortuosityHypre tortuosityz_11(geom_11,ba_11,dm_11,mf_phase_11,vf_11.value(),1,Direction::Z,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);

        amrex::Real tau_value_z_11 = tortuosityz_11.value(); 

        i++;  
    }
                                                             
}
    else
    {
    amrex::Print() << std::endl << "File format not recognised." << std::endl; 
    
} 
 
  }

  amrex::Print() << std::endl << " Total Volume Computation " 
                   << std::endl;      
    
  VolumeFraction vf(mf_phase, 1);

  // Print volume fraction value
  amrex::Print() << std::endl << " Volume Fraction: "
                  << amrex::Real(vf.value()) << std::endl;

  amrex::Print() << std::endl << " Direction: X" << std::endl;

  // Compute tortuosity in x direction
  TortuosityHypre tortuosityx(geom,ba,dm,mf_phase,vf.value(),1,Direction::X,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
    
  
  amrex::Real tau_value_x = tortuosityx.value();  
    
  amrex::Print() << std::endl << " Direction: Y" << std::endl;

  // Compute tortuosity in y direction
  TortuosityHypre tortuosityy(geom,ba,dm,mf_phase,vf.value(),1,Direction::Y,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
    
  
  amrex::Real tau_value_y = tortuosityy.value();    
    
  amrex::Print() << std::endl << " Direction: Z" << std::endl;

  // Compute tortuosity in z direction
  TortuosityHypre tortuosityz(geom,ba,dm,mf_phase,vf.value(),1,Direction::Z,TortuosityHypre::SolverType::GMRES,RESULTS_PATH);
    
  amrex::Real tau_value_z = tortuosityz.value();   


  // Call the timer again and compute the maximum difference between the start time and stop time
  //   over all processors
  amrex::Real stop_time = amrex::second() - strt_time;
  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
  amrex::ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

  // Tell the I/O Processor to write out the "run time"
  amrex::Print() << std::endl << "Total run time (seconds) = " << stop_time << std::endl;

  } // Ensure amrex related destructors have been called before tearing down the whole thing
    // by putting everything in curly brackets.
  amrex::Finalize();
}
