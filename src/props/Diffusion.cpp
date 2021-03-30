/** Diffusion calculation
 *
 * This programme calculates diffusion and tortuosity in the 3 cartesian directions,
 * as well volume fraction for a 2 phase segmented dataset.
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
#include "TortuosityHypre.H"
#include "VolumeFraction.H"

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
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
  amrex::Print() << AMREX_SPACEDIM << "D test" << std::endl;

  std::string FILENAME;
  pp.get("FILENAME", FILENAME);  // query filename from inputs file
    
  std::string DATA_PATH = "/openImpala/data/";
  pp.get("DATA_PATH", DATA_PATH);  // query data directory from inputs file
  if(DATA_PATH.at(0) == '~')
  {
      // Get the users home directory to write plot file to right place
      const char* homeDir = getenv("HOME");
      DATA_PATH.erase(DATA_PATH[0]); 
      DATA_PATH = homeDir + DATA_PATH;
     amrex::Print() << " DATA_PATH: " << DATA_PATH << "homeDir" << homeDir << std::endl;
  }
    

  amrex::Real DIRECTION = Direction::X;
  pp.query("DIRECTION", DIRECTION);  // query direction from inputs file

  // BOX_SIZE can have a considerable influence on the calculation speed of the programme
  // further work is required to work out the optimum value, if not, leave the value as 32
  amrex::Real BOX_SIZE = 32;
  pp.query("BOX_SIZE", BOX_SIZE);

  amrex::Real EPS= 1e-10;
  pp.query("EPS", EPS);

  // Define physical geometry, index space, and multifab
  amrex::Geometry geom;
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  amrex::iMultiFab mf_phase;

  {
    // Reading the tiff file
    // The TiffReader potentially holds significant data in memory (the full voxel set).
    // The code is not parallelised, potentially creating a large memory burden per node.
    // It's best to let the reader go out of scope as soon as it is not needed anymore
    // to free up memory before further computations.
    amrex::Print() << "tTiffReader - Reading file " << DATA_PATH + FILENAME << std::endl;
    TiffReader reader(DATA_PATH + FILENAME);

    const amrex::Box bx = reader.box();
    amrex::Real fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    amrex::Real fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    amrex::Real fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
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

  VolumeFraction vf(mf_phase, 0);

  // Print volume fraction value
  amrex::Print() << std::endl << " Volume Fraction: "
                  << amrex::Real(vf.value()) << std::endl;

  if (DIRECTION==0)
  {
  amrex::Print() << std::endl << " Direction: X" << std::endl;

  // Compute tortuosity in x direction
  TortuosityHypre tortuosityx(geom,ba,dm,mf_phase,vf.value(),0,Direction::X,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_x = tortuosityx.value();
  amrex::Print() << " Tortuosity value: " << tau_value_x << std::endl;
}

  else if (DIRECTION==1)
  {
  amrex::Print() << std::endl << " Direction: Y" << std::endl;

  // Compute tortuosity in y direction
  TortuosityHypre tortuosityy(geom,ba,dm,mf_phase,vf.value(),0,Direction::Y,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_y = tortuosityy.value();
  amrex::Print() << " Tortuosity value: " << tau_value_y << std::endl;
}

  else if (DIRECTION==2)
  {
  amrex::Print() << std::endl << " Direction: Z" << std::endl;

  // Compute tortuosity in z direction
  TortuosityHypre tortuosityz(geom,ba,dm,mf_phase,vf.value(),0,Direction::Z,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_z = tortuosityz.value();
  amrex::Print() << " Tortuosity value: " << tau_value_z << std::endl;
}

  else
  {
  amrex::Print() << std::endl << " Direction: X" << std::endl;

  // Compute tortuosity in x direction
  TortuosityHypre tortuosityx(geom,ba,dm,mf_phase,vf.value(),0,Direction::X,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_x = tortuosityx.value();
  amrex::Print() << " Tortuosity value: " << tau_value_x << std::endl;

  amrex::Print() << std::endl << " Direction: Y" << std::endl;

  // Compute tortuosity in y direction
  TortuosityHypre tortuosityy(geom,ba,dm,mf_phase,vf.value(),0,Direction::Y,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_y = tortuosityy.value();
  amrex::Print() << " Tortuosity value: " << tau_value_y << std::endl;

  amrex::Print() << std::endl << " Direction: Z" << std::endl;

  // Compute tortuosity in z direction
  TortuosityHypre tortuosityz(geom,ba,dm,mf_phase,vf.value(),0,Direction::Z,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value_z = tortuosityz.value();
  amrex::Print() << " Tortuosity value: " << tau_value_z << std::endl;
}

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
