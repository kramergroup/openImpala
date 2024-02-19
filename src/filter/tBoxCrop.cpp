/** Test VolumeFractions class
 *
 * This test tests the VolumeFraction class. It requires the SAMPLE_TIFF_FILENAME file
 * as test data.
 *
 * The test will open the sample tiff file, read its data, and then asset the
 * following conditions:
 *
 *  1) volume fractions of the two phases
 *
 */

#include "../io/TiffReader.H"
#include "BoxCrop.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include <sstream>

#define SAMPLE_TIFF_FILENAME "/openImpala/data/SampleData_2Phase.tif"
#define SAMPLE_HDF5_FILENAME "/openImpala/data/CropData_2Phase"
#define BOX_SIZE 32
#define EPS 1e-10

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv, false);
  {
  // Parameters
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
  amrex::Print() << AMREX_SPACEDIM << "D test" << std::endl;

  // Define physical geometry, index space, and multifab
  amrex::Geometry geom;
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  amrex::iMultiFab mf_phase;
  amrex::Box box;

  {
    // Reading the tiff file
    // The TiffReader potentially holds significant data in memory (the full voxel set).
    // The code is not parallelised, potentially creating a large memory burden per node.
    // It's best to let the reader go out of scope as soon as it is not needed anymore
    // to free up memory before further computations.
    std::string tiffFileName = (argc > 1) ? argv[1] : SAMPLE_TIFF_FILENAME;
    amrex::Print() << "tTiffReader - Reading file " << tiffFileName << std::endl;
    TiffReader reader(tiffFileName);

    amrex::RealBox rb({0.0,0.0,0.0}, {1.0,1.0,1.0}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
    geom.define(reader.box(), &rb, 0, is_periodic.data());
    box = reader.box();

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);

    dm.define(ba);
    mf_phase.define(ba,dm,1,0);

    // Threshold image data
    reader.threshold(1,mf_phase);
  }

  // Compute BoxCrop
  // We crop out the middle
  amrex::IntVect se = (box.bigEnd() - box.smallEnd()) / 4 + box.smallEnd();
  amrex::IntVect be = (box.bigEnd() - box.smallEnd()) * 3 / 4 + box.smallEnd();
  amrex::Box cropBox(se,be);

  // Create a box array for the receiving multifab
  amrex::BoxArray t_ba(cropBox);
  t_ba.maxSize(BOX_SIZE);

  // Create a distribution mapping for the target BoxArray
  amrex::DistributionMapping t_dm(t_ba);

  // Create the filter
  BoxCropFilter<amrex::iMultiFab> crop(t_ba, t_dm);
  
  // Filter the phase
  amrex::iMultiFab mf_target;
  crop.filter(mf_phase, mf_target);

  // Plot the result
  // - First copy the data into a amrex::MultiFab, because the plotting routines cannot deal with amrex:iMultiFab
  // - To make that easier, we use the same box array and distribution mapping for both MultiFabs
  {
    amrex::MultiFab mf_plot;
    mf_plot.define(t_ba,t_dm,1,0);

    for (amrex::MFIter mfi(t_ba,t_dm); mfi.isValid(); ++mfi) {
      const amrex::Box& bx = mfi.validbox();
      const amrex::Array4<amrex::Real>& mf_plt_array = mf_plot.array(mfi);
      const amrex::Array4<int>& mf_trg_array = mf_target.array(mfi);

      amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k){
        mf_plt_array(i,j,k) = mf_trg_array(i,j,k);
      });
    }
    
    std::string hdf5FileName = (argc > 2) ? argv[2] : SAMPLE_HDF5_FILENAME;
    amrex::Print() << "writing crop result - HDF5 file " << hdf5FileName << std::endl;
#ifdef AMREX_USE_HDF5
    amrex::WriteSingleLevelPlotfileHDF5(hdf5FileName, mf_plot, {"phase"},geom, 0.0, 0);
#else
    amrex::WriteSingleLevelPlotfile(hdf5FileName, mf_plot, {"phase"},geom, 0.0, 0);
#endif
  }
 


  } // Ensure amrex related destructors have been called before tearing down the whole thing
    // by putting everything in curly brackets.
  amrex::Finalize();

}
