#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <boost/python/tuple.hpp>

#include <AMReX.H>
#include <AMReX_iMultiFab.H>

#include "../io/TiffReader.H"
#include "../props/VolumeFraction.H"
#include "../filter/Threshold.H"

/*****************************************************************************
 * Data Structures
 *****************************************************************************/

/**
 * VoxelSet is a container for all AMReX objects that make up a computation.
 * 
 * These are:
 *  - a MultiFab containing index-space data segmented in boxes and distributed
 *    across MPI processes
 *  
 * The MultiFab (and its supporting data structures) are held as a shared pointer
 * so we can copy the container to pass back and forth between python and c++.
 * 
 * The class is essentially a wrapper around the shared pointer with some 
 * additional convinience methods to access information from the held MultiFab.
 *  
 */
template<typename T>
struct VoxelSet
{
  using value_t = typename T::value_type;

  /**
   * Shared pointer holding the MultiFab 
   */
  const boost::shared_ptr<T> spt_multifab;

  /**
   * Constructs a new VoxelSet from the defining data structures of a MultiFab
   * 
   * @param ba the BoxArray defining rectangular areas in index-space
   * @param dm the DistributionMapping assigning each Box to an MPI rank
   * @param ncomp the number of components in the MultiFab
   * @param ngrow the number of ghost-cells in the MultiFab 
   */
  VoxelSet(const amrex::BoxArray ba, const amrex::DistributionMapping dm, const int ncomp = 1, const int ngrow = 0) : spt_multifab(new T(ba,dm,ncomp,ngrow)) {};
  
  /**
   * Constructs a new VoxelSet from a shared pointer to a MultiFab
   * 
   * @param spt_mf shared pointer to the wrapped MultiFab
   */
  VoxelSet(const boost::shared_ptr<T> spt_mf) : spt_multifab(spt_mf) {};

  /**
   * Copy Constructor
   * 
   * @param src source VoxelSet to copy. Note that this only copies the shared pointer. The underlying
   * data structure is not copied but referenced. 
   */
  VoxelSet(const VoxelSet<T>& src) : spt_multifab(src.spt_multifab) {};

  VoxelSet() {}; // Not sure why we need this, VoxelSets are not instantiable from python

  /**
   * @returns the number of boxes in the VoxelSet
  */
  int size() { return spt_multifab->size(); }

  /**
   * Sums all values in the VoxelSet
   * 
   * @param ncomp index of the component to sum (defaults to zero)
   * 
   * @return sum of values 
   */
  value_t sum(int ncomp = 0) { return spt_multifab->sum(ncomp); }
  
};

typedef VoxelSet<amrex::iMultiFab> IntVoxelSet;
typedef VoxelSet<amrex::MultiFab>  RealVoxelSet;


/*****************************************************************************
 * Initialization routines
 *****************************************************************************/

/**
 * Initialise the underlying AmReX library. This function has
 * to be called before any other functions in the library. 
 */
amrex::AMReX* initialize()
{
   amrex::AMReX* amrex = amrex::Initialize(MPI_COMM_WORLD);
   return amrex;
}

/**
 * Finalize by cleaning up used AmReX resources. This function has
 * to be called at the end of any program. No further calls to the 
 * library should follow. 
 */
void finalize(amrex::AMReX* amrex = nullptr)
{
  if ( amrex == nullptr )
  {
    amrex::Finalize();
  } else {
    amrex::Finalize(amrex);
  } 
}

BOOST_PYTHON_FUNCTION_OVERLOADS(finalize_overloads, finalize, 0, 1)

/*****************************************************************************
 * IO Routines
 *****************************************************************************/

/**
 * Print in correct order within MPI environments 
 */
template<typename T>
void print(const T s) { amrex::Print() << s << std::endl; }

IntVoxelSet read_tiff_stack(const std::string filename, const size_t box_size = 32)
{
  TiffReader reader(filename);

  int ncomp = 1;
  int ngrow = 0;

  amrex::Geometry geom;
  {
    amrex::RealBox rb({-1.0,-1.0,-1.0}, {1.0,1.0,1.0}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    geom.define(reader.box());
  }

  amrex::BoxArray ba(geom.Domain());
  ba.maxSize(box_size);
  amrex::DistributionMapping dm(ba);
  
  IntVoxelSet vs(ba,dm,ncomp,ngrow);
  reader.fill(*(vs.spt_multifab));

  return vs;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(read_tiff_stack_overloads, read_tiff_stack, 1, 2)

/*****************************************************************************
 * Filter 
 *****************************************************************************/

template<typename T>
IntVoxelSet threshold(const boost::shared_ptr<VoxelSet<T>> src, const typename T::value_type value)
{
  ThresholdFilter<amrex::iMultiFab> f_threshold(value);
  boost::shared_ptr<amrex::iMultiFab> vs = boost::make_shared<amrex::iMultiFab>();
  
  f_threshold.filter(*(src->spt_multifab), *vs);
  
  return vs;
}


/*****************************************************************************
 * Properties 
 *****************************************************************************/

template<typename T>
float calculate_volume_fraction(boost::shared_ptr<VoxelSet<T>> vs)
{

}

/*****************************************************************************
 * Python Interface
 *****************************************************************************/

BOOST_PYTHON_MODULE(pyoimp)
{

  using namespace boost::python;

  /*** Data Structures ***/
  // class_<IntVoxelSet,boost::shared_ptr<IntVoxelSet>>("IntVoxelSet",no_init)
  //   .def("size",&IntVoxelSet::size, "Number of boxes containing voxels");
  
  class_<IntVoxelSet>("IntVoxelSet",no_init)
    .def("size",&IntVoxelSet::size, "Number of boxes containing voxels")
    .def("sum",&IntVoxelSet::sum, "Sum of all voxel values");
  
  class_<RealVoxelSet>("RealVoxelSet");
  register_ptr_to_python<boost::shared_ptr<RealVoxelSet>>();
  

  /*** Initialization ***/

  // Enable passing of smart pointer to amrex
  class_<amrex::AMReX>("AMReX");
  register_ptr_to_python<boost::shared_ptr<amrex::AMReX>>();

  // We return an unmanaged raw pointer from initialize, because the lifetime of amrex is managed 
  // by explicit calls to initialize/finalize. Automatically destructing/deleting amrex pointers
  // leads to segmentation faults.
  def("initialize", initialize, return_value_policy<reference_existing_object>(), "Initialize the environment");
  def("finalize", finalize, finalize_overloads());
  def("isInitialized", amrex::Initialized, "Returns true if the enviroment is initialized");
  
  /*** IO Routines ***/

  def("print", print<const bool>, "Print to standard output");
  def("print", print<const int>, "Print to standard output");
  def("print", print<const float>, "Print to standard output");
  def("print", print<const std::string>, "Print to standard output");
  
  def("read_tiff_stack", read_tiff_stack, "Import integer-valued voxel-set from a stack of tiff images");

  /*** Filter routines ***/
  def("threshold", threshold<amrex::iMultiFab>,"Threshold integer valued VoxelSets (IntVoxelSet)");

}
