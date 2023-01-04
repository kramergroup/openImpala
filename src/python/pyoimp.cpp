#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <AMReX.H>


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

bool isInitialized()
{
  return amrex::Initialized();
}

BOOST_PYTHON_FUNCTION_OVERLOADS(finalize_overloads,finalize,0,1)

BOOST_PYTHON_MODULE(pyoimp)
{
  using namespace boost::python;

  // Enable passing of smart pointer to amrex
  class_<amrex::AMReX>("AMReX");
  
  // We return an unmanaged raw pointer, because the lifetime of amrex is managed 
  // by explicit calls to initialize/finalize
  def("initialize", initialize, return_value_policy<reference_existing_object>(), "Initialize the environment");
  def("finalize", finalize, "Clean-up environment");
  def("isInitialized", isInitialized, "Returns true if the enviroment is initialized");
}
