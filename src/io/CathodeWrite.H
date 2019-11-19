#ifndef _CATHODEWRITE_H
#define _CATHODEWRITE_H

#include <AMReX_REAL.H>

/** A Writer of Cathode Parameter Files
 *
 * This class provides a writer of cathode parameter files. This class can be used to
 * create files to interface with popular opensource DFN models.
 *
 */

class CathodeWrite
{
  public:

    /** Constructs a new CathodeWrite property
     *
     * This constructor creates a CathodeWrite property.
     *
     * @param vf amrex::Real containing the volume fraction
     */
    CathodeWrite(const amrex::Real vf = 0.0);

    /** Create DandeLiion Parameter File
     *
     * Creates a .txt parameter file that can be used directly in DandeLiion
     *
     */
    void DandeLiionWrite();

    /** Create PyBamm Parameter File
     *
     * Creates a .csv parameter file that can be used directly in PyBamm
     *
     */
    void PyBammWrite();

  private:

    const amrex::Real m_vf;


};

#endif // CATHODEWRITE_H
