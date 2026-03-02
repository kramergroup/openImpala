#ifndef Tortuosity_H
#define Tortuosity_H

/** Direction definition */
enum Direction { X=0, Y=1, Z=2 };

/** Cell types 
 *
 * The cell types are used to identify cells near boundaries. The kernel will implement 
 * internal boundary conditions accordinly.
 *  
 */
enum CellType { 
  BLOCKED       = 0x0,
  FREE          = 0x1,
  BOUNDARY_X_LO = 0x2,
  BOUNDARY_X_HI = 0x3,
  BOUNDARY_Y_LO = 0x4,
  BOUNDARY_Y_HI = 0x5,
  BOUNDARY_Z_LO = 0x6,
  BOUNDARY_Z_HI = 0x7
};
 
/** Computes the tortuosity of a porous structure
 * 
 * This class solved the Laplace equation on a porous structure and extracts
 * the tortuosity factor. 
 * 
 */
class Tortuosity 
{
  public:

    /** Return the tortuosity value
     * 
     * This function returns the tortuosity value. This is a potentially costly computation. The solver buffers the
     * result for repeated calls. To force recalculation use refresh=false.
     * 
     * @param refresh if true, tortuosity is re-calculated. Otherwise, a buffered value from an earlier invocation is returned.
     * 
     */
    virtual amrex::Real value(const bool refresh = false) = 0;

};

#endif