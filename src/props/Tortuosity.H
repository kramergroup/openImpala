#ifndef TORTUOSITY_H // Standard include guard
#define TORTUOSITY_H

#include <AMReX_REAL.H> // Include necessary dependency for amrex::Real

namespace OpenImpala { // Wrap definitions in a project-specific namespace

/** @brief Defines cardinal directions for calculations. */
enum class Direction {
    X = 0,
    Y = 1,
    Z = 2
};

/**
 * @brief Cell types identifying the role of a cell within the domain.
 *
 * Used by solver kernels to identify cells near boundaries or blocked regions,
 * potentially aiding in applying specific boundary conditions or skipping calculations.
 */
enum class CellType {
    /** @brief Cell is fully blocked (e.g., solid phase, no transport). */
    BLOCKED = 0, // Use regular integer literals
    /** @brief Cell is fully open (e.g., conductive phase, allows transport). */
    FREE = 1,
    /** @brief Cell is on the lower boundary in the X-direction. */
    BOUNDARY_X_LO = 2,
    /** @brief Cell is on the higher boundary in the X-direction. */
    BOUNDARY_X_HI = 3,
    /** @brief Cell is on the lower boundary in the Y-direction. */
    BOUNDARY_Y_LO = 4,
    /** @brief Cell is on the higher boundary in the Y-direction. */
    BOUNDARY_Y_HI = 5,
    /** @brief Cell is on the lower boundary in the Z-direction. */
    BOUNDARY_Z_LO = 6,
    /** @brief Cell is on the higher boundary in the Z-direction. */
    BOUNDARY_Z_HI = 7
};

/**
 * @brief Abstract base class for computing the tortuosity of a structure.
 *
 * This class defines the interface for solvers that compute tortuosity,
 * often by solving a transport equation (like Laplace's equation for diffusion)
 * on a representation of the structure.
 */
class Tortuosity
{
public:
    /**
     * @brief Virtual destructor (essential for base classes with virtual functions).
     * Ensures proper cleanup when derived classes are deleted via a base class pointer.
     */
    virtual ~Tortuosity() = default;

    /**
     * @brief Calculate and return the tortuosity value.
     *
     * This function computes or retrieves the tortuosity for the structure,
     * typically defined in relation to effective transport properties.
     * Implementations may cache the result due to potentially high computation cost.
     *
     * @param refresh If true, forces a recalculation of the tortuosity.
     * If false (default), a cached value may be returned if available.
     * @return The calculated tortuosity value (often defined as VF / (Deff/D0)).
     * Return value characteristics (e.g., range, behavior on error) depend
     * on the derived class implementation.
     */
    virtual amrex::Real value(const bool refresh = false) = 0; // Pure virtual function

protected:
    // Protected constructor/destructor if needed, but often not for pure abstract base
    Tortuosity() = default; // Can allow derived classes to default construct

private:
    // Prevent copying/assignment if derived classes manage resources non-copyably
    // (Place these in the public section if you prefer, style varies)
    // Tortuosity(const Tortuosity&) = delete;             // C++11 delete
    // Tortuosity& operator=(const Tortuosity&) = delete; // C++11 delete

}; // class Tortuosity

} // namespace OpenImpala

#endif // TORTUOSITY_H
