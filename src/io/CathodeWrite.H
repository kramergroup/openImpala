#ifndef CATHODE_WRITE_H // Corrected and styled header guard
#define CATHODE_WRITE_H

#include <AMReX_REAL.H>
#include <string>
#include <vector> // Often useful for parameters like stoichiometric ranges, etc.
// #include <map> // May be needed depending on parameter complexity

namespace OpenImpala { // Wrap in project namespace

/**
 * @brief Structure to hold cathode material and electrode parameters.
 *
 * @warning This is an EXAMPLE structure. Populate it with ALL parameters
 * required by the target file formats (DandeLiion, PyBamm),
 * including material properties, electrode design parameters, etc.
 *
 * @warning Document the EXPECTED UNITS for every parameter (e.g., SI units).
 */
struct CathodeParams
{
    // Example parameters - ADD ALL NECESSARY PARAMETERS HERE
    amrex::Real volume_fraction_solid = 0.5; // [-] Volume fraction of active material
    amrex::Real particle_radius = 5e-6;      // [m] Average particle radius
    amrex::Real active_material_conductivity = 100.0; // [S/m] Solid phase conductivity
    amrex::Real max_concentration = 51000.0; // [mol/m^3] Max concentration in solid
    // Add diffusivity, reaction rates, porosity (1-vf), Bruggeman coeffs, etc.
    // Add thermal parameters if needed.
    // Add stoichiometric coefficients/ranges if needed.

    // Constructor (optional, can use aggregate initialization)
    CathodeParams() = default;
};


/**
 * @brief A writer for generating cathode parameter files for battery models.
 *
 * This class takes cathode parameters (via CathodeParams struct or potentially
 * ParmParse in the future) and writes them into formatted files compatible
 * with specific battery simulation platforms like DandeLiion and PyBamm.
 */
class CathodeWrite
{
public:
    /**
     * @brief Constructs the CathodeWrite object using provided parameters.
     *
     * Stores the necessary parameters internally for later use by the write methods.
     *
     * @param params A struct containing all required cathode parameters.
     * Ensure all values and units are correctly specified.
     */
    explicit CathodeWrite(const CathodeParams& params); // Use explicit

    /**
     * @brief Virtual default destructor. Important if inheriting.
     */
    virtual ~CathodeWrite() = default;

    // --- Deleted Copy Operations ---
    // Prevent copying; writer is configured at construction.
    CathodeWrite(const CathodeWrite&) = delete;
    CathodeWrite& operator=(const CathodeWrite&) = delete;

    /**
     * @brief Writes a parameter file compatible with the DandeLiion model.
     *
     * @param filename The full path and name for the output .txt parameter file.
     * @return true if the file was written successfully, false otherwise (e.g., file open error).
     */
    bool writeDandeLiionParameters(const std::string& filename) const; // Mark const if state doesn't change

    /**
     * @brief Writes a parameter file compatible with the PyBamm model.
     *
     * @param filename The full path and name for the output .csv parameter file.
     * @return true if the file was written successfully, false otherwise (e.g., file open error).
     */
    bool writePyBammParameters(const std::string& filename) const; // Mark const if state doesn't change

private:
    // --- Member Variables ---
    // Store copies of parameters needed for writing files.
    // Mark 'const' if they are immutable after construction.
    // Example parameters (match CathodeParams):
    const amrex::Real m_vf_solid;
    const amrex::Real m_particle_radius;
    const amrex::Real m_conductivity;
    const amrex::Real m_max_concentration;
    // Add other stored parameters corresponding to CathodeParams

    // Internal helper methods could go here if needed
};

} // namespace OpenImpala

#endif // CATHODE_WRITE_H
