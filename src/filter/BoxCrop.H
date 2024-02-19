#ifndef BOXCROP_H
#define BOXCROP_H

#include <AMReX_iMultiFab.H>
#include "Filter.H"

template<typename MultiFab>
class BoxCropFilter : public Filter<MultiFab,MultiFab> {

private:

    /**
     * The receiving BoxArray. This has the dimensions of the crop-out in 
     * index space.
     */
    const amrex::BoxArray& m_cropBox;

    /**
     * A reference to the distribution mapping of the receiving BoxArray.
     * 
     */
    const amrex::DistributionMapping& m_dm;

    /**
     * The number of value arrays to copy
     */
    const int m_nvar; 

public:

    /*
    Create a BoxCropFilter that crops a rectangular domain `cropBox` 
    from the source MultiFab and returns a new MultiFab of the sub-domain.
    Data is copied in the process.
    */
    BoxCropFilter(const amrex::BoxArray& cropBox, const amrex::DistributionMapping& dm) : m_cropBox(cropBox), m_dm(dm), m_nvar(1) { };
    BoxCropFilter(const amrex::BoxArray& cropBox, int nvar) : m_cropBox(cropBox), m_nvar(nvar) { };

    void filter(const MultiFab& source, MultiFab& target);

};

#endif