#ifndef THRESHOLD_H
#define THRESHOLD_H

#include <AMReX_iMultiFab.H>
#include "Filter.H"

/**
 * ThresholdFilter applies a threshold to the provided MultiFab. 
 * 
 * @tparam T the type of the threshold parameters
 * @tparam MF the MultiFab type of the source, has to hold values that can be compared to T 
 *   
 */

template<typename MF>
class ThresholdFilter : public Filter<MF,amrex::iMultiFab> {
    
using V = typename MF::value_type;

private:
    
    /**
     * The threshold value
     */
    const V m_threshold; 

    /**
     * The component value that is used for thresholding 
     */
    const int m_nComp;

public:

    /**
     * Create a threshold filter that returns a binary index-spaced map of voxel values that are larger than a threshold
     * 
     * @param threshold the threshold value
     */
    ThresholdFilter(V threshold, int nComp = 0) : m_threshold(threshold), m_nComp(nComp) {};

    void filter(const MF& source, amrex::iMultiFab& target);

};

#endif