#ifndef FILTER_H
#define FILTER_H

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

/**
 Filter is the abstact base class of filters. A filter takes one or more inputs and
 generates one or more outputs.
*/

template <class S, class T>
class Filter {

public:

    virtual T filter(const S& source) = 0;
    
};

typedef Filter<amrex::MultiFab,amrex::MultiFab> MultiFabFilter;
typedef Filter<amrex::iMultiFab,amrex::iMultiFab> iMultiFabFilter;

#endif