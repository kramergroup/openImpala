#include "CathodeWrite.H"
#include <AMReX_REAL.H>

#include <iostream>
#include <fstream>

CathodeWrite::CathodeWrite(amrex::Real vf) : m_vf(vf)
{

}

void CathodeWrite::DandeLiionWrite()
{

  std::ofstream myfile;
  myfile.open ("cathode.txt");
  myfile << "###########################################################\n";
  myfile << "# Cathode parameter container (according to Ecker et al.) #\n";
  myfile << "###########################################################\n";
  myfile << "\n";
  myfile << "N = 55              # Number of mesh nodes in liquid\n";
  myfile << "M = 50              # Number of mesh nodes in solid\n";
  myfile << "\n";
  myfile << "L = 54.0e-6         # Electrode thickness, m\n";
  myfile << "R = 6.5e-6          # Particle radius, m\n";
  myfile << "\n";
  myfile << "act     = 0.58      # Active part of the electrode\n";
  myfile << "A       = 8.585e-3  # Electrode cross-sectional area, m^2\n";
  myfile << "el      = "<< amrex::Real(m_vf) <<"  # Volume fraction of electrolyte, Calculated using OpenImpala\n";
  myfile << "bet     = 188455    # bet = 3 * (1 - el) / R * act; BET (Brunauer-Emmett-Teller) surface area, m^-1\n";
  myfile << "B       = 0.1526    # B = el / 1.94; Permeability factor of electrolyte\n";
  myfile << "cmax    = 48580     # Maximum concentration of Li ions in solid, mol m^-3\n";
  myfile << "sigma_s = 68.1      # Solid conductivity, S m^-1\n";
  myfile << "###########################################################\n";
  myfile.close();

}

void CathodeWrite::PyBammWrite()
{

  std::ofstream myfile;
  myfile.open ("parameters.csv");
  myfile << "Name [units],Value,Reference,Notes\n";
  myfile << "# Empty rows and rows starting with ‘#’ will be ignored,,,\n";
  myfile << ",,,\n";
  myfile << "# Electrode properties,,,\n";
  myfile << "Positive electrode conductivity [S.m-1],10,Scott Moura FastDFN,lithium cobalt oxide\n";
  myfile << "Maximum concentration in positive electrode [mol.m-3],51217.9257309275,Scott Moura FastDFN,\n";
  myfile << "Positive electrode diffusivity [m2.s-1],[function]lico2_diffusivity_Dualfoil1998,,\n";
  myfile << "Positive electrode OCP [V],[function]lico2_ocp_Dualfoil1998,\n";
  myfile << ",,,\n";
  myfile << "# Microstructure,,,\n";
  myfile << "Positive electrode porosity,"<< amrex::Real(m_vf) <<",OpenImpala Calculated Parameter,electrolyte volume fraction\n";
  myfile << "Positive electrode active material volume fraction,0.7,,assuming zero binder volume fraction\n";
  myfile << "Positive particle radius [m],1E-05,Scott Moura FastDFN,\n";
  myfile << "Positive electrode surface area density [m-1],150000,Scott Moura FastDFN,\n";
  myfile << "Positive electrode Bruggeman coefficient (electrolyte),1.5,Scott Moura FastDFN,\n";
  myfile << "Positive electrode Bruggeman coefficient (electrode),1.5,Scott Moura FastDFN,\n";
  myfile << ",,,\n";
  myfile << "# Interfacial reactions,,,\n";
  myfile << "Positive electrode cation signed stoichiometry,-1,,\n";
  myfile << "Positive electrode electrons in reaction,1,,\n";
  myfile << "Positive electrode reference exchange-current density [A.m-2(m3.mol)1.5],6E-07,Scott Moura FastDFN,Be careful how we implement BV\n";
  myfile << "Reference OCP vs SHE in the positive electrode [V],,,\n";
  myfile << "Positive electrode charge transfer coefficient,0.5,Scott Moura FastDFN,\n";
  myfile << "Positive electrode double-layer capacity [F.m-2],0.2,,\n";
  myfile << ",,,\n";
  myfile << "# Density,,,\n";
  myfile << "Positive electrode density [kg.m-3],3262,,\n";
  myfile << ",,,\n";
  myfile << "# Thermal parameters,,,\n";
  myfile << "Positive electrode specific heat capacity [J.kg-1.K-1],700,,\n";
  myfile << "Positive electrode thermal conductivity [W.m-1.K-1],2.1,,\n";
  myfile << "Positive electrode OCP entropic change [V.K-1],[function]lico2_entropic_change_Moura2016,,\n";
  myfile << ",,,\n";
  myfile << "# Activation energies,,,\n";
  myfile << "Reference temperature [K],298.15,25C,\n";
  myfile << "Positive electrode reaction rate,[function]lico2_electrolyte_reaction_rate_Dualfoil1998,,\n";
  myfile << "Positive reaction rate activation energy [J.mol-1],39570,,\n";
  myfile << "Positive solid diffusion activation energy [J.mol-1],18550,,\n";
  myfile.close();

}
