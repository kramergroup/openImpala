# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Improved version incorporating fixes for make errors and portability.

# ============================================================
# Environment Setup (MUST BE SET OR ADJUSTED)
# ============================================================
# Set these variables in your environment or modify defaults here if paths differ
AMREX_HOME    ?= /path/to/amrex      # Base AMReX install directory
HYPRE_HOME    ?= /path/to/hypre      # Base HYPRE install directory
HDF5_HOME     ?= /usr                # Base HDF5 install directory (often /usr)
# Assuming official HDF5 C++ bindings are installed with HDF5 C library
H5CPP_HOME    ?= $(HDF5_HOME)        # Base H5CPP install directory
TIFF_HOME     ?= /usr                # Base LibTIFF install directory (often /usr)
# BOOST_HOME    ?= /usr                # Base Boost install directory (if needed & non-std)

# ============================================================
# Compilers and Flags
# ============================================================
CXX           := mpic++
F90           := mpif90

# Base Flags + Auto-Dependency Generation for C++ (-MMD -MP) + C++17
# Add -DOMPI_SKIP_MPICXX if needed by your MPI implementation
CXXFLAGS      := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS      := -g -O3

INCLUDE       := -I$(AMREX_HOME)/include\
-I$(HYPRE_HOME)/include\
-I$(HDF5_HOME)/include\
-I$(H5CPP_HOME)/include\
-I$(TIFF_HOME)/include
#                -I$(BOOST_HOME)/include  # If Boost headers are needed and non-standard

# Linker Flags (using variables for portability)
# Rely on MPI wrappers to link MPI libs and potentially Fortran runtime
# List required application libraries
LDFLAGS       := -L$(AMREX_HOME)/lib -lamrex\
-L$(HYPRE_HOME)/lib -lHYPRE\
-L$(HDF5_HOME)/lib -lhdf5 -lhdf5_cpp\
-L$(TIFF_HOME)/lib -ltiff\
-lm

# ============================================================
# Project Structure
# ============================================================
INC_DIR       := build/include
APP_DIR       := build/apps
TST_DIR       := build/tests
IO_DIR        := build/io
PROPS_DIR     := build/props

MODULES       := io props
SRC_DIRS      := $(addprefix src/,$(MODULES)) # src/io src/props
BUILD_DIRS    := $(addprefix build/,$(MODULES)) # build/io build/props

# ============================================================
# Source and Object Files
# ============================================================
# Find source files
SOURCES_IO_CPP  := $(wildcard src/io/*.cpp)
SOURCES_PRP_CPP := $(wildcard src/props/*.cpp)
SOURCES_PRP_F90 := $(wildcard src/props/*.F90)

# Define object file targets based on source locations
OBJECTS_IO_CPP  := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_CPP))
OBJECTS_PRP_CPP := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_CPP))
OBJECTS_PRP_F90 := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(SOURCES_PRP_F90))

OBJECTS_CPP := $(OBJECTS_IO_CPP) $(OBJECTS_PRP_CPP)
OBJECTS_F90 := $(OBJECTS_PRP_F90)
OBJECTS       := $(OBJECTS_CPP) $(OBJECTS_F90)

# Let Make search for source files in relevant directories
VPATH := $(subst $(space),:,$(SRC_DIRS))

# ============================================================
# Main Targets
# ============================================================
.PHONY: all main tests builddirs clean debug release

all: main tests

main: builddirs $(APP_DIR)/Diffusion

tests: builddirs $(TST_DIR)/tTiffReader $(TST_DIR)/tDatReader $(TST_DIR)/tVolumeFraction $(TST_DIR)/tTortuosity $(TST_DIR)/tHDF5Reader # Removed tTiffStackReader

# ============================================================
# Compilation Rules (Using Standard Pattern Rules)
# ============================================================
# Order-only prerequisites (| $(DIR)) ensure directory exists before compiling
# Added @mkdir -p $(@D) in recipes for extra safety

# Rule for C++ objects in build/io
$(IO_DIR)/%.o : %.cpp | $(IO_DIR)
  @echo "Compiling (IO) $< ..."
  @mkdir -p $(@D)
  $(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Rule for C++ objects in build/props
$(PROPS_DIR)/%.o : %.cpp | $(PROPS_DIR)
  @echo "Compiling (Props) $< ..."
  @mkdir -p $(@D)
  $(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Rule for F90 objects in build/props
$(PROPS_DIR)/%.o : %.F90 | $(PROPS_DIR) $(INC_DIR)
  @echo "Compiling (Props Fortran) $< ..."
  @mkdir -p $(@D)
  $(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $< -o $@

# ============================================================
# Linking Executables
# ============================================================

# Define object lists for clarity
APP_OBJS       := $(OBJECTS)

# Specific objects needed for each test (adjust based on actual dependencies)
TEST_OBJS_TIFF    := $(IO_DIR)/TiffReader.o
TEST_OBJS_DAT     := $(IO_DIR)/DatReader.o
TEST_OBJS_HDF5    := $(IO_DIR)/HDF5Reader.o
TEST_OBJS_VF      := $(PROPS_DIR)/VolumeFraction.o $(IO_DIR)/TiffReader.o
# Tortuosity depends on Hypre backend, VolumeFraction, Readers (Tiff?), Fortran kernels
TEST_OBJS_TORT    := $(PROPS_DIR)/TortuosityHypre.o $(PROPS_DIR)/VolumeFraction.o \
$(OBJECTS_PRP_F90) \
$(IO_DIR)/TiffReader.o $(IO_DIR)/DatReader.o $(IO_DIR)/HDF5Reader.o
# Note: Included all readers here, adjust if tTortuosity only uses one type


# Main application
$(APP_DIR)/Diffusion: Diffusion.cpp $(APP_OBJS) | $(APP_DIR)
@echo "Linking $@ ..."
# @mkdir -p $(@D) # Directory created by order-only dependency
$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use CXX for linking C++/Fortran with MPI wrapper

# Test executables
$(TST_DIR)/tTiffReader: tTiffReader.cpp $(TEST_OBJS_TIFF) | $(TST_DIR)
@echo "Linking $@ ..."
$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tDatReader: tDatReader.cpp $(TEST_OBJS_DAT) | $(TST_DIR)
@echo "Linking $@ ..."
$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tHDF5Reader: tHDF5Reader.cpp $(TEST_OBJS_HDF5) | $(TST_DIR)
@echo "Linking $@ ..."
$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tVolumeFraction: tVolumeFraction.cpp $(TEST_OBJS_VF) | $(TST_DIR)
@echo "Linking $@ ..."
$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tTortuosity: tTortuosity.cpp $(TEST_OBJS_TORT) | $(TST_DIR)
@echo "Linking $@ ..."
$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

# ============================================================
# Supporting Targets
# ============================================================

# Target to create all necessary build directories
builddirs: $(BUILD_DIRS) $(APP_DIR) $(TST_DIR) $(INC_DIR)

# General rule for creating build directories
$(BUILD_DIRS) $(APP_DIR) $(TST_DIR) $(INC_DIR):
@echo "Creating directory $@"
@mkdir -p $@

# Build environments (simple examples)
debug: CXXFLAGS += -DDEBUG
debug: F90FLAGS += -fbounds-check # Example Fortran debug flag
debug: all

release: # Using defaults: -O3 defined initially
release: all

# Clean target (removes objects, module files, executables, dependency files)
clean:
@echo "Cleaning build artifacts..."
-@rm -rf $(IO_DIR) $(PROPS_DIR) $(INC_DIR) $(APP_DIR) $(TST_DIR) build # Remove specific dirs and top build dir
# Clean dependency files potentially generated in source dirs if CXXFLAGS didn't handle output dir well
-@find src -name '*.d' -delete
-@find . -name '*.mod' -delete # Remove Fortran module files from source tree if accidentally created there

# ============================================================
# Debugging Output (Temporary)
# ============================================================
# $(info --- Make Debug Variables ---)
# $(info IO_DIR       = [${IO_DIR}])
# $(info PROPS_DIR    = [${PROPS_DIR}])
# $(info OBJECTS_IO_CPP = [${OBJECTS_IO_CPP}])
# $(info OBJECTS_PRP_CPP= [${OBJECTS_PRP_CPP}])
# $(info OBJECTS_CPP = [${OBJECTS_CPP}])
# $(info DEPS_FILES   = [${OBJECTS_CPP:.o=.d}])
# $(info --- End Make Debug ---)


# ============================================================
# Include Auto-Generated Dependencies
# ============================================================
# Include the .d files generated by CXXFLAGS -MMD -MP
# Put this near the end so Make knows all explicit rules first
-include $(OBJECTS_CPP:.o=.d)

# ============================================================
# Debugging Help (Uncomment to use)
# ============================================================
# print-%:
# 	@echo '$* = $($*)'
# Example: make print-OBJECTS_CPP
