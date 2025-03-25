# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Oxford, UK - Tue Mar 25 2025 10:29 PM GMT

# ============================================================
# Environment Setup (MUST BE SET OR ADJUSTED)
# ============================================================
# Set these variables in your environment or modify defaults here
AMREX_HOME     ?= /path/to/amrex       # Base AMReX install directory
HYPRE_HOME     ?= /path/to/hypre       # Base HYPRE install directory
HDF5_HOME      ?= /usr                # Base HDF5 install directory (often /usr)
H5CPP_HOME     ?= /usr/local          # Base H5CPP install directory (if not standard)
TIFF_HOME      ?= /usr                # Base LibTIFF install directory (often /usr)
# BOOST_HOME     ?= /usr              # Base Boost install directory (if needed & non-std)

# ============================================================
# Compilers and Flags
# ============================================================
CXX         := mpic++
F90         := mpif90

# Base Flags (add -MMD -MP for auto C++ dependencies, use C++17)
# Add -DOMPI_SKIP_MPICXX if needed by your MPI implementation
CXXFLAGS    := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS    := -g -O3

# Include Paths (using variables for portability)
INCLUDE     := -I$(AMREX_HOME)/include \
               -I$(HYPRE_HOME)/include \
               -I$(HDF5_HOME)/include \
               -I$(H5CPP_HOME)/include \
               -I$(TIFF_HOME)/include
            #  -I$(BOOST_HOME)/include  # If Boost headers are needed and non-standard

# Linker Flags (using variables for portability)
# Rely on MPI wrappers to link MPI libs and potentially Fortran runtime
# List required application libraries
LDFLAGS     := -L$(AMREX_HOME)/lib -lamrex \
               -L$(HYPRE_HOME)/lib -lHYPRE \
               -L$(HDF5_HOME)/lib -lhdf5 \
               -L$(H5CPP_HOME)/lib -lh5cpp \
               -L$(TIFF_HOME)/lib -ltiff \
               -lm # Math library usually needed

# ============================================================
# Project Structure
# ============================================================
INC_DIR     := build/include  # For Fortran module files
APP_DIR     := build/apps
TST_DIR     := build/tests
IO_DIR      := build/io       # Build dir for io module
PROPS_DIR   := build/props    # Build dir for props module

MODULES     := io props
SRC_DIR     := $(addprefix src/,$(MODULES))
BUILD_DIR   := $(addprefix build/,$(MODULES)) # build/io build/props

# ============================================================
# Source and Object Files
# ============================================================
# Find source files
SOURCES_CPP := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SOURCES_F90 := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.F90))

# Define object file targets based on source locations
OBJECTS_IO_CPP  := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(wildcard src/io/*.cpp))
OBJECTS_PRP_CPP := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(wildcard src/props/*.cpp))
OBJECTS_PRP_F90 := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(wildcard src/props/*.F90)) # Assume F90 only in props

OBJECTS_CPP := $(OBJECTS_IO_CPP) $(OBJECTS_PRP_CPP)
OBJECTS_F90 := $(OBJECTS_PRP_F90)
OBJECTS     := $(OBJECTS_CPP) $(OBJECTS_F90)

# Let Make search for source files in src/io and src/props
vpath %.cpp src/io:src/props
vpath %.F90 src/props

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

# Rule for C++ objects in build/io
$(IO_DIR)/%.o : %.cpp
	@echo "Compiling (IO) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Rule for C++ objects in build/props
$(PROPS_DIR)/%.o : %.cpp
	@echo "Compiling (Props) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Rule for F90 objects in build/props
$(PROPS_DIR)/%.o : %.F90 | $(INC_DIR)
	@echo "Compiling (Props Fortran) $< ..."
	@mkdir -p $(@D)
	$(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $< -o $@

# ============================================================
# Linking Executables
# ============================================================

# Define object lists for clarity
APP_OBJS      := $(OBJECTS) # Main app needs all objects

TEST_OBJS_TIFF    := $(IO_DIR)/TiffReader.o
TEST_OBJS_DAT     := $(IO_DIR)/DatReader.o
TEST_OBJS_HDF5    := $(IO_DIR)/HDF5Reader.o
TEST_OBJS_VF      := $(PROPS_DIR)/VolumeFraction.o $(IO_DIR)/TiffReader.o
TEST_OBJS_TORT    := $(filter-out $(IO_DIR)/DatReader.o $(IO_DIR)/HDF5Reader.o, $(OBJECTS)) # All except Dat/HDF5 readers

# Main application
$(APP_DIR)/Diffusion: Diffusion.cpp $(APP_OBJS)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

# Test executables
$(TST_DIR)/tTiffReader: tTiffReader.cpp $(TEST_OBJS_TIFF)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tDatReader: tDatReader.cpp $(TEST_OBJS_DAT)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tHDF5Reader: tHDF5Reader.cpp $(TEST_OBJS_HDF5)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tVolumeFraction: tVolumeFraction.cpp $(TEST_OBJS_VF)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tTortuosity: tTortuosity.cpp $(TEST_OBJS_TORT)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

# ============================================================
# Supporting Targets
# ============================================================

builddirs: $(BUILD_DIR) $(APP_DIR) $(TST_DIR) $(INC_DIR)

# General rule for creating build directories
$(BUILD_DIR) $(APP_DIR) $(TST_DIR) $(INC_DIR):
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
	-@rm -rf $(IO_DIR) $(PROPS_DIR) $(INC_DIR) $(APP_DIR) $(TST_DIR)

# ============================================================
# Include Auto-Generated Dependencies
# ============================================================
# Include the .d files generated by CXXFLAGS -MMD -MP
-include $(OBJECTS_CPP:.o=.d)
