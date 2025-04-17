# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Corrected version 2: Uses Static Pattern Rules for compilation.

# ============================================================
# Environment Setup (Defaults set for the Singularity container)
# ============================================================
# Default paths matching the Singularity container build.
# Can still be overridden by setting environment variables before running make.
AMREX_HOME    ?= /opt/amrex/23.11          # Base AMReX install directory
HYPRE_HOME    ?= /opt/hypre/v2.30.0         # Base HYPRE install directory
HDF5_HOME     ?= /opt/hdf5/1.12.3         # Base HDF5 install directory (built from source)
H5CPP_HOME    ?= $(HDF5_HOME)              # Base H5CPP install directory
TIFF_HOME     ?= /usr                       # Base LibTIFF install directory (dnf package)

# ============================================================
# Compilers and Flags
# ============================================================
CXX           := mpic++
F90           := mpif90

CXXFLAGS      := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS      := -g -O3

# Include project source dir and dependency includes
INCLUDE       := -Isrc \
                 -I$(AMREX_HOME)/include \
                 -I$(HYPRE_HOME)/include \
                 -I$(HDF5_HOME)/include \
                 -I$(H5CPP_HOME)/include \
                 -I$(TIFF_HOME)/include

# Linker Flags
LDFLAGS       := -L$(AMREX_HOME)/lib -lamrex \
                 -L$(HYPRE_HOME)/lib -lHYPRE \
                 -L$(HDF5_HOME)/lib -lhdf5 -lhdf5_cpp \
                 -L$(TIFF_HOME)/lib64 -ltiff \
                 -lm

# ============================================================
# Project Structure
# ============================================================
INC_DIR       := build/include # For Fortran modules
APP_DIR       := build/apps    # For main executable
TST_DIR       := build/tests   # For test executables
IO_DIR        := build/io      # For IO object files
PROPS_DIR     := build/props   # For Props object files

MODULES       := io props
SRC_DIRS      := $(addprefix src/,$(MODULES)) # src/io src/props
BUILD_DIRS    := $(addprefix build/,$(MODULES)) # build/io build/props

# ============================================================
# Source and Object Files
# ============================================================
# Find source files
SOURCES_IO_CPP  := $(wildcard src/io/*.cpp)
# All non-test CPP files in src/props (includes Diffusion.cpp, Tortuosity*.cpp etc.)
SOURCES_PRP_CPP_ALL := $(wildcard src/props/*.cpp)
# Test CPP files in src/props (assuming they start with 't')
SOURCES_TST_CPP := $(filter src/props/t%.cpp, $(SOURCES_PRP_CPP_ALL))
# Non-test CPP files in src/props
SOURCES_PRP_CPP := $(filter-out $(SOURCES_TST_CPP), $(SOURCES_PRP_CPP_ALL))
# Fortran files
SOURCES_PRP_F90 := $(wildcard src/props/*.F90)

# Define object file targets based on source locations
OBJECTS_IO_CPP  := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_CPP))
OBJECTS_PRP_CPP := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_CPP))
OBJECTS_PRP_F90 := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(SOURCES_PRP_F90))
# Test objects (compile test sources to the props build dir as well)
OBJECTS_TST_CPP := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_TST_CPP))

# Consolidate non-test objects
OBJECTS_APP_CPP := $(OBJECTS_IO_CPP) $(OBJECTS_PRP_CPP)
OBJECTS_APP_F90 := $(OBJECTS_PRP_F90)
OBJECTS_APP     := $(OBJECTS_APP_CPP) $(OBJECTS_APP_F90)

# Let Make search for source files in relevant directories
VPATH := $(subst $(space),:,$(SRC_DIRS)):src

# ============================================================
# Main Targets
# ============================================================
.PHONY: all main tests clean debug release

all: main tests

# Main application executable (Diffusion)
main: $(APP_DIR)/Diffusion

# Define test executables based on found test sources
TEST_EXECS := $(patsubst src/props/%.cpp,$(TST_DIR)/%,$(SOURCES_TST_CPP))
tests: $(TEST_EXECS)

# ============================================================
# Compilation Rules (Using Static Pattern Rules) # <<< SECTION REPLACED >>>
# ============================================================

# Static Pattern Rule for C++ objects in build/io
$(OBJECTS_IO_CPP): $(IO_DIR)/%.o : src/io/%.cpp
	@echo "Compiling (IO) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for non-test C++ objects in build/props
$(OBJECTS_PRP_CPP): $(PROPS_DIR)/%.o : src/props/%.cpp
	@echo "Compiling (Props) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for F90 objects in build/props
$(OBJECTS_PRP_F90): $(PROPS_DIR)/%.o : src/props/%.F90
	@echo "Compiling (Props Fortran) $< ..."
	@mkdir -p $(@D) $(INC_DIR) # Ensure both obj and mod dirs exist
	$(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $< -o $@

# Static Pattern Rule for test C++ objects (output to build/props)
$(OBJECTS_TST_CPP): $(PROPS_DIR)/%.o : src/props/%.cpp
	@echo "Compiling (Test) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# ============================================================
# Linking Executables
# ============================================================

# Define object lists needed by the main application and tests
# Main App depends on all non-test C++ and Fortran objects
APP_OBJS          := $(OBJECTS_APP)

# Specific objects needed for each test (adjust based on actual dependencies)
# Using OBJECTS_APP includes all non-test code objects, which might be simplest
TEST_DEPS_BASE    := $(OBJECTS_APP) # Assume most tests need most app objects for now

# Main application
$(APP_DIR)/Diffusion: src/props/Diffusion.cpp $(APP_OBJS)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $< $(APP_OBJS) $(LDFLAGS) # Link source + objects

# Test executables (General rule using specific test object + base dependencies)
# This links the specific t*.o file with TEST_DEPS_BASE
$(TST_DIR)/t%: $(PROPS_DIR)/t%.o $(TEST_DEPS_BASE)
	@echo "Linking $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# ============================================================
# Supporting Targets
# ============================================================

# Build environments (simple examples)
debug: CXXFLAGS += -DDEBUG # Example C++ debug flag
debug: F90FLAGS += -fbounds-check # Example Fortran debug flag
debug: all

release: # Using defaults: -O3 defined initially
release: all

# Clean target (removes objects, module files, executables, dependency files)
clean:
	@echo "Cleaning build artifacts..."
	-@rm -rf build # Remove entire build directory is simplest
	# Clean dependency files potentially generated in source dirs if CXXFLAGS didn't handle output dir well
	-@find src -name '*.d' -delete
	-@find . -name '*.mod' -delete # Remove Fortran module files from source tree if accidentally created there

# ============================================================
# Include Auto-Generated Dependencies
# ============================================================
# Include the .d files generated by CXXFLAGS -MMD -MP for C++ files
# Includes both app and test objects dependencies now
-include $(OBJECTS_APP_CPP:.o=.d) $(OBJECTS_TST_CPP:.o=.d)

# ============================================================
# Debugging Help (Uncomment to use)
# ============================================================
# print-%:
# 	@echo '$* = $($*)'
# Example: make print-OBJECTS_APP
