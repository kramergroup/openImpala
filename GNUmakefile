# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Improved version incorporating fixes for make errors and portability.

# ============================================================
# Environment Setup (Defaults set for the Singularity container)
# ============================================================
# Default paths matching the Singularity container build.
# Can still be overridden by setting environment variables before running make.
AMREX_HOME    ?= /opt/amrex/23.11           # Base AMReX install directory
HYPRE_HOME    ?= /opt/hypre/v2.30.0         # Base HYPRE install directory
HDF5_HOME     ?= /opt/hdf5/1.12.3           # Base HDF5 install directory (built from source)
# Assuming official HDF5 C++ bindings are installed with HDF5 C library
H5CPP_HOME    ?= $(HDF5_HOME)               # Base H5CPP install directory
TIFF_HOME     ?= /usr                       # Base LibTIFF install directory (dnf package)
# BOOST_HOME    ?= /usr                       # Base Boost install directory (if needed & non-std)

# ============================================================
# Compilers and Flags
# ============================================================
CXX           := mpic++
F90           := mpif90

# Base Flags + Auto-Dependency Generation for C++ (-MMD -MP) + C++17
# Add -DOMPI_SKIP_MPICXX if needed by your MPI implementation
CXXFLAGS      := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS      := -g -O3

# <<< ADDED -Isrc for easier project includes >>>
INCLUDE       := -Isrc \
                 -I$(AMREX_HOME)/include \
                 -I$(HYPRE_HOME)/include \
                 -I$(HDF5_HOME)/include \
                 -I$(H5CPP_HOME)/include \
                 -I$(TIFF_HOME)/include
#                -I$(BOOST_HOME)/include  # If Boost headers are needed and non-standard

# Linker Flags (using variables for portability)
# Rely on MPI wrappers to link MPI libs and potentially Fortran runtime
# List required application libraries
# <<< Using updated *_HOME defaults >>>
LDFLAGS       := -L$(AMREX_HOME)/lib -lamrex \
                 -L$(HYPRE_HOME)/lib -lHYPRE \
                 -L$(HDF5_HOME)/lib -lhdf5 -lhdf5_cpp \
                 -L$(TIFF_HOME)/lib64 -ltiff \
                 -lm # Link math library

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
SOURCES_APP_CPP := $(wildcard src/*.cpp) # Find any .cpp files directly in src/ (like Diffusion.cpp if moved)
SOURCES_TST_CPP := $(wildcard tests/*.cpp) # Find test files in tests/ directory

# Define object file targets based on source locations
OBJECTS_IO_CPP  := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_CPP))
OBJECTS_PRP_CPP := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_CPP))
OBJECTS_PRP_F90 := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(SOURCES_PRP_F90))

OBJECTS_CPP := $(OBJECTS_IO_CPP) $(OBJECTS_PRP_CPP)
OBJECTS_F90 := $(OBJECTS_PRP_F90)
OBJECTS       := $(OBJECTS_CPP) $(OBJECTS_F90)

# Let Make search for source files in relevant directories
# Adding src and tests directories to VPATH
VPATH := $(subst $(space),:,$(SRC_DIRS)):src:tests

# ============================================================
# Main Targets
# ============================================================
.PHONY: all main tests builddirs clean debug release

all: main tests

# Assume main application source is src/Diffusion.cpp
main: builddirs $(APP_DIR)/Diffusion

# Define test executables explicitly
TEST_EXECS := $(patsubst tests/%.cpp,$(TST_DIR)/%,$(SOURCES_TST_CPP))
tests: builddirs $(TEST_EXECS)

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

# Rule for top-level src/*.cpp files (if any exist) - goes to build/ dir perhaps?
# Or adjust if main app source is elsewhere. Assuming main app source might be in src/
# build/%.o : %.cpp | build
# 	@echo "Compiling (Main) $< ..."
# 	@mkdir -p $(@D)
# 	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# ============================================================
# Linking Executables
# ============================================================

# Define object lists needed by the main application and tests
# Main App depends on all C++ and Fortran objects
APP_OBJS          := $(OBJECTS)

# Specific objects needed for each test (adjust based on actual dependencies)
# Assumes test sources are named like tests/t<ClassName>.cpp
TEST_OBJS_TiffReader       := $(IO_DIR)/TiffReader.o
TEST_OBJS_DatReader        := $(IO_DIR)/DatReader.o
TEST_OBJS_HDF5Reader       := $(IO_DIR)/HDF5Reader.o
TEST_OBJS_VolumeFraction   := $(PROPS_DIR)/VolumeFraction.o $(IO_DIR)/TiffReader.o # Example dependency
TEST_OBJS_TortuosityHypre  := $(PROPS_DIR)/TortuosityHypre.o $(PROPS_DIR)/VolumeFraction.o $(OBJECTS_PRP_F90) $(OBJECTS_IO_CPP) # Depends on most things
TEST_OBJS_TortuosityDirect := $(PROPS_DIR)/TortuosityDirect.o $(PROPS_DIR)/VolumeFraction.o $(OBJECTS_PRP_F90) $(OBJECTS_IO_CPP) # Depends on most things


# Main application (Assuming src/Diffusion.cpp)
$(APP_DIR)/Diffusion: src/Diffusion.cpp $(APP_OBJS) | $(APP_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use CXX for linking C++/Fortran with MPI wrapper

# Test executables (using specific object lists)
$(TST_DIR)/tTiffReader: tests/tTiffReader.cpp $(TEST_OBJS_TiffReader) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tDatReader: tests/tDatReader.cpp $(TEST_OBJS_DatReader) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tHDF5Reader: tests/tHDF5Reader.cpp $(TEST_OBJS_HDF5Reader) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tVolumeFraction: tests/tVolumeFraction.cpp $(TEST_OBJS_VolumeFraction) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

# Combine Tortuosity tests if they share many objects, or keep separate
$(TST_DIR)/tTortuosityDirect: tests/tTortuosityDirect.cpp $(TEST_OBJS_TortuosityDirect) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tTortuosityHypre: tests/tTortuosityHypre.cpp $(TEST_OBJS_TortuosityHypre) | $(TST_DIR)
	@echo "Linking $@ ..."
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)


# ============================================================
# Supporting Targets
# ============================================================

# Target to create all necessary build directories
builddirs: $(APP_DIR) $(TST_DIR) $(INC_DIR) $(BUILD_DIRS)

# General rule for creating build directories
$(APP_DIR) $(TST_DIR) $(INC_DIR) $(BUILD_DIRS):
	@echo "Creating directory $@"
	@mkdir -p $@

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
# Include the .d files generated by CXXFLAGS -MMD -MP
# Put this near the end so Make knows all explicit rules first
-include $(OBJECTS_CPP:.o=.d)

# ============================================================
# Debugging Help (Uncomment to use)
# ============================================================
# print-%:
# 	@echo '$* = $($*)'
# Example: make print-OBJECTS_CPP
