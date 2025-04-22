# GNU MakeFile for OpenImpala Diffusion Application and Tests
# Corrected version 2: Uses Static Pattern Rules for compilation.

# ============================================================
# Environment Setup (Defaults set for the Singularity container)
# ============================================================
# Default paths matching the Singularity container build.
# Can still be overridden by setting environment variables before running make.
AMREX_HOME    ?= /opt/amrex/23.11
HYPRE_HOME    ?= /opt/hypre/v2.30.0
HDF5_HOME     ?= /opt/hdf5/1.12.3
H5CPP_HOME    ?= $(HDF5_HOME)
TIFF_HOME     ?= /usr

# ============================================================
# Compilers and Flags
# ============================================================
CXX           := mpic++
F90           := mpif90

CXXFLAGS      := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++17 -MMD -MP
F90FLAGS      := -g -O3

# Include project source dir and dependency includes
INCLUDE       := -Isrc \
		 -Isrc/props \
                 -I$(AMREX_HOME)/include \
                 -I$(HYPRE_HOME)/include \
                 -I$(HDF5_HOME)/include \
                 -I$(TIFF_HOME)/include

# Linker Flags
LDFLAGS       := -L$(AMREX_HOME)/lib -lamrex \
                 -L$(HYPRE_HOME)/lib -lHYPRE \
                 -L$(HDF5_HOME)/lib -lhdf5 -lhdf5_cpp \
                 -L$(TIFF_HOME)/lib64 -ltiff \
                 -lm \
                 -lgfortran

# ============================================================
# Project Structure
# ============================================================
INC_DIR       := build/include# For Fortran modules
APP_DIR       := build/apps# For main executable
TST_DIR       := build/tests# For test executables
IO_DIR        := build/io# For IO object files   <<< REMOVE SPACES BEFORE #
PROPS_DIR     := build/props# For Props object files <<< REMOVE SPACES BEFORE #

MODULES       := io props
SRC_DIRS      := $(addprefix src/,$(MODULES))# src/io src/props
BUILD_DIRS    := $(addprefix build/,$(MODULES))# build/io build/props

# ============================================================
# Source and Object Files (Revised for Linking - v4 style)
# ============================================================
# --- Find source files ---
SOURCES_IO_ALL     := $(wildcard src/io/*.cpp)
SOURCES_PRP_ALL    := $(wildcard src/props/*.cpp)
SOURCES_F90_ALL    := $(wildcard src/props/*.F90) # Assuming Fortran only in props

# --- Identify Drivers (containing main()) ---
SOURCES_APP_DRIVER := src/props/Diffusion.cpp
SOURCES_IO_TESTS   := $(filter src/io/t%.cpp, $(SOURCES_IO_ALL))
SOURCES_PRP_TESTS  := $(filter src/props/t%.cpp, $(SOURCES_PRP_ALL))
SOURCES_TEST_DRIVERS:= $(SOURCES_IO_TESTS) $(SOURCES_PRP_TESTS)

# --- Identify Library Sources (excluding drivers) ---
SOURCES_IO_LIB     := $(filter-out $(SOURCES_IO_TESTS), $(SOURCES_IO_ALL))
SOURCES_PRP_LIB    := $(filter-out $(SOURCES_PRP_TESTS) $(SOURCES_APP_DRIVER), $(SOURCES_PRP_ALL))
SOURCES_F90_LIB    := $(SOURCES_F90_ALL) # Assuming all F90 are library code

# --- Define Object Files based on Categories ---
OBJECTS_APP_DRIVER := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_APP_DRIVER))
OBJECTS_IO_TESTS   := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_TESTS))
OBJECTS_PRP_TESTS  := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_TESTS))
OBJECTS_TEST_DRIVERS:= $(OBJECTS_IO_TESTS) $(OBJECTS_PRP_TESTS)

TEST_EXECS_IO      := $(patsubst src/io/%.cpp,$(TST_DIR)/%,$(SOURCES_IO_TESTS))
TEST_EXECS_PRP      := $(patsubst src/props/%.cpp,$(TST_DIR)/%,$(SOURCES_PRP_TESTS))

OBJECTS_IO_LIB     := $(patsubst src/io/%.cpp,$(IO_DIR)/%.o,$(SOURCES_IO_LIB))
OBJECTS_PRP_LIB    := $(patsubst src/props/%.cpp,$(PROPS_DIR)/%.o,$(SOURCES_PRP_LIB))
OBJECTS_F90_LIB    := $(patsubst src/props/%.F90,$(PROPS_DIR)/%.o,$(SOURCES_F90_LIB))

# --- Consolidate Library Objects ---
OBJECTS_LIB_ALL    := $(OBJECTS_IO_LIB) $(OBJECTS_PRP_LIB) $(OBJECTS_F90_LIB)

# Let Make search for source files in relevant directories
VPATH := $(subst $(space),:,$(SRC_DIRS)):src

# ============================================================
# Main Targets
# ============================================================
.PHONY: all main tests test clean debug release

all: main tests

# Main application executable (Diffusion)
main: $(APP_DIR)/Diffusion

# Define test executables based on found test sources
TEST_EXECS := $(patsubst src/props/%.cpp,$(TST_DIR)/%,$(SOURCES_TST_CPP))
tests: $(TEST_EXECS_IO) $(TEST_EXECS_PRP)

# Target to run the tests after they are built, linking specific input files
test: tests
	@echo ""
	@echo "--- Running Tests ---"
	@passed_all=true; \
	list_of_tests='$(TEST_EXECS_IO) $(TEST_EXECS_PRP)'; \
	rm -f ./inputs; \
	if [ -z "$$list_of_tests" ]; then \
	    echo "No test executables found to run."; \
	else \
	    for tst in $$list_of_tests; do \
	        echo "Running test $$tst..."; \
	        test_name=$$(basename $$tst); \
	        input_file="tests/inputs/$${test_name}.inputs"; \
	        if [ -f "$$input_file" ]; then \
	            echo "  Using input file: $$input_file"; \
	            ln -sf "$$input_file" ./inputs; \
	        else \
	            echo "  Warning: No specific input file found at $$input_file. Running without './inputs'."; \
	            rm -f ./inputs; \
	            # Optional: Fail if input file is missing (uncomment below) \
	            # echo "  ERROR: Required input file $$input_file not found for test $$test_name!"; \
	            # passed_all=false; \
	            # continue; \
	        fi; \
	        if mpirun -np 1 --allow-run-as-root $$tst; then \
	            echo "  PASS: $$tst"; \
	        else \
	            echo "  FAIL: $$tst"; \
	            passed_all=false; \
	        fi; \
	        rm -f ./inputs; \
	    done; \
	fi; \
	echo "--- Test Summary ---"; \
	if $$passed_all; then \
	    echo "All tests passed."; \
	    exit 0; \
	else \
	    echo "ERROR: One or more tests failed."; \
	    exit 1; \
	fi

# ============================================================
# Compilation Rules (Using Structure Similar to Working v2)
# ============================================================

# Static Pattern Rule for IO Lib C++ objects
$(OBJECTS_IO_LIB): $(IO_DIR)/%.o: src/io/%.cpp
	@echo "Compiling (IO Lib) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for IO Test C++ objects
$(OBJECTS_IO_TESTS): $(IO_DIR)/%.o: src/io/%.cpp
	@echo "Compiling (IO Test) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props Lib C++ objects
$(OBJECTS_PRP_LIB): $(PROPS_DIR)/%.o: src/props/%.cpp
	@echo "Compiling (Props Lib) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props App Driver C++ object
$(OBJECTS_APP_DRIVER): $(PROPS_DIR)/%.o: src/props/%.cpp
	@echo "Compiling (App Driver) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props Test C++ objects
$(OBJECTS_PRP_TESTS): $(PROPS_DIR)/%.o: src/props/%.cpp
	@echo "Compiling (Props Test) $< ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

# Static Pattern Rule for Props Lib F90 objects
$(OBJECTS_F90_LIB): $(PROPS_DIR)/%.o: src/props/%.F90
	@echo "Compiling (Props Fortran) $< ..."
	@mkdir -p $(@D) $(INC_DIR) # Ensure both obj and mod dirs exist
	$(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $< -o $@

# ============================================================
# Linking Executables (Revised Rules - v4 style)
# ============================================================

# Main application
$(APP_DIR)/Diffusion: $(OBJECTS_APP_DRIVER) $(OBJECTS_LIB_ALL)
	@echo "Linking Main App $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use $^ for all prerequisites

# Test executables from src/props
# Note: This relies on OBJECTS_PRP_TESTS defining the specific .o files
$(TEST_EXECS_PRP): $(TST_DIR)/%: $(PROPS_DIR)/%.o $(OBJECTS_LIB_ALL)
	@echo "Linking Test $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use $^ for all prerequisites

# Test executables from src/io
# Note: This relies on OBJECTS_IO_TESTS defining the specific .o files
$(TEST_EXECS_IO): $(TST_DIR)/%: $(IO_DIR)/%.o $(OBJECTS_LIB_ALL)
	@echo "Linking Test $@ ..."
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) # Use $^ for all prerequisites

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
-include $(OBJECTS_IO_LIB:.o=.d) $(OBJECTS_PRP_LIB:.o=.d) $(OBJECTS_APP_DRIVER:.o=.d) $(OBJECTS_TEST_DRIVERS:.o=.d)

# ============================================================
# Debugging Help (Uncomment to use)
# ============================================================
# print-%:
# 	@echo '$* = $($*)'
# Example: make print-OBJECTS_APP
