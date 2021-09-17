# Compilers and library flags
CXX      := mpic++
F90      := mpif90
CXXFLAGS := -Wextra -O3 -fopenmp -DOMPI_SKIP_MPICXX -g -std=c++11
F90FLAGS := -g -O3
LDFLAGS  := -L/usr/lib64 -lstdc++ -lm -lamrex -ltiff -lgfortran -lHYPRE -L/usr/lib -lh5cpp -lhdf5 -I/usr/lib64/boost169
INCLUDE  := -I/usr/include/amrex -I/usr/include/hypre -I/usr/include/hdf5 -I/usr/local/include/h5cpp -I/usr/include/boost169


# Directories
INC_DIR  := build/include
APP_DIR  := build/apps
TST_DIR  := build/tests
OBJ_DIR  := build/props
IO_DIR   := build/io

MODULES   := io props
SRC_DIR   := $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES))


# Source files
SRC      := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
SRC      += $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.F90))
HEADERS  := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.H))
OBJECTS := $(patsubst src/%.cpp,build/%.o,$(SRC))

vpath %.cpp $(SRC_DIR)
vpath %.H $(SRC_DIR)
vpath %.F90 $(SRC_DIR)


# Targets
all: tests main

main: builddirs $(APP_DIR)/Diffusion

tests: builddirs $(TST_DIR)/tTiffReader $(TST_DIR)/tTiffStackReader $(TST_DIR)/tDatReader $(TST_DIR)/tVolumeFraction $(TST_DIR)/tTortuosity $(TST_DIR)/tHDF5Reader

# General compile targets
define make-object-goal
$1/%.o: %.cpp %.H
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $$< -o $$@

$1/%.o: %.F90 | $(INC_DIR)
	$(F90) $(F90FLAGS) $(INCLUDE) -J$(INC_DIR) -c $$< -o $$@
endef

# Main program
$(APP_DIR)/Diffusion: Diffusion.cpp $(addprefix build/props/,TortuosityHypre.o Tortuosity_filcc.o Tortuosity_poisson_3d.o TortuosityHypreFill.o VolumeFraction.o) build/io/TiffReader.o build/io/CathodeWrite.o build/io/DatReader.o build/io/HDF5Reader.o build/io/TiffStackReader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

# Executable tests
$(TST_DIR)/tTiffReader: tTiffReader.cpp build/io/TiffReader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tTiffStackReader: tTiffStackReader.cpp build/io/TiffStackReader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tDatReader: tDatReader.cpp build/io/DatReader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)
	
$(TST_DIR)/tHDF5Reader: tHDF5Reader.cpp build/io/HDF5Reader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)	

$(TST_DIR)/tVolumeFraction: tVolumeFraction.cpp $(addprefix build/props/,VolumeFraction.o) build/io/TiffReader.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)

$(TST_DIR)/tTortuosity: tTortuosity.cpp $(addprefix build/props/,TortuosityHypre.o Tortuosity_filcc.o Tortuosity_poisson_3d.o TortuosityHypreFill.o VolumeFraction.o) build/io/TiffReader.o build/io/CathodeWrite.o
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $^ $(LDFLAGS)


# Supporting targets
.PHONY: all builddirs clean debug release

builddirs: $(BUILD_DIR) $(APP_DIR) $(TST_DIR) $(INC_DIR)

$(BUILD_DIR) $(APP_DIR) $(TST_DIR) $(INC_DIR):
	@mkdir -p $@


# Build environments
debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

# Clean
clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(IO_DIR)/*
	-@rm -rvf $(INC_DIR)/*

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-object-goal,$(bdir))))
