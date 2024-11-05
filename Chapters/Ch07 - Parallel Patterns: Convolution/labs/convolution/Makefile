
# lab-sobel-opt-judge
# Compiler settings
NVCC := nvcc
CXX := g++
CUDA_PATH := /usr/local/cuda

# Compiler flags
NVCC_FLAGS := -O3 -arch=sm_60 -Xptxas -O3 
CXXFLAGS := -O3 -Wall -Wextra

# Include and library paths
INC := -I$(CUDA_PATH)/include -I/usr/include
LIB := -L$(CUDA_PATH)/lib64 -L/usr/lib

# Libraries to link
LIBS := -lpng -lz -lcudart

# Source files
# CUDA_SOURCE := sobel.cu
# TARGET := sobel
CUDA_SOURCE := sobel_validation.cu
TARGET := sobel_validation

# Object files
OBJECTS := $(CUDA_SOURCE:.cu=.o)

# Default target
all: $(TARGET)

# Linking
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(LIB) $^ -o $@ $(LIBS)

# CUDA compilation
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

# Clean target
clean:
	rm -f $(TARGET) *.o

# Run target with example input/output
run: $(TARGET)
	./$(TARGET) input.png output.png

.PHONY: all clean run