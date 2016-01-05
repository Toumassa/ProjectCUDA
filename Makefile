NVCCFLAGS=$(CCONFIG)
NVCCFLAGS+=`pkg-config --cflags opencv` 

NVCCLFLAGS=$(LCONFIG) `pkg-config opencv --libs` -lstdc++ 
NVCC=nvcc
		


CC = g++
LD = g++

WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual

# --- With optimisation
CPPFLAGS = -std=c++0x -DNDEBUG -O3 -msse2 $(WARNGCC) -L/usr/local/cuda-7.5/lib64 -lcudart
LDFLAGS = -DNEBUG -O3 -msse2 -L/usr/local/cuda-7.5/lib64 -lcudart

# --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 


INCLUDE_DIR =
LIB_DIR =
LIBS = `pkg-config --libs opencv`  -L/usr/local/cuda-7.5/lib64 -lcudart

simple:	sf1_cpu lab2rgb


testcpu:
	./sf1_cpu simple-data/config.txt 6 simple-data/tree

kernel.o: kernel.cu
	$(NVCC) -c -O3 kernel.cu -o kernel.o -L/usr/local/cuda-7.5/lib64 -lcudart
	
%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

main_test_transformed.o: main_test_transformed.cpp GPUAdapter.o kernel.o
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@ 

main_test_simple.o: main_test_simple.cpp
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

sf1_cpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_transformed.o GPUAdapter.o kernel.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

lab2rgb: lab2rgb.o label.o
	$(LD) $+ -o $@ $(LDFLAGS) $(LIB_DIR) $(LIBS)

clean:
	rm -f *.o sf1_cpu lab2rgb

