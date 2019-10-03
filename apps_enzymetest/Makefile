ifdef LONG
INTT = -DLONG
endif

ifdef EDGELONG
INTE = -DEDGELONG
endif

ifdef PD
PD = -DPD
endif

ifdef BYTE
CODE = -DBYTE
else ifdef NIBBLE
CODE = -DNIBBLE
else
CODE = -DBYTERLE
endif

ifdef LOWMEM
MEM = -DLOWMEM
endif

#compilers
#ifdef CILK
PCC = ./setup.sh clang++
PCFLAGS = -std=c++14 -g -lcilkrts -lblas -fno-vectorize -fno-slp-vectorize -lprofiler -I./adept-serial/include -O1 -DADEPT_STORAGE_THREAD_SAFE $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
PLFLAGS = -lcilkrts -lblas -lprofiler

#else ifdef MKLROOT
#PCC = icpc
#PCFLAGS = -std=c++14 -O3 -DCILKP $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
#
#else ifdef OPENMP
#PCC = ./setup.sh clang++
#PCFLAGS = -std=c++14 -fopenmp -lblas -lopenblas -O3 -DUSE_OPENMP -I./adept-serial/include -DADEPT_STORAGE_THREAD_SAFE $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
#PLFLAGS = -lblas -lopenblas
#else
#PCC = g++
#PCFLAGS = -std=c++14 -O3 $(INTT) $(INTE) $(CODE) $(PD) $(MEM)
#endif

COMMON= ligra.h graph.h compressedVertex.h vertex.h utils.h IO.h parallel.h gettime.h index_map.h maybe.h sequence.h edgeMap_utils.h binary_search.h quickSort.h blockRadixSort.h transpose.h parseCommandLine.h byte.h byteRLE.h nibble.h byte-pd.h byteRLE-pd.h nibble-pd.h vertexSubset.h encoder.C decoder.C

ALL= encoder decoder BFS BC BellmanFord Components Components-Shortcut Radii PageRank PageRankDelta BFSCC BFS-Bitvector KCore MIS Triangle CF GCN

all: $(ALL)

% : %.C $(COMMON)
	$(PCC) $(PCFLAGS) -fno-exceptions -S -emit-llvm -o $@.ll $<
	./setup.sh opt $@.ll -indvars -load=/efs/home/tfk/Enzyme/enzyme/build10/Enzyme/LLVMEnzyme-7.so -enzyme -o $@.bc
	$(PCC) $@.bc $(PCFLAGS) -fno-exceptions -o $@




$(COMMON):
	ln -s ../ligra/$@ .

.PHONY : clean

clean :
	rm -f *.o $(ALL)

cleansrc :
	rm -f *.o $(ALL)
	rm $(COMMON)
