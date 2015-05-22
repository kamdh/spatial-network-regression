CC = g++

CFLAGS = -Ofast -Wall -Wno-uninitialized 
CFLAGS += -lm -larmadillo
#CFLAGS += -static-libstdc++
#CFLAGS = -Wall -Wno-uninitialized -lm -larmadillo -g
LBFGSB_SRC = ../L-BFGS-B-C/src
MMIO_SRC = ../mmio


INCLUDES = -I/usr/include -I. -I$(LBFGSB_SRC) -I$(MMIO_SRC)
INCLUDES += -I$HOME/.local/usr/local/include
ARMADILLO = -L$HOME/.local/usr/local/lib64
#CFLAGS += $(INCLUDES)

LBFGSB=$(LBFGSB_SRC)/lbfgsb.c $(LBFGSB_SRC)/linesearch.c $(LBFGSB_SRC)/subalgorithms.c $(LBFGSB_SRC)/print.c
MMIO=$(MMIO_SRC)/libmmio.a
LINPACK = $(LBFGSB_SRC)/linpack.c
BLAS 	= $(LBFGSB_SRC)/miniCBLAS.c
#BLAS = -lblas -lgfortran
TIMER   = $(LBFGSB_SRC)/timer.c

default:
	$(CC) -o solve -I $(INCLUDES) solve.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS)

test:
	$(CC) -o test -I $(INCLUDES) test.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS)
profile:
	$(CC) -o test -I $(INCLUDES) -pg test.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS) -pg
clean:
	rm test solve
