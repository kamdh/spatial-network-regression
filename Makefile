CC = g++

CFLAGS = -O3 -Wall -Wno-uninitialized -fpic -lm -larmadillo
#CFLAGS += -static-libstdc++
#CFLAGS = -Wall -Wno-uninitialized -lm -larmadillo -g
LBFGSB_SRC = ../L-BFGS-B-C/src
MMIO_SRC = ../mmio

INCLUDES = -I/usr/include -I. -I$(LBFGSB_SRC) -I$(MMIO_SRC)
INCLUDES += -I$(HOME)/.local/usr/local/include
ARMADILLO = -L$(HOME)/.local/usr/local/lib64
#CFLAGS += $(INCLUDES)

LBFGSB=$(LBFGSB_SRC)/lbfgsb.c $(LBFGSB_SRC)/linesearch.c $(LBFGSB_SRC)/subalgorithms.c $(LBFGSB_SRC)/print.c
MMIO=$(MMIO_SRC)/libmmio.a
LINPACK = $(LBFGSB_SRC)/linpack.c
TIMER   = $(LBFGSB_SRC)/timer.c
BLAS 	= $(LBFGSB_SRC)/miniCBLAS.c
#BLAS = -lblas -lgfortran

default:
	$(CC) -o solve $(INCLUDES) solve.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS)

save_test:
	$(CC) -o save_test -I $(INCLUDES) save_test.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS)

test:
	$(CC) -o test -I $(INCLUDES) test.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS)

all: default save_test test

profile:
	$(CC) -o test -I $(INCLUDES) -pg test.cpp functions.cpp $(LBFGSB) $(MMIO) $(LINPACK) $(BLAS) $(TIMER) $(CFLAGS) -pg

clean:
	rm test solve save_test
