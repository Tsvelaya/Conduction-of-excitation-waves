NVCC =/usr/local/cuda/bin/nvcc
#NVCFLAGS = -arch=sm_30 --ptxas-options=-v
NVCFLAGS = -arch=sm_35
CFLAGS = -Wall -pedantic -O2
LDLIBS = -lz -lpthread

ifdef PNG
CFLAGS += -DPNG
LDLIBS += -lpng
endif

SRCS := util.cu constants.cu tabulation.cu tabulate.cu ran.c statevar.c \
	medium.c fibrosis.c step.cu cuda.cu functions.c init.c output.c s1s2.c \
	reduction.cu activation.cu options.c \
	main.c test_main.c

OBJS := util.o constants.o tabulation.o tabulate.o ran.o statevar.o \
	medium.o fibrosis.o step.o cuda.o functions.o init.o output.o s1s2.o \
	reduction.o activation.o options.o

all: .depend tnnp

tnnp: ${OBJS} main.o
	${NVCC} ${NVCFLAGS} ${LDLIBS} $^ -o $@

test-tnnp: ${OBJS} test_main.o
	${NVCC} ${NVCFLAGS} ${LDLIBS} $^ -o $@

test-alloc:  ${OBJS} main_alloc.o
	${NVCC} ${NVCFLAGS} ${LDLIBS} $^ -o $@

%.o: %.cu
	${NVCC} ${NVCFLAGS} -dc $<

ref_rev32_step5000_float.bin:
	wget http://files.hem.su/tnnp-cuda/ref_rev32_step5000_float.bin

spiral.bin:
	wget http://files.hem.su/tnnp-cuda/spiral.bin

.PHONY: depend clean test

test: test-tnnp compare ref_rev32_step5000_float.bin spiral.bin
	./test-tnnp -d 1 -i spiral.bin -v test_results && \
		./compare test_backup.bin ref_rev32_step5000_float.bin

compare: utils/compare.ml
	ocamlopt -o $@ unix.cmxa bigarray.cmxa $<

depend:
	${NVCC} -M ${SRCS} > .depend

.depend: ${SRCS} makefile
	${NVCC} -M ${SRCS} > .depend

clean:
	${RM} ${OBJS} tnnp test-tnnp compare *.cmo *.cmx *.cmi *.o

include .depend
