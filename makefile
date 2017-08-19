all:
	mpiifort $(FFLAGS) mps.f90 -lTensor -mkl -O0
run:
	./a.out
