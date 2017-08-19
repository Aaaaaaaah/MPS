all:
	mpiifort $(FFLAGS) mps.f90 -lTensor -mkl
run:
	./a.out
