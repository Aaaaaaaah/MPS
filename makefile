compile:
	mpiifort $(FFLAGS) mps.f90 -lTensor -mkl -O0 -g
run: compile
	./a.out
