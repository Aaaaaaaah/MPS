compile:
	mpiifort $(FFLAGS) mps.f90 -lTensor -mkl -O0 -g
run: compile
	./a.out 0.1
	./a.out 0.01
	./a.out 0.001
	./a.out 0.0001
	./a.out 0.00001
	./a.out 0.000001
	./a.out 0.0000001
