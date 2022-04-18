LDFLAGS = -pthread -lpthread
CCS= mpicc
CFLAGS = -g -Wall #-Werror
backprop: backprop.o layer.o neuron.o
	$(CCS) $(LDFLAGS) -o backprop main.o layer.o neuron.o -lm

backprop.o: main.c
	$(CCS) $(CFLAGS) -c main.c

layer.o: layer.c
	$(CCS) $(CFLAGS) -c layer.c

neuron.o: neuron.c
	$(CCS) $(CFLAGS) -c neuron.c

# remove object files and executable when user executes "make clean"
clean:
	rm *.o backprop
