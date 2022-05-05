LDFLAGS = -pthread -lpthread
CFLAGS = -g -Wall -Werror
CC = mpicc
backprop: backprop.o layer.o neuron.o readData.o
	$(CC) $(LDFLAGS) -o backprop main.o layer.o neuron.o readData.o -lm

backprop.o: main.c
	$(CC) $(CFLAGS) -c main.c

layer.o: layer.c
	$(CC) $(CFLAGS) -c layer.c

neuron.o: neuron.c
	$(CC) $(CFLAGS) -c neuron.c
readData.o: readData.c
	$(CC) $(CFLAGS) -c readData.c
# remove object files and executable when user executes "make clean"
clean:
	rm *.o backprop
