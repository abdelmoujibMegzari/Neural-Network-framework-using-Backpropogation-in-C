#ifndef BACKPROP_H
#define BACKPROP_H

#include "layer.h"

#define SUCCESS_INIT 0
#define ERR_INIT 1

#define SUCCESS_DINIT 0
#define ERR_DINIT 1

#define SUCCESS_INIT_WEIGHTS 0
#define ERR_INIT_WEIGHTS 1

#define SUCCESS_UPDATE_WEIGHTS 0

#define SUCCESS_CREATE_ARCHITECTURE 0
#define ERR_CREATE_ARCHITECTURE 1


int init(void);
int dinit(void);

int create_architecture(void);
int initialize_weights(void);
void feed_input(int i, unsigned char **data);
void train_neural_net(void);
void forward_prop(int out);
void compute_cost(int i);
void back_prop(int p, int first_run);
void update_weights(void);
void get_inputs(unsigned char ** inputs,int rank, int processes, int isTrain);
void get_desired_outputs(unsigned char ** labels,int rank, int processes, int isTrain);
void test_nn(void);

int initialize_dummy_weights(void);


#endif



