#include "backprop.h"
#include "layer.h"
#include "neuron.h"
#include <mpi.h>

#include "readData.h"

#define NUM_LAYERS  3
#define NUM_NEURONS_0  NUM_COL*NUM_ROWS
#define NUM_NEURONS_1  32
layer *lay = NULL;
int num_layers;
int *num_neurons;
float alpha;
float *cost;
float full_cost;
unsigned char **input;
unsigned char **desired_outputs;
unsigned char **input_test;
unsigned char **desired_outputs_test;
int n=1;
int num_testSamples_ex = 2000;
int p,P,tag,rc;
float weights[NUM_LAYERS][NUM_NEURONS_0][NUM_NEURONS_1];
float dweights[NUM_LAYERS][NUM_NEURONS_0][NUM_NEURONS_1];
float bias[NUM_LAYERS][NUM_NEURONS_0];
float dbias[NUM_LAYERS][NUM_NEURONS_0];
int s=0;
MPI_Status  status;


int main(int argc,char * argv[])
{
    int i;
    rc = MPI_Init(&argc,&argv);
    rc = MPI_Comm_size(MPI_COMM_WORLD,&P);
    rc = MPI_Comm_rank(MPI_COMM_WORLD,&p);
    tag =45;



    /**** Initialize parameters ****/
    num_layers = NUM_LAYERS;

    num_neurons = (int*) malloc(num_layers * sizeof(int));
    memset(num_neurons, 0,num_layers *sizeof(int));

    // Neurons per layer
    num_neurons[0] = NUM_NEURONS_0;
    num_neurons[1] = NUM_NEURONS_1;
    num_neurons[2] = 10;

    /* => Total Number of Parameters is 25,450 */

    // Learning rate
    alpha = 0.15;


    // Initialize the neural network module
    if(init()!= SUCCESS_INIT)
    {
        printf("Error in Initialization...\n");
        exit(0);
    }


    int num_training_ex = 2000;


    /****** Mofidy this part to take inputs from file directly ******/

    input = (unsigned char **) malloc(num_training_ex * sizeof(unsigned char *));
    for(i=0;i<num_training_ex;i++)
    {
        input[i] = (unsigned char *) malloc(num_neurons[0] * sizeof(unsigned char));
    }

    /************************Desired outputs Modify as well ***********/
    desired_outputs = (unsigned char **) malloc(num_training_ex* sizeof(unsigned char*));
    for(i=0; i<num_training_ex; i++)
    {
        desired_outputs[i] = (unsigned char *) malloc(num_neurons[num_layers-1] * sizeof(unsigned char));
    }
    /******************************************************************/

    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
    memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

    // Get Training Examples #Modify
    get_inputs(input, 1);

    // Get Output Labels #Modify
    get_desired_outputs(desired_outputs, 1);

    train_neural_net();



    /****** Mofidy this part to take inputs from file directly ******/

    input_test = (unsigned char **) malloc(num_testSamples_ex * sizeof(unsigned char *));
    for(int i=0;i<num_testSamples_ex;i++)
    {
        input_test[i] = (unsigned char *) malloc(num_neurons[0] * sizeof(unsigned char));
    }

    /************************Desired outputs Modify as well ***********/
    desired_outputs_test = (unsigned char **) malloc(num_testSamples_ex* sizeof(unsigned char*));
    for(int i=0; i<num_testSamples_ex; i++)
    {
        desired_outputs_test[i] = (unsigned char *) malloc(num_neurons[num_layers-1] * sizeof(unsigned char));
    }
    /******************************************************************/


    test_nn();

    if(dinit()!= SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }

    return 0;
}


int init()
{
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

//Get Inputs
void  get_inputs(unsigned char **dataInput, int isTrain)
{
    /** Get input training data **/
    loadData(1, 1, dataInput, isTrain);
}

//Get Labels
void get_desired_outputs(unsigned char **labels, int isTrain)
{
    loadLabels(1, 1, labels, isTrain);
}

// Feed inputs to input layer
void feed_input(int i, unsigned char **inputSet)
{
    int j;

    for(j=0;j<num_neurons[0];j++)
    {
        lay[0].neu[j].actv = inputSet[i][j];
        //printf("Input: %f\n",lay[0].neu[j].actv);
    }
}

// Create Neural Network Architecture
int create_architecture()
{
    int i=0,j=0;
    lay = (layer*) malloc(num_layers * sizeof(layer));

    for(i=0;i<num_layers;i++)
    {
        lay[i] = create_layer(num_neurons[i]);
        lay[i].num_neu = num_neurons[i];
        printf("Created Layer: %d\n", i+1);
        printf("Number of Neurons in Layer %d: %d\n", i+1,lay[i].num_neu);

        for(j=0;j<num_neurons[i];j++)
        {
            if(i < (num_layers-1))
            {
                lay[i].neu[j] = create_neuron(num_neurons[i+1]);
            }

            printf("Neuron %d in Layer %d created\n",j+1,i+1);
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(p==0) {
        if (initialize_weights() != SUCCESS_INIT_WEIGHTS) {
            printf("Error Initilizing weights...\n");
            return ERR_CREATE_ARCHITECTURE;
        }
    }


    return SUCCESS_CREATE_ARCHITECTURE;
}

int initialize_weights(void)
{
    int i,j,k;

    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");

    for(i=0;i<num_layers-1;i++)
    {
        
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Initialize Output Weights for each neuron

                lay[i].neu[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                weights[i][j][k]=lay[i].neu[j].out_weights[k];
                printf("%d:w[%d][%d]: %f\n",k,i,j, lay[i].neu[j].out_weights[k]);
                lay[i].neu[j].dw[k] = 0.0;
                dweights[i][j][k]=0.0;
            }

            if(i>0) 
            {
                lay[i].neu[j].bias = ((double)rand())/((double)RAND_MAX);
                bias[i][j]=lay[i].neu[j].bias;
            }
        }
    }   
    printf("\n");
    
    for (j=0; j<num_neurons[num_layers-1]; j++)
    {
        lay[num_layers-1].neu[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}

// Train Neural Network
void train_neural_net(void)
{
    int i;
    int it=0;

    // Gradient Descent
    for(it=0;it<20000;it++)
    {
        //printf("%d\n",it);
        int first_run=1;
        if(p==0){
            //TODO IMPROVE SCATRING METHOD TO be in log(N)
            for(int j=1;j<P;j++){
                rc = MPI_Send(weights, num_layers*s*s*sizeof(float), MPI_REAL, j, tag, MPI_COMM_WORLD);
                rc = MPI_Send(bias, num_layers*s*sizeof(float), MPI_REAL, j, tag, MPI_COMM_WORLD);
            }
        }else{

            rc = MPI_Recv(weights, num_layers*s*s*sizeof(float), MPI_REAL, 0, tag, MPI_COMM_WORLD, &status);
            rc = MPI_Recv(bias, num_layers*s*sizeof(float), MPI_REAL, 0, tag, MPI_COMM_WORLD, &status);
            for(int i=0;i<num_layers-1;i++){
                for(int j=0; j<num_neurons[i]; j++){
                    for(int k=0;k<num_neurons[i+1];k++){
                        lay[i].neu[j].out_weights[k]=weights[i][j][k];
                    }
                    lay[i].neu[j].bias=bias[i][j];
                }
            }
        }
        for(i=0;i<num_training_ex;i++)
        {
            feed_input(i, input);
            forward_prop(0);
            compute_cost(i);
            back_prop(i,first_run);
            first_run=0;
        }
        if(p==0){
            //TODO IMPROVE gathering METHOD TO be in log(N)
            for(int s=1;s<P;s++) {
                rc = MPI_Recv(dweights, num_layers * s * s * sizeof(float), MPI_REAL, s, tag, MPI_COMM_WORLD,
                              &status);
                rc = MPI_Recv(dbias, num_layers * s * sizeof(float), MPI_REAL, s, tag, MPI_COMM_WORLD, &status);
                for (int i = 0; i < num_layers - 1; i++) {
                    for (int j = 0; j < num_neurons[i]; j++) {
                        for (int k = 0; k < num_neurons[i + 1]; k++) {
                            lay[i].neu[j].dw[k] += dweights[i][j][k];
                            //printf("%f, ",dweights[i][j][k]);

                        }
                        lay[i].neu[j].dbias += dbias[i][j];
                    }
                }
            }
            for (int i = 0; i < num_layers - 1; i++) {
                for (int j = 0; j < num_neurons[i]; j++) {
                    for (int k = 0; k < num_neurons[i + 1]; k++) {
                        //printf("%f, ",dweights[i][j][k]);
                    }
                }
            }
            update_weights();
        }else{

            rc = MPI_Send(dweights, num_layers*s*s*sizeof(float), MPI_REAL, 0, tag, MPI_COMM_WORLD);
            rc = MPI_Send(dbias, num_layers*s*sizeof(float), MPI_REAL, 0, tag, MPI_COMM_WORLD);
        }

    }
}



void update_weights(void)
{
    int i,j,k;

    for(i=0;i<num_layers-1;i++)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            for(k=0;k<num_neurons[i+1];k++)
            {
                // Update Weights
                lay[i].neu[j].out_weights[k] = (lay[i].neu[j].out_weights[k]) - (alpha * lay[i].neu[j].dw[k]);
            }
            
            // Update Bias
            lay[i].neu[j].bias = lay[i].neu[j].bias - (alpha * lay[i].neu[j].dbias);
        }
    }   
}

void forward_prop(int out)
{
    int i,j,k;

    for(i=1;i<num_layers;i++)
    {   
        for(j=0;j<num_neurons[i];j++)
        {
            lay[i].neu[j].z = lay[i - 1].neu[j].bias;

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i].neu[j].z  = lay[i].neu[j].z + ((lay[i-1].neu[k].out_weights[j]) * (lay[i-1].neu[k].actv));
            }

            // Relu Activation Function for Hidden Layers
            if(i < num_layers-1)
            {
                if((lay[i].neu[j].z) < 0)
                {
                    lay[i].neu[j].actv = 0;
                }

                else
                {
                    lay[i].neu[j].actv = lay[i].neu[j].z;
                }
            }
            
            // Sigmoid Activation function for Output Layer
            else
            {
                lay[i].neu[j].actv = 1/(1+exp(-lay[i].neu[j].z));
                if(out) {
                    printf("Output: %d\n", (int) round(lay[i].neu[j].actv));
                    printf("\n");
                }
            }
        }
    }
}

// Compute Total Cost
void compute_cost(int i)
{
    int j;
    float tmpcost=0;
    float tcost=0;

    for(j=0;j<num_neurons[num_layers-1];j++)
    {
        tmpcost = desired_outputs[i][j] - lay[num_layers-1].neu[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }   

    full_cost = (full_cost + tcost)/n;
    n++;
    // printf("Full Cost: %f\n",full_cost);
}

// Back Propogate Error
void back_prop(int p, int first_run)
{
    int i,j,k;

    // Output Layer
    for(j=0;j<num_neurons[num_layers-1];j++)
    {           
        lay[num_layers-1].neu[j].dz = (lay[num_layers-1].neu[j].actv - desired_outputs[p][j]) * (lay[num_layers-1].neu[j].actv) * (1- lay[num_layers-1].neu[j].actv);

        for(k=0;k<num_neurons[num_layers-2];k++)
        {   
            lay[num_layers-2].neu[k].dw[j] = lay[num_layers-2].neu[k].dw[j] + (lay[num_layers-1].neu[j].dz * lay[num_layers-2].neu[k].actv);
            if(first_run)
                dweights[num_layers-2][k][j] =0;
            dweights[num_layers-2][k][j] +=lay[num_layers-2].neu[k].dw[j];
            lay[num_layers-2].neu[k].dactv = lay[num_layers-2].neu[k].out_weights[j] * lay[num_layers-1].neu[j].dz;
            //TODO do we also need d actv to update weights.
        }
            
        lay[num_layers-1].neu[j].dbias = lay[num_layers-1].neu[j].dbias + lay[num_layers-1].neu[j].dz;
        if(first_run)
            dbias[num_layers-1][j]=0;
        dbias[num_layers-1][j]+=lay[num_layers-1].neu[j].dbias;

    }

    // Hidden Layers
    for(i=num_layers-2;i>0;i--)
    {
        for(j=0;j<num_neurons[i];j++)
        {
            if(lay[i].neu[j].z >= 0)
            {
                lay[i].neu[j].dz = lay[i].neu[j].dactv;
                //printf("actv %f,\n",lay[i].neu[j].dactv);
                //printf("z %f,\n",lay[i].neu[j].z);
                //printf("dz %f,\n",lay[i].neu[j].dz);
            }
            else
            {
                lay[i].neu[j].dz = 0;
            }

            for(k=0;k<num_neurons[i-1];k++)
            {
                lay[i-1].neu[k].dw[j] = lay[i-1].neu[k].dw[j] + lay[i].neu[j].dz * lay[i-1].neu[k].actv;


                //printf("actv %f,\n",lay[i-1].neu[k].actv);
                if(first_run)
                    dweights[i-1][k][j]=0;
                dweights[i-1][k][j]+=lay[i-1].neu[k].dw[j];
                //printf("%f,\n",lay[i-1].neu[k].dw[j]);

                if(i>1)
                {
                    lay[i-1].neu[k].dactv = lay[i-1].neu[k].out_weights[j] * lay[i].neu[j].dz;
                }
            }

            lay[i].neu[j].dbias = lay[i].neu[j].dbias + lay[i].neu[j].dz;
            if(first_run)
                dbias[i][j]=0;
            dbias[i][j]+=lay[i].neu[j].dbias;
        }
    }
}

// Test the trained network
void test_nn(void) 
{
    // Get Training Examples #Modify
    get_inputs(input_test, 0);

    // Get Output Labels #Modify
    get_desired_outputs(desired_outputs_test, 0);

    for (int i = 0; i < num_testSamples_ex; i++){
        feed_input(i, input_test);
        forward_prop(1);
    }

}

// TODO: Add different Activation functions
//void activation_functions()

int dinit(void)
{
    // Free up all the structures

    // Free input and output
    free(input);
    free(input_test);
    free(desired_outputs);
    free(desired_outputs_test);


    return SUCCESS_DINIT;
}