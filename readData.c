#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "readData.h"

int toLowEndian(int number){
    unsigned char c1, c2, c3, c4;

    c1 = number & 255;
    c2 = (number >> 8) & 255;
    c3 = (number >> 16) & 255;
    c4 = (number>> 24) & 255;

    return  ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void loadLabels(int rank, int proc, unsigned char **data, int isTrain){
    /* Load the labels; desired outputs */
    
    /** Open the file **/

    FILE *trainData;
    
    if (isTrain == 1){
        trainData = fopen("data/train-labels.idx1-ubyte", "rb"); 
    } else{
        trainData = fopen("data/t10k-labels.idx1-ubyte", "rb");
    }
     
    if (trainData == NULL){
        perror("Data couldn't be loaded");
    }

    /* Reading and verifying Metadata */ 
    int magic_number, items;

    
    int x1 = fread((char *) &magic_number, 1, 4, trainData);
    int x2 = fread((char *) &items, 1, 4, trainData);
    if (x1 != 4 || x2 != 4){
        perror("Data not correctly loaded\n");
    }
    int n = 1;
    int littleEnd = 0;
    if (*((char * ) &n) ==  1){
        littleEnd = 1;
    }
    if (littleEnd == 1){
        magic_number = toLowEndian(magic_number);
        items = toLowEndian(items);
    }

    //printf("Parameters %d, %d \n", magic_number, items);
    /* Verification of proper data loading */
    assert(magic_number == MAGIC_LABEL);

    if (isTrain == 1){
        assert(items == NUM_ITEMS_TRAIN);
    } else{
        assert(items == NUM_ITEMS_TEST);
    }
    

    /** Reafing labels **/ 

    /** Chunk size and offset for the thread **/
    int offset, localItems;
    if (isTrain == 1){
        offset = (int) (NUM_ITEMS_TRAIN/proc)*(rank-1);
        localItems = (int) (NUM_ITEMS_TRAIN/proc);
    } else{
        offset = (int) (NUM_ITEMS_TEST/proc)*(rank-1);
        localItems = (int) (NUM_ITEMS_TEST/proc);      
    }

    /** Chosen to train on a small dataset **/
    //localItems = 2000;
    /****************************************/
    fseek(trainData, offset, SEEK_CUR);
    
    
    /** Reading and copying data to the output **/
    unsigned char index;
    int intIndex = 0;
    for (int i =0; i < localItems; i++){
        int read = fread((void *) &index, 1, 1, trainData);
        intIndex = (int) index;
        *((unsigned char *) data[i] + intIndex) = (unsigned char) 1;
        if (read !=  1){
            perror("An error has occured while loading labels \n");
        }
    }
}



void loadData(int rank, int proc, unsigned char **data, int isTrain){
    /* Load training data, the input for the neural network */ 
    FILE *trainData;

    if (isTrain == 1){
        trainData = fopen("data/train-images.idx3-ubyte", "rb"); 
    } else{
        trainData = fopen("data/t10k-images.idx3-ubyte", "rb");
    }
    if (trainData == NULL){
        perror("Data couldn't be loaded");
    }

    // char buffer[4]; 
    // for (int i = 0; i < 1024; i = i + 4){
    //     fread(buffer, 1, 4, trainData);
    //     printf("Num %x: %x\n", i, *(buffer));
    // }
    int magic_number, items, rows, columns;

    /* Reading data */ 
    int x1 = fread((char *) &magic_number, 1, 4, trainData);
    int x2 = fread((char *) &items, 1, 4, trainData);
    int x3 = fread((char *) &rows, 1, 4, trainData);
    int x4 = fread((char *) &columns, 1, 4, trainData);

    //printf("Read bytes %d %d %d %d\n", x1, x2, x3, x4);
    /* Dealing with endianess */
    int n = 1;
    int littleEnd = 0;
    if (*((char * ) &n) ==  1){
        littleEnd = 1;
    }
    if (littleEnd == 1){
        magic_number = toLowEndian(magic_number);
        items = toLowEndian(items);
        rows = toLowEndian(rows);
        columns = toLowEndian(columns);
    }
    
    /* Verification of proper data loading */
    if (isTrain == 1){
        assert(items == NUM_ITEMS_TRAIN);
    } else{
        assert(items == NUM_ITEMS_TEST);
    }
    assert(magic_number == MAGIC_IMAGE);
    assert(rows == NUM_ROWS);
    assert(columns == NUM_COL);

    //printf("Cool\n");
    int offset, localItems;
    if (isTrain == 1){
        offset = (int) (NUM_ITEMS_TRAIN/proc)*NUM_COL*NUM_ROWS*(rank-1);
        localItems = (int) (NUM_ITEMS_TRAIN/proc);
    } else{
        offset = (int) (NUM_ITEMS_TEST/proc)*NUM_COL*NUM_ROWS*(rank-1);
        localItems = (int) (NUM_ITEMS_TEST/proc);
    }
    /* Used to train on a small dataset To remove later */
    //localItems = 2000;
    /***************************/
    
    fseek(trainData, offset, SEEK_CUR);

    for (int i =0; i < localItems; i++){
        int read = fread((void *) data[i], 1, NUM_COL*NUM_ROWS, trainData);
        if (read !=  NUM_COL*NUM_ROWS){
            perror("Data couldn't be loaded correctly \n");
        }
    }
}


// void main(){
//     // int testSize = 28*28*2;
//     int rank, P, locItems, totalItems;
//     rank = 1;
//     P = 3000;
//     totalItems = 60000;
//     locItems = totalItems/P;

//     printf("local Items %d \n ", locItems);
//     unsigned char **data = (unsigned char **) malloc(locItems*sizeof(unsigned char *));
//     for (int i = 0; i < locItems; i++){
//         data[i] = (unsigned char *) malloc(10*sizeof(unsigned char));
//         memset(data[i],0, 10*sizeof(unsigned char));
//     }
//     printf("Wtf");
//     loadLabels(1, 2000, data, 0);
//     for (int i = 0; i < locItems; i++){
//         printf("item %d \t: ", i);
//         for (int j = 0; j < 10; j++){
//             printf("%d", *((unsigned char*) data[i] + j));
//         }
//         printf("\n");
//     }
// }