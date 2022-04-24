#include <stdio.h>
#include <stdlib.h>
#include <stdlib.h>
#include <assert.h>

#include "readData.h"

int toLowEndian(int number){
    unsigned char c1, c2, c3, c4;

    c1 = number & 255;
    c2 = (number >> 8) & 255;
    c3 = (number >> 16) & 255;
    c4 = (number>> 24) & 255;

    return  ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;

}

void loadData(int rank, int proc, char * data){

    FILE *trainData = fopen("data\\Cptrain-images.idx3-ubyte", "rb");

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

    printf("Read bytes %d %d %d %d", x1, x2, x3, x4);
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
    assert(magic_number == MAGIC_NUMBER);
    assert(items == NUM_ITEMS);
    assert(rows == NUM_ROWS);
    assert(columns == NUM_COL);

    printf("Cool\n");

    int offset = (NUM_ITEMS/proc)*NUM_COL*NUM_ROWS*(rank-1);
    printf("%d \n", offset);
    fseek(trainData, offset, SEEK_CUR);
    int chunkSize = (NUM_ITEMS/proc)*NUM_COL*NUM_ROWS;
    printf("%d \n", chunkSize);
    int red = fread((void *) data, 1, chunkSize, trainData);
    printf("rrr %d \n", red);
}

void main(){
    int testSize = 28*28*2;
    char data[testSize];
    loadData(1, 30000, data);
    for (int i = 0; i < 128; i = i + 4){
        printf("in num 0x%0x, 0x%x \n", i, *((int *) ((char*)data + i)));
    }
}