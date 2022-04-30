
#define MAGIC_LABEL 2049
#define MAGIC_IMAGE 2051
#define NUM_ITEMS_TRAIN 60000
#define NUM_ITEMS_TEST 10000
#define NUM_ROWS 28
#define NUM_COL 28


void loadLabels(int rank, int proc, char **data, int isTrain);
void loadData(int rank, int proc, char **data, int isTrain);
