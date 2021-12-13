#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

struct mpi_counter_t {
    MPI_Win win;
    int  hostrank ;
    int  myval;
    int *data;
    int rank, size;
};

struct mpi_counter_t *create_counter(int hostrank) {
    struct mpi_counter_t *count;

    count = (struct mpi_counter_t *)malloc(sizeof(struct mpi_counter_t));
    count->hostrank = hostrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &(count->rank));
    MPI_Comm_size(MPI_COMM_WORLD, &(count->size));

    if (count->rank == hostrank) {
        MPI_Alloc_mem(count->size * sizeof(int), MPI_INFO_NULL, &(count->data));
        for (int i=0; i<count->size; i++) count->data[i] = 0;
        MPI_Win_create(count->data, count->size * sizeof(int), sizeof(int),
                       MPI_INFO_NULL, MPI_COMM_WORLD, &(count->win));
    } else {
        count->data = NULL;
        MPI_Win_create(count->data, 0, 1,
                       MPI_INFO_NULL, MPI_COMM_WORLD, &(count->win));
    }
    count -> myval = 0;

    return count;
}

int increment_counter(struct mpi_counter_t *count, int increment) {
    int *vals = (int *)malloc( count->size * sizeof(int) );
    int val;

    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, count->hostrank, 0, count->win);

    for (int i=0; i<count->size; i++) {

        if (i == count->rank) {
            MPI_Accumulate(&increment, 1, MPI_INT, 0, i, 1, MPI_INT, MPI_SUM,
                           count->win);
        } else {
            MPI_Get(&vals[i], 1, MPI_INT, 0, i, 1, MPI_INT, count->win);
        }
    }

    MPI_Win_unlock(0, count->win);
    count->myval += increment;

    vals[count->rank] = count->myval;
    val = 0;
    for (int i=0; i<count->size; i++)
        val += vals[i];

    free(vals);
    return val;
}

void delete_counter(struct mpi_counter_t **count) {
    if ((*count)->rank == (*count)->hostrank) {
        MPI_Free_mem((*count)->data);
    }
    MPI_Win_free(&((*count)->win));
    free((*count));
    *count = NULL;

    return;
}

void print_counter(struct mpi_counter_t *count) {
    if (count->rank == count->hostrank) {
        for (int i=0; i<count->size; i++) {
            printf("%2d ", count->data[i]);
        }
        puts("");
    }
}

int test1() {
    struct mpi_counter_t *c;
    int rank;
    int result;

    c = create_counter(0);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    result = increment_counter(c, 1);
    printf("%d got counter %d\n", rank, result);

    MPI_Barrier(MPI_COMM_WORLD);
    print_counter(c);
    delete_counter(&c);
    return 0;
}


int test2() {
    const int WORKITEMS=50;

    struct mpi_counter_t *c;
    int rank;
    int result = 0;

    c = create_counter(0);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(rank);

    while (result < WORKITEMS) {
        result = increment_counter(c, 1);
        if (result <= WORKITEMS) {
            printf("%d working on item %d...\n", rank, result);
            sleep(rand() % 10);
        } else {
            printf("%d done\n", rank);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    print_counter(c);
    delete_counter(&c);
    return 0;
}

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    test1();
//    test2();

    MPI_Finalize();
}