#include <stdbool.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"
#include <assert.h>


#define PARALLEL_SORT_THRESHOLD 10000

/* Sort the bodies by the x axis using parallel odd even sort. Used for large scale problems. */
inline static void parallel_fixedpoint_odd_even_sort(body *bodies,
                                                     const int start_index, const int end_index) {
    bool no_fixed_point = true;
    int iteration = 1;
    while (no_fixed_point) {
        int odd_even = iteration % 2;
        no_fixed_point = false;
#pragma omp parallel for reduction(|:no_fixed_point)
        for (int i = start_index + odd_even; i < end_index; i += 2) {
            body curr = bodies[i];
            body next = bodies[i + 1];
            if (curr.x > next.x) {
                bodies[i] = next;
                bodies[i + 1] = curr;
                no_fixed_point |= true;
            }
        }
        iteration++;
    }
}

/* Sort the bodies by the x axis using single threaded insertion sort. Used for small scale problems. */
inline static void insertion_sort(body *bodies, const int start, const int end) {
    body tmp;
    for (int i = start + 1; i < end; i++) {
        for (int j = i; j > 0; j--) {
            if (bodies[j - 1].x < bodies[j].x) {
                break;
            }
            tmp = bodies[j];
            bodies[j] = bodies[j - 1];
            bodies[j - 1] = tmp;
        }
    }
}

/* Sort the bodies.
 * For hard problems(more than 10000 bodies) use an algorithm that can run in parallel with openmp.
 * Only use sorting algorithms that make use of the fact that most commonly only neighbours switch places
 * and terminate in O(n) if the array is already sorted.
 * After sorting copy the x/y positions into the xs/ys buffer.*/
inline static void sort_bodies(body *bodies, const int n_bodies, long double *xs, long double *ys) {
    if (n_bodies > PARALLEL_SORT_THRESHOLD) {
        parallel_fixedpoint_odd_even_sort(bodies, 0, (n_bodies - 1));
    } else {
        insertion_sort(bodies, 0, n_bodies);
    }

    if (xs != NULL && ys != NULL) {
#pragma omp parallel for
        for (int i = 0; i < n_bodies; i++) {
            xs[i] = bodies[i].x;
            ys[i] = bodies[i].y;
        }
    }
}

/* Compute the relative acceleration a_ij. */
inline static void compute_acceleration_opt(const body *bodies_i, const body *bodies_j, const int i, const int j,
                                            long double *ax, long double *ay) {
    long double delta_x = bodies_j[j].x - bodies_i[i].x;
    long double delta_y = bodies_j[j].y - bodies_i[i].y;
    const long double eucl = sqrtl(delta_x * delta_x + delta_y * delta_y);
    // writing the statement explicit performs way faster
    long double r3 = eucl * eucl * eucl;
    *ax = (delta_x / r3) * bodies_j[j].mass;
    *ay = (delta_y / r3) * bodies_j[j].mass;
}

/* Update the postion of an body using the the accelerations axs, ays and the time frame deltaT */
inline static void update_position_opt(body *b, long double *xs, long double *ys, const long double g_delta,
                                       const long double deltaT, long double ax, long double ay) {
    long double delta_vx = ax * g_delta;
    long double delta_vy = ay * g_delta;
    *xs += (b->vx + delta_vx / 2) * deltaT;
    *ys += (b->vy + delta_vy / 2) * deltaT;
    b->vx += delta_vx;
    b->vy += delta_vy;
}

/* Initialize a buffer for the accelerations to save intermediate results of the computations */
inline static void init_accelerations(int size, long double **axs, long double **ays) {
    *axs = malloc(sizeof(long double) * size);
    *ays = malloc(sizeof(long double) * size);
    if (*(axs) == NULL || *(ays) == NULL) {
        fprintf(stderr, "Error could not allocate memory for accelerations.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

/* Free the acceleration buffers */
inline static void free_accelerations(long double *axs, long double *ays) {
    free(axs);
    free(ays);
}

/* Gather the x(xs) and y(ys) positions of bodies in all processes. */
inline static void allgatherv_xy(long double *xs, long double *ys, int *sendcounts, int *displs, int self) {
    MPI_Allgatherv(MPI_IN_PLACE, sendcounts[self], MPI_LONG_DOUBLE, ys, sendcounts, displs,
                   MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, sendcounts[self], MPI_LONG_DOUBLE, xs, sendcounts, displs,
                   MPI_LONG_DOUBLE, MPI_COMM_WORLD);
}

/* Distribute the velocity to all bodies at the end of the simulation. As for non surrogate methods each process
 * handles the same bodies of the course of the simulation, these methods only require an exchange of velocity at the
 * end of the simulation in order to correctly compute the momentum */
inline static void finalize_velocity(body *bodies, const int n_to_simulate, const int offset, long double *xs,
                                     long double *ys, int *sendcounts, int *displs, const int self,
                                     const int n_bodies) {
#pragma omp parallel for
    for (int i = offset; i < n_to_simulate; i++) {
        xs[i] = bodies[i].vx;
        ys[i] = bodies[i].vy;
    }

    allgatherv_xy(xs, ys, sendcounts, displs, self);

#pragma omp parallel for
    for (int i = 0; i < n_bodies; i++) {
        bodies[i].vx = xs[i];
        bodies[i].vy = ys[i];
    }
}

/* Debug output about the distribution of bodies */
inline static void dbg_print_start(const bool dbg, const int self, const int offset, const int n_to_simulate) {
    if (dbg) {
        printf("Process %d: Simulating body %d to %d.\n", self, offset, n_to_simulate);
        printf("Process %d: Starting simulation.\n", self);
    }
}

/* Print the image at process 0 if required */
inline static void print_image(const int iterations_per_img, int iteration, const int self, body *bodies,
                               const int n_bodies) {
    if (iterations_per_img > 0 && iteration % iterations_per_img == 0 && self == 0)
        saveImage(iteration / iterations_per_img, bodies, n_bodies);
}

inline static void dbg_print_end(const int dbg, const int self) {
    if (dbg) {
        printf("Process %d: Simulation finished.\n", self);
    }
}

/* Initialize buffers that save the x,y positions with the values of the corresponding bodies */
inline static void initialize_position_buffer(const int start_index_to_simulate, const int end_index_to_simulate,
                                              long double *xs, long double *ys, body *bodies) {
#pragma omp parallel for
    for (int i = start_index_to_simulate; i < end_index_to_simulate; i++) {
        xs[i] = bodies[i].x;
        ys[i] = bodies[i].y;
    }
}

/* Propagate the x,y positions saved in xs,ys to all other processes.
 * In case the method with surrogates and sorting is used the other informations is also propagated to the other
 * processes(in this case xs, ys are not required but PPP_BODY is required) and the bodies are sorted after receiving
 * all data.
 */
inline static void propagate_positions_opt(long double *xs, long double *ys, body *bodies, const int self,
                                           const int n_bodies, int *sendcounts, int *displs, const bool sort,
                                           MPI_Datatype PPP_BODY) {
    if (sort) {
        // if we sort we also need to distribute the velocity because we can not assume that a body will always be assigned
        // to the process that also keeps track of the velocity
        // Then sending all instead of seperate is faster
        MPI_Allgatherv(MPI_IN_PLACE, 0, PPP_BODY, bodies, sendcounts, displs, PPP_BODY, MPI_COMM_WORLD);
        sort_bodies(bodies, n_bodies, xs, ys);
    } else {
        allgatherv_xy(xs, ys, sendcounts, displs, self);
#pragma omp parallel for
        for (int i = 0; i < n_bodies; i++) {
            bodies[i].x = xs[i];
            bodies[i].y = ys[i];
        }
    }
}

/*
 * Print debug output, free buffers, print the image if required and propagate the velocity if required.
 * If the velocity should not be propagated xs and ys can be NULL.
 */
inline static void finalize_computation(const bool dbg, const int self, body *bodies, const int end_index_to_simulate,
                                        const int start_index_to_simulate, long double *xs, long double *ys,
                                        long double *a_xs, long double *a_ys, int *sendcounts, int *displs,
                                        const int n_bodies, const int image_step, const int steps) {
    dbg_print_end(dbg, self);
    if (xs != NULL && ys != NULL) {
        finalize_velocity(bodies, end_index_to_simulate, start_index_to_simulate, xs, ys, sendcounts, displs, self,
                          n_bodies);
        free(xs);
        free(ys);
    }
    free_accelerations(a_xs, a_ys);
    print_image(image_step, steps, self, bodies, n_bodies);
}

/*
 * Perform the simulation without using local or global newton 3.
 * In each iteration the acceleration of the bodies assigned to the process executing this method are computed,
 * the new positions are computed and buffered and then distributed and written to bodies.
 */
inline static void perform_simulation_naive(int n_steps, int imgStep, body *bodies, int nBodies, bool debug,
                                            long double delta_t, int *sendcounts, int *displs, int self) {
    const int iterations = n_steps;
    const int iterations_per_img = imgStep;
    const bool dbg = debug;
    const int n_bodies = nBodies;
    const int start_index_to_simulate = displs[self];
    const int end_index_to_simulate = displs[self] + sendcounts[self];
    const long double g_delta = (G * delta_t);
    long double *a_xs;
    long double *a_ys;
    long double *xs = malloc(sizeof(long double) * n_bodies);
    long double *ys = malloc(sizeof(long double) * n_bodies);

    init_accelerations(sendcounts[self], &a_xs, &a_ys);
    initialize_position_buffer(start_index_to_simulate, end_index_to_simulate, xs, ys, bodies);
    dbg_print_start(dbg, self, start_index_to_simulate, end_index_to_simulate);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        print_image(iterations_per_img, iteration, self, bodies, n_bodies);

#pragma omp parallel for
        for (int i = start_index_to_simulate; i < end_index_to_simulate; ++i) {
            const int idx = i - start_index_to_simulate;
            a_xs[idx] = a_ys[idx] = 0;
            for (int j = 0; j < n_bodies; ++j) {
                if (i == j)
                    continue;
                long double ax, ay;
                compute_acceleration_opt(bodies, bodies, i, j, &ax, &ay);
                a_xs[idx] += ax;
                a_ys[idx] += ay;
            }
        }

#pragma omp parallel for
        for (int i = start_index_to_simulate; i < end_index_to_simulate; i++) {
            update_position_opt(&bodies[i], &xs[i], &ys[i], g_delta, delta_t, a_xs[i - start_index_to_simulate],
                                a_ys[i - start_index_to_simulate]);
        }

        propagate_positions_opt(xs, ys, bodies, self, n_bodies, sendcounts, displs, false, NULL);
    }
    finalize_computation(dbg, self, bodies, end_index_to_simulate, start_index_to_simulate, xs, ys, a_xs, a_ys,
                         sendcounts, displs, n_bodies, iterations_per_img, iterations);
}

/* Initialize acceleration buffers with 0s. This is required for instances where buffers are not traversed
 * in a linear fashion.*/
inline static void c_initialize_accelerations(long double *a_xs, long double *a_ys, const int size) {
    // Actually works faster than filling using parallel loop
    memset(a_xs, 0, sizeof(long double) * size);
    memset(a_ys, 0, sizeof(long double) * size);
}

/* Compute accelerations between local bodies and non local bodies. For reusability both can be stored in
 * separate arrays. Start and end are the loop counter values of the non local iteration, whereas i is the
 * index of the local iteration.*/
inline static void compute_inner_loop(body *bodies_i, body *bodies_j, const int i, const int start, const int end,
                                      long double *a_xs, long double *a_ys, const int idx_i) {
    long double xbuff_i = 0;
    long double ybuff_i = 0;
#pragma omp parallel for reduction(+:xbuff_i) reduction(+:ybuff_i)
    for (int j = start; j < end; ++j) {
        long double ax, ay;
        compute_acceleration_opt(bodies_i, bodies_j, i, j, &ax, &ay);
        xbuff_i += ax;
        ybuff_i += ay;
    }
    a_xs[idx_i] += xbuff_i;
    a_ys[idx_i] += ybuff_i;
}

/* Perform simulation using Newton 3 for local-local computation.
 * The principle is similar to the previous simulation however the inner loop is split in 3 parts, with 2 loops
 * being the lower and bigger indices that are non local that are computed without newton. The middle loop uses Newton 3
 * and only does half the iterations.
 * Can slightly deviate from the results of the single threaded implementation due to different order of computation leading to
 * different rounding errors.
 */
inline static void perform_simulation_local(int n_steps, int imgStep, body *bodies, int nBodies, bool debug,
                                            long double delta_t, int *sendcounts, int *displs, int self) {
    const int iterations = n_steps;
    const int iterations_per_img = imgStep;
    const bool dbg = debug;
    const int n_bodies = nBodies;
    const int start_index_to_simulate = displs[self];
    const int end_index_to_simulate = displs[self] + sendcounts[self];
    const long double g_delta = (G * delta_t);
    long double *a_xs;
    long double *a_ys;
    long double *xs = malloc(sizeof(long double) * n_bodies);
    long double *ys = malloc(sizeof(long double) * n_bodies);

    init_accelerations(sendcounts[self], &a_xs, &a_ys);
    initialize_position_buffer(start_index_to_simulate, end_index_to_simulate, xs, ys, bodies);
    dbg_print_start(dbg, self, start_index_to_simulate, end_index_to_simulate);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        print_image(iterations_per_img, iteration, self, bodies, n_bodies);

        c_initialize_accelerations(a_xs, a_ys, sendcounts[self]);

        for (int i = start_index_to_simulate; i < end_index_to_simulate; ++i) {
            // As only buffer space required to calculate local accelerations is allocated
            // the indices need to shifted
            const int idx_i = i - start_index_to_simulate;

            // Splitting up the loops actually performs better in the parallel case than handling everything in one
            // single loop
            compute_inner_loop(bodies, bodies, i, 0, start_index_to_simulate, a_xs, a_ys, idx_i);

            long double xbuff_i = 0;
            long double ybuff_i = 0;
#pragma omp parallel for reduction(+:xbuff_i) reduction(+:ybuff_i)
            for (int j = i; j < end_index_to_simulate - 1; ++j) {
                long double ax, ay;
                const int idx_j = j + 1 - start_index_to_simulate;
                compute_acceleration_opt(bodies, bodies, i, j + 1, &ax, &ay);
                xbuff_i += ax;
                ybuff_i += ay;
                a_xs[idx_j] += -(bodies[i].mass / bodies[j + 1].mass) * ax;
                a_ys[idx_j] += -(bodies[i].mass / bodies[j + 1].mass) * ay;
            }
            a_xs[idx_i] += xbuff_i;
            a_ys[idx_i] += ybuff_i;

            compute_inner_loop(bodies, bodies, i, end_index_to_simulate, n_bodies, a_xs, a_ys, idx_i);
        }

#pragma omp parallel for
        for (int i = start_index_to_simulate; i < end_index_to_simulate; i++) {
            update_position_opt(&bodies[i], &xs[i], &ys[i], g_delta, delta_t, a_xs[i - start_index_to_simulate],
                                a_ys[i - start_index_to_simulate]);
        }
        propagate_positions_opt(xs, ys, bodies, self, n_bodies, sendcounts, displs, false, NULL);
    }

    finalize_computation(dbg, self, bodies, end_index_to_simulate, start_index_to_simulate, xs, ys, a_xs, a_ys,
                         sendcounts, displs, n_bodies, iterations_per_img, iterations);
}


/* Perform simulation using global Newton 3. For each element in bodies only the acceleration for bodies with higher
 * indices is is directly computed(The other way round is computed with Newton).
 * The results are the summed up using MPI reduction. */
inline static void perform_simulation_global(int n_steps, int imgStep, body *bodies, int nBodies, bool debug,
                                             long double delta_t, int *sendcounts, int *displs, int self) {
    const int iterations = n_steps;
    const int iterations_per_img = imgStep;
    const bool dbg = debug;
    const int n_bodies = nBodies;
    const int start_index_to_simulate = displs[self];
    const int end_index_to_simulate = displs[self] + sendcounts[self];
    const long double g_delta = (G * delta_t);
    long double *a_xs;
    long double *a_ys;
    long double *xs = malloc(sizeof(long double) * n_bodies);
    long double *ys = malloc(sizeof(long double) * n_bodies);

    init_accelerations(n_bodies, &a_xs, &a_ys);
    initialize_position_buffer(start_index_to_simulate, end_index_to_simulate, xs, ys, bodies);
    dbg_print_start(dbg, self, start_index_to_simulate, end_index_to_simulate);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        print_image(iterations_per_img, iteration, self, bodies, n_bodies);

        c_initialize_accelerations(a_xs, a_ys, n_bodies);

        for (int i = start_index_to_simulate; i < end_index_to_simulate; ++i) {
            long double xbuff_i = 0;
            long double ybuff_i = 0;

#pragma omp parallel for reduction(+:xbuff_i) reduction(+:ybuff_i)
            for (int j = i + 1; j < n_bodies; ++j) {
                long double ax, ay;
                compute_acceleration_opt(bodies, bodies, i, j, &ax, &ay);
                xbuff_i += ax;
                ybuff_i += ay;
                a_xs[j] += -(bodies[i].mass / bodies[j].mass) * ax;
                a_ys[j] += -(bodies[i].mass / bodies[j].mass) * ay;
            }

            a_xs[i] += xbuff_i;
            a_ys[i] += ybuff_i;
        }

        MPI_Allreduce(MPI_IN_PLACE, a_xs, n_bodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, a_ys, n_bodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for
        for (int i = start_index_to_simulate; i < end_index_to_simulate; i++) {
            update_position_opt(&bodies[i], &xs[i], &ys[i], g_delta, delta_t, a_xs[i], a_ys[i]);
        }

        propagate_positions_opt(xs, ys, bodies, self, n_bodies, sendcounts, displs, false, NULL);
    }

    finalize_computation(dbg, self, bodies, end_index_to_simulate, start_index_to_simulate, xs, ys, a_xs, a_ys,
                         sendcounts, displs, n_bodies, iterations_per_img, iterations);
}

/* Each process computes a surrogate body that represents its own bodies as a new body that has the sum of the mass
 * and the mass wieghted center point. The bodies are then distributed to/gathered from all processes
 * and saved in surrogates. */
inline static void compute_surrogates(body *bodies, body *surrogates, const int *sendcounts, const int *displs,
                   const int *displs_surrogates, const int *sendcounts_surrogates, int self, MPI_Datatype PPP_SURROGATE) {
    int associated_to_body = sendcounts[self];
    long double sum_x = 0;
    long double sum_y = 0;
    long double sum_mass = 0;
    // Probably very prone to overflow
#pragma omp parallel for reduction(+:sum_x) reduction(+:sum_y) reduction(+:sum_mass)
    for (int j = displs[self]; j < displs[self] + associated_to_body; j++) {
        sum_x += bodies[j].x * bodies[j].mass;
        sum_y += bodies[j].y * bodies[j].mass;
        sum_mass += bodies[j].mass;
    }
    if (associated_to_body > 0) {
        // Formula in task sheet incorrect
        // to weight by mass it needs to be divided by sum of mass not number of bodies
        surrogates[self].x = sum_x / sum_mass;
        surrogates[self].y = sum_y / sum_mass;
        surrogates[self].mass = sum_mass;
    }
    MPI_Allgatherv(MPI_IN_PLACE, 1, PPP_SURROGATE, surrogates, sendcounts_surrogates, displs_surrogates,
                   PPP_SURROGATE, MPI_COMM_WORLD);
}

/* Perform simulation using surrogates and local Newton.
 * Local newton is similar to the other implementation, however for non local computations the surrogates of the
 * other processes are used. Additionally this method distributes all information about the bodies after each iteration
 * because it can not be assumed that the bodies will always be assigned to the process that stores the information
 * about the velocity */
inline static void perform_simulation_surrogate(int n_steps, int imgStep, body *bodies, int nBodies, bool debug,
                                                long double delta_t, int *sendcounts, int *displs, int self, int np,
                                                MPI_Datatype PPP_SURROGATE) {
    const int iterations = n_steps;
    const int iterations_per_img = imgStep;
    const bool dbg = debug;
    const int n_bodies = nBodies;
    const int start_index_to_simulate = displs[self];
    const int end_index_to_simulate = displs[self] + sendcounts[self];
    const long double g_delta = (G * delta_t);
    long double *a_xs;
    long double *a_ys;
    body surrogates[np];
    int max_np = np > n_bodies ? n_bodies : np;
    int sendcounts_surrogates[np];
    int displs_surrogates[np];

    for (int i = 0; i < np; i++) {
        if (i < n_bodies) {
            displs_surrogates[i] = i;
            sendcounts_surrogates[i] = 1;
        } else {
            displs_surrogates[i] = n_bodies;
            sendcounts_surrogates[i] = 0;
        }
    }

    init_accelerations(sendcounts[self], &a_xs, &a_ys);
    dbg_print_start(dbg, self, start_index_to_simulate, end_index_to_simulate);

    sort_bodies(bodies, n_bodies, NULL, NULL);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        print_image(iterations_per_img, iteration, self, bodies, n_bodies);

        compute_surrogates(bodies, surrogates, sendcounts, displs, displs_surrogates, sendcounts_surrogates,
                           self, PPP_SURROGATE);
        c_initialize_accelerations(a_xs, a_ys, sendcounts[self]);

        for (int i = start_index_to_simulate; i < end_index_to_simulate; ++i) {
            const int idx_i = i - start_index_to_simulate;

            // Compute with surrogates
            compute_inner_loop(bodies, surrogates, i, 0, self < max_np ? self : 0, a_xs, a_ys, idx_i);

            long double xbuff_i = 0;
            long double ybuff_i = 0;
#pragma omp parallel for reduction(+:xbuff_i) reduction(+:ybuff_i)
            for (int j = i; j < end_index_to_simulate - 1; ++j) {
                long double ax, ay;
                const int idx_j = j + 1 - start_index_to_simulate;
                compute_acceleration_opt(bodies, bodies, i, j + 1, &ax, &ay);
                xbuff_i += ax;
                ybuff_i += ay;
                a_xs[idx_j] += -(bodies[i].mass / bodies[j + 1].mass) * ax;
                a_ys[idx_j] += -(bodies[i].mass / bodies[j + 1].mass) * ay;
            }
            a_xs[idx_i] += xbuff_i;
            a_ys[idx_i] += ybuff_i;

            compute_inner_loop(bodies, surrogates, i, self + 1 < max_np ? self + 1 : max_np, max_np, a_xs, a_ys, idx_i);
        }
#pragma omp parallel for
        for (int i = start_index_to_simulate; i < end_index_to_simulate; i++) {
            update_position_opt(&bodies[i], &bodies[i].x, &bodies[i].y, g_delta, delta_t,
                                a_xs[i - start_index_to_simulate],
                                a_ys[i - start_index_to_simulate]);
        }
        propagate_positions_opt(NULL, NULL, bodies, self, n_bodies, sendcounts, displs, true, PPP_SURROGATE);
    }

    finalize_computation(dbg, self, bodies, end_index_to_simulate, start_index_to_simulate, NULL, NULL, a_xs, a_ys,
                         sendcounts, displs, n_bodies, iterations_per_img, iterations);
}

/* Allocate displacements and send counts for each process */
inline static void alloc_sendcnts_displ(int np, int problem_size, int **out_sendcts, int **out_displs) {
    int *sendcnts = calloc(np, sizeof(int));
    int *displs = calloc(np, sizeof(int));
    if (sendcnts == NULL || displs == NULL) {
        fprintf(stderr, "Error. Could not allocate memory for sendcount and/or displs.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (np <= problem_size) {

        int bodies_per_process = (problem_size / np);
        for (int i = 0; i < np; i++) {
            sendcnts[i] = bodies_per_process + (i < problem_size % np ? 1 : 0);
            displs[i] = i * bodies_per_process + (i > problem_size % np ? problem_size % np : i);
        }

    } else {
        for (int i = 0; i < np; i++) {
            if (i < problem_size) {
                sendcnts[i] = 1;
                displs[i] = i;
            } else {
                sendcnts[i] = 0;
                displs[i] = problem_size;
            }
        }
    }
    *out_displs = displs;
    *out_sendcts = sendcnts;
}

void compute_parallel(struct TaskInput *TI) {
    int np, self;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single

            printf("Number of OMP threads in each MPI process: %d\n",
                   omp_get_num_threads());
        }
    }

    const bool debug = TI->debug;
    const int n_bodies = TI->nBodies;
    body *all_bodies = TI->bodies;

    if (debug && self == 0) {
        printf("Total number of bodies: %d\n", n_bodies);
    }

    int *sendcnts, *displs;
    alloc_sendcnts_displ(np, n_bodies, &sendcnts, &displs);
    if (debug) {
        printf("Process #%d starts. Performs simulation for %d bodies. Displacement: %d.\n", self, sendcnts[self],
               displs[self]);
    }
    if (TI->newton3) {
        perform_simulation_global(TI->nSteps, TI->imageStep, all_bodies, n_bodies, debug, TI->deltaT,
                                  sendcnts, displs, self);
    } else if (TI->newton3local) {
        perform_simulation_local(TI->nSteps, TI->imageStep, all_bodies, n_bodies, debug, TI->deltaT,
                                 sendcnts, displs, self);
    } else if (TI->approxSurrogate) {
        MPI_Datatype PPP_SURROGATE;
        MPI_Type_contiguous(5, MPI_LONG_DOUBLE, &PPP_SURROGATE);
        MPI_Type_commit(&PPP_SURROGATE);
        // implementation with big surrogate bodies for bodies in other processes
        // sorting can change the values of the momentum so comparison has to be handled carefully
        perform_simulation_surrogate(TI->nSteps, TI->imageStep, all_bodies, n_bodies, debug, TI->deltaT,
                                     sendcnts, displs, self, np, PPP_SURROGATE);
    } else {
        perform_simulation_naive(TI->nSteps, TI->imageStep, all_bodies, n_bodies, debug, TI->deltaT,
                sendcnts, displs, self);
    }
    free(sendcnts);
    free(displs);
}
