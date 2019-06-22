#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"
#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

/*
 * Compute the force body j exercises on body i.
 * The acceleration is returned in *ax and *ay;
 */
static void compute(body *bodies, int i, int j,
                    long double *ax, long double *ay) {
    long double aij_x, aij_y, dx, dy, r3;
    dx = bodies[j].x - bodies[i].x;
    dy = bodies[j].y - bodies[i].y;
    r3 = powl(sqrtl(dx * dx + dy * dy), 3);
    aij_x = dx / r3;
    aij_y = dy / r3;
    *ax = aij_x * bodies[j].mass;
    *ay = aij_y * bodies[j].mass;
}

/*
 * Update (x,y) and (vx,vy) of body b from its
 * current position, velocity and acceleration.
 */
static void update(body *b, double deltaT,
                   long double ax, long double ay) {
    long double dvx, dvy;
    dvx = ax * (G * deltaT);
    dvy = ay * (G * deltaT);
    b->x += (b->vx + dvx / 2) * deltaT;
    b->y += (b->vy + dvy / 2) * deltaT;
    b->vx += dvx;
    b->vy += dvy;
}

void compute_single(struct TaskInput *TI) {
    const bool debug = TI->debug;
    const long double deltaT = TI->deltaT;
    const int nSteps = TI->nSteps;
    const int imageStep = TI->imageStep;
    const int nBodies = TI->nBodies;
    body *bodies = TI->bodies;

    long double accelsX[nBodies];
    long double accelsY[nBodies];

    for (int step = 0; step < nSteps; ++step) {
        if (imageStep > 0 && step % imageStep == 0)
            saveImage(step / imageStep, bodies, nBodies);
        if (debug)
            printf("%d\r", step);

        for (int i = 0; i < nBodies; ++i) {
            accelsX[i] = accelsY[i] = 0;
            for (int j = 0; j < nBodies; ++j) {
                if (i == j)
                    continue;
                long double ax, ay;
                compute(bodies, i, j, &ax, &ay);
                accelsX[i] += ax;
                accelsY[i] += ay;
            }
        }

        for (int i = 0; i < nBodies; i++) {
            update(&bodies[i], deltaT, accelsX[i], accelsY[i]);
        }
    }

    if (imageStep > 0 && nSteps % imageStep == 0)
        saveImage(nSteps / imageStep, bodies, nBodies);
    if (debug)
        printf("\n");

}
