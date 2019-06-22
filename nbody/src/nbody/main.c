#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include "mpi.h"
#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

/*
 * Kommentare (mit "# ...") in ".dat" Dateien ueberspringen.
 */
static void skipComments(FILE *f) {
    int n;
    int dummy; /* um "unused result" Warnungen zu unterdruecken */
    do {
        n=0;
        dummy = fscanf(f, " #%n", &n);
        if (n > 0) {
            dummy += fscanf(f, "%*[^\n]"); 
            dummy += fscanf(f, "\n");
        }
    } while (n>0);
}

/*
 * Eine ".dat" Datei mit Beschreibungen der Koerper einlesen.
 * (Format siehe Uebungsblatt).
 *    f: Dateihandle, aus dem gelesen wird
 *    n: Output-Parameter fuer die Anzahl der gelesenen Koerper
 * Die Koerper werden in einem Array von body-Strukturen
 * zurueckgeliefert. Im Fehlerfall wird NULL zurueckgegeben.
 */
body* readBodies(FILE *f, int *n) {
    int i, conv;
    body *bodies;

    skipComments(f);
    if (fscanf(f, " %d", n) != 1)
        return NULL;
    bodies = (body *) malloc(sizeof(body) * *n);
    if (bodies == NULL)
        return NULL;

    for (i=0; i<*n; i++) {
        long double mass, x, y, vx, vy;
        skipComments(f);
        conv = fscanf(f, " %Lf %Lf %Lf %Lf %Lf", &mass, &x, &y, &vx, &vy);
        bodies[i].mass = mass;
        bodies[i].x    = x;
        bodies[i].y    = y;
        bodies[i].vx   = vx;
        bodies[i].vy   = vy;
        if (conv != 5) {
            free(bodies);
            return NULL;
        }
    }
    return bodies;
}

/*
 * Schreibe 'n' Koerper aus dem Array 'bodies' in die
 * durch das Dateihandle 'f' bezeichnete Datei im ".dat" Format.
 */
void writeBodies(FILE *f, const body *bodies, int n) {
    int i;
    fprintf(f, "%d\n", n);
    for (i=0; i<n; i++) {
        fprintf(f, "% 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg\n",
                (long double)bodies[i].mass,
                (long double)bodies[i].x, (long double)bodies[i].y,
                (long double)bodies[i].vx, (long double)bodies[i].vy);
    }
}

void usage(const char *progname) {
    fprintf(stderr,
            "USAGE: %s -i input.dat [-o output.dat] [-t step] [-n n_steps]\n"
            "  [-d] [-s] [-p] [-3] [-l] [-a] [-W pixels] [-H pixels]\n"
            "  [-w width] [-h width] [-A asteps]\n"
            "   step     time step in seconds\n"
            "   n_steps  number of time steps\n"
            "   pixels   image width/height\n"
            "   width    width of depicted space in meters\n"
            "   height   heigt of depicted space in meters\n"
            "   asteps   number of steps between two images\n"
            "   -p       parallel execution\n"
            "   -3       use Newton's third law globally\n"
            "   -l       use Newton's third law in local computations\n"
            "   -a       use approximation with big surrogate bodies\n"
            "   -d       show some debugging output\n",
            progname);
}

/*
 * Compute the momentum in the system of the bodies.
 * The momentum is returned in *px and *py.
 */
void totalMomentum(body *bodies, int nBodies, long double *px, long double *py)
{
    long double px_=0, py_=0;
    int i;

    for (i=0; i<nBodies; i++) {
	px_ += bodies[i].mass * bodies[i].vx;
	py_ += bodies[i].mass * bodies[i].vy;
    }
    *px = px_;
    *py = py_;
}

int imageWidth, imageHeight;
double width, height;
const char *imageFilePrefix;

void saveImage(int imageNum, const body *bodies, int nBodies)
{
    int i, x, y;
    uint8_t *img = (uint8_t *) malloc(sizeof(uint8_t) * imageWidth * imageHeight);
    char name[strlen(imageFilePrefix)+11];

    if (img == NULL)
	return;

    sprintf(name, "%s-%05d.pbm", imageFilePrefix, imageNum);
    for (i=0; i<imageWidth*imageHeight; i++)
	img[i] = 0;

    for (i=0; i<nBodies; i++) {
	x = imageWidth/2  + bodies[i].x*imageWidth/width;
	y = imageHeight/2 - bodies[i].y*imageHeight/height;

	if (x >= 0 && x < imageWidth && y >=0 && y < imageHeight) {
	    img[y*imageWidth + x] = 1;
	}
    }

    ppp_pnm_write(name, PNM_KIND_PBM, imageHeight, imageWidth, 1, img);
    free(img);
}

int main(int argc, char *argv[])
{
    FILE *f;
    char *filename, *outfilename;
    bool parallel;
    int retCode;
    
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }
    int self;
    MPI_Comm_rank(MPI_COMM_WORLD, &self);
    
    filename = NULL;
    outfilename = NULL;
    parallel = false;
    
    struct TaskInput TI;
    TI.nSteps = 0;
    TI.deltaT = 1;
    TI.newton3 = false;
    TI.newton3local = false;
    TI.approxSurrogate = false;
    TI.imageStep = 0;
    TI.debug = false;
    
    imageWidth = 100;
    imageHeight = 100;
    width = 1;
    height = 1;

    int option;
    while ((option = getopt(argc,argv,"i:o:t:n:A:w:h:W:H:ps3lad")) != -1) {
        switch(option) {
        case 'i': filename    = strdup(optarg); break;
        case 'o': outfilename = strdup(optarg); break;
        case 't': TI.deltaT = atof(optarg); break;
	case 'n': TI.nSteps = atoi(optarg); break;
	case 'A': TI.imageStep = atoi(optarg); break;
	case 'w': width  = atof(optarg); break;
	case 'h': height = atof(optarg); break;
	case 'W': imageWidth  = atoi(optarg); break;
	case 'H': imageHeight = atoi(optarg); break;
	case 'p': parallel = true; break;
	case '3': TI.newton3 = true; break;
	case 'l': TI.newton3local = true; break;
        case 'a': TI.approxSurrogate = true; break;
	case 'd': TI.debug = true; break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    if (filename == NULL ||
        (TI.imageStep > 0 && outfilename == NULL)) {
        usage(argv[0]);
	MPI_Finalize();
	return 1;
    }

    imageFilePrefix = outfilename;

    retCode = 0;
    f = fopen(filename, "r");
    if (f != NULL) {
        int nBodies;
	body *bodies = readBodies(f, &nBodies);
        TI.nBodies = nBodies;
        TI.bodies = bodies;
	fclose(f);
	if (bodies != NULL) {
            long double pxStart, pyStart;
	    totalMomentum(bodies, nBodies, &pxStart, &pyStart);
	    double simTime = seconds();
	    if (parallel)
                compute_parallel(&TI);
	    else if (self == 0)
                compute_single(&TI);
	    simTime = seconds() - simTime;
            long double pxEnd, pyEnd;
	    totalMomentum(bodies, nBodies, &pxEnd, &pyEnd);
	    if (self == 0) {
		if (outfilename != NULL) {
		    f = fopen(outfilename, "w");
		    if (f != NULL) {
			writeBodies(f, bodies, nBodies);
			fclose(f);
		    } else {
			fprintf(stderr, "Could not open '%s' for writing.",
				outfilename);
		    }
		} else
		    writeBodies(stdout, bodies, nBodies);
		free(bodies);
		long double pAbs = hypotl(pxStart, pyStart);
		long double pxRel = (pxEnd-pxStart)/pAbs;
		long double pyRel = (pyEnd-pyStart)/pAbs;
                double interactionRate =
                    (double)nBodies*(double)(nBodies-1)*(double)TI.nSteps/simTime;
                const char *pMsg =
                    fabsl(pxRel) <= 1e-6 && fabsl(pyRel) <= 1e-6 ? "OK" : "error";
		printf("Simulation time: %.6g s\n"
		       "Interaction rate: %g s^-1\n"
		       "Momentum: (%Lg,%Lg) vs. (%Lg,%Lg),\n"
		       "         relative change (%Lg,%Lg): %s\n",
		       simTime, interactionRate,
		       pxStart, pyStart, pxEnd, pyEnd, pxRel, pyRel,
                       pMsg);
	    }
	} else {
	    fprintf(stderr, "Error reading data form file '%s'.\n", filename);
	    retCode = 1;
	}
    } else {
	fprintf(stderr, "Error opening file '%s'.\n", filename);
	retCode = 1;
    }

    MPI_Finalize();
    return retCode;
}
