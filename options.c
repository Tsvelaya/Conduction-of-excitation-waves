#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "options.h"

struct proto proto = { UNDEF };
struct fibrosis fibrosis = { 0 };
struct ablation ablation = { 0 };

int device_num = 0;
int verbose = 0;
int pretend = 0;

char *output_dir = "results";
char *init_file = NULL;
char *medium_file = NULL;
float ratio = 1.0;
float frequency = 0.0;
int wait = 0;
int number_of_waves = 5;
float dec = 1.0;

int meas_activation = 0;

int stop = 500000;

static char *options = "q:r:a:d:i:m:vf:pt:s:nhw:b:D:";
static char *program_name;

static void usage(FILE *stream, int code)
{
    fprintf(stream,
"Usage: %s [OPTION]... OUTPUT_FOLDER\n"
"where OPTIONS are:\n"
" -a X,Y,R                    Abrate a circle of radius R at (X, Y)\n"
" -f MEAN,HETERO,SIZE,N,SEED  Initiate fibrotic medium with the mean percentage\n"
"                             of fibrosis MEAN, heterogeneity HETERO, size of\n"
"                             heterogeneity SIZE, number of square types N\n"
"                             and seed SEED\n"
" -s PROTO                    Specify the protocol PROTO:\n"
"                               'clear' or 'none' for nothing\n"
"                               's1' or 'left' -- one stimulus at the left boundary\n"
"                               'right' -- one stimulus at the right boundary\n"
"                               's1s2' -- spiral initialization\n"
"                               'circ:X,Y,R' -- circular stimulus at (X, Y) with\n"
"                                               the radius R\n"
"                               'rect:X0,X1,Y0,Y1' -- rectangular stimulus from\n"
"                                                   (X0, Y0) to (X1, Y1)\n"
" -i FILENAME                 Start from a given pattern in FILENAME\n"
" -n                          Measure activation time\n"
" -m FILENAME                 Use the medium defined in FILENAME\n"
" -t NUM                      Take NUM of integration steps\n"
" -d NUM                      Use the device number NUM\n"
" -v                          Be verbose\n"
" -p                          Pretend, do not perform computations\n"
" -r 						  Anisotropy ratio\n"
" -q 						  Frequency of stimulation\n"
" -w 						  Number of steps to wait\n"
" -b 						  Number of stimuli\n"
" -h                          Read this message\n"
"The OUTPUT_FOLDER defines the place to store the results\n",
	    program_name);
    exit(code);
}

void parse_options(int argc, char *argv[])
{
    int opt;
    struct stat dst;

    program_name = argv[0];

    while ((opt = getopt(argc, argv, options)) != -1) {
	switch (opt) {
	case 'h':
	    usage(stdout, EXIT_SUCCESS);
	    break;
        case 'a':
	    ablation.x = atoi(strtok(optarg, ","));
	    ablation.y = atoi(strtok(NULL, ","));
	    ablation.r = atoi(strtok(NULL, ","));
	    break;
	case 'f':
	    fibrosis.mean = atof(strtok(optarg, ",")) / 100.0f;
	    fibrosis.hetero = atof(strtok(NULL, ",")) / 100.0f;
	    fibrosis.size = atoi(strtok(NULL, ","));
	    fibrosis.discr = atoi(strtok(NULL, ","));
	    fibrosis.seed = atol(strtok(NULL, ","));
	    break;
	case 's':
	    if (strcmp(optarg, "clear") == 0 || strcmp(optarg, "none") == 0)
		proto.type = CLEAR;
	    else if (strcmp(optarg, "s1") == 0 || strcmp(optarg, "left") == 0)
		proto.type = S1;
	    else if (strcmp(optarg, "s1s2") == 0)
		proto.type = S1S2;
	    else if (strcmp(optarg, "right") == 0)
		proto.type = RIGHT;
		else if (strcmp(optarg, "top") == 0)
		proto.type = TOP;
	    else if (strncmp(optarg, "circ:", strlen("circ:")) == 0) {
		proto.type = CIRC;
		strtok(optarg, ":,");
		proto.data.circ.x = atoi(strtok(NULL, ":,"));
		proto.data.circ.y = atoi(strtok(NULL, ":,"));
		proto.data.circ.r = atoi(strtok(NULL, ":,"));
	    } else if (strncmp(optarg, "rect:", strlen("rect:")) == 0) {
		proto.type = RECT;
		strtok(optarg, ":,");
		proto.data.rect.x0 = atoi(strtok(NULL, ":,"));
		proto.data.rect.x1 = atoi(strtok(NULL, ":,"));
		proto.data.rect.y0 = atoi(strtok(NULL, ":,"));
		proto.data.rect.y1 = atoi(strtok(NULL, ":,"));
	    }
	    else {
		fprintf(stderr, "Unknown protocol: %s\n", optarg);
		exit(EXIT_FAILURE);
	    }
	    break;
	case 'i':
	    init_file = optarg;
	    break;
	case 'r':
		ratio = atof(optarg);
		break;
	case 'q':
		frequency = atof(optarg);
		break;
	case 'w':
		wait = atoi(optarg);
		break;
	case 'b':
		number_of_waves = atoi (optarg);
		break;
	case 'n':
	    meas_activation++;
	    break;
        case 'm':
            medium_file = optarg;
            break;
	case 't':
	    stop = atoi(optarg);
	    break;
	case 'D':
        dec = atof(optarg);
        break;
	case 'd':
	    device_num = atoi(optarg);
	    break;
        case 'v':
            verbose++;
            break;
        case 'p':
            pretend++;
            break;
	default:
	    usage(stderr, EXIT_FAILURE);
	}
    }

    if (optind != argc - 1) usage(stderr, EXIT_FAILURE);

    output_dir = argv[optind];
    switch (stat(output_dir, &dst)) {
    case 0:
	if (!S_ISDIR(dst.st_mode)) {
	    fprintf(stderr, "'%s' is not a directory\n", output_dir);
	    usage(stderr, EXIT_FAILURE);
	}
	break;
    default:
	if (mkdir(output_dir, 0777) != 0) {
	    perror(output_dir);
	    usage(stderr, EXIT_FAILURE);
	}
	break;
    }
}
