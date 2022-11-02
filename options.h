#ifndef OPTIONS_H
#define OPTIONS_H

struct proto {
    enum { UNDEF = -1, CLEAR = 0, S1, S1S2, TOP, RIGHT, RECT, CIRC } type;
    union {
	struct {
	    int x0, x1, y0, y1;
	} rect;
	struct {
	    int x, y, r;
	} circ;
    } data;
};

extern struct proto proto;

struct fibrosis { int size; float mean; float hetero; int discr; long seed; };
extern struct fibrosis fibrosis;

struct ablation { int r, x, y; };
extern struct ablation ablation;

extern int verbose;
extern int pretend;
extern int device_num;

extern char *output_dir;
extern char *init_file;
extern char *medium_file;
extern float ratio;
extern float frequency;
extern int wait;
extern int number_of_waves;
extern float dec;

extern int meas_activation;

extern int stop;

void parse_options(int argc, char *argv[]);

#endif	/* OPTIONS_H */
