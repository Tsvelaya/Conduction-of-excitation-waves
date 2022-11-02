#include <assert.h>
#include <math.h>

#include "fibrosis.h"
#include "ran.h"
#include "debug.h"

static inline float square2(float low, float high, long *state)
{
    return ran1(state) < 0.5 ? low : high;
}

void set_fibrosis(char *medium, float prob, long seed)
{
    int i;

    for (i = 0; i < DIMX * DIMY; i++)
	SET_ARRAY_ADDRESS(medium, i, ran1(&seed) < prob ? 0 : 1);
}

void set_square_fibrosis(medium_t medium,
			 float mean, float hetero, int size, long seed)
{
    int i, j;
    float delta, low, high;

    assert(mean >= 0.0 && mean <= 1.0);
    assert(hetero >= 0.0 && hetero <= 1.0);
    assert(size > 0);

    delta = fminf(mean, 1 - mean) * hetero;
    low = mean - delta;
    high = mean + delta;

    for (i = 0; i < DIMY; i += size)
	for (j = 0; j < DIMY; j += size) {
	    int i2, j2;
	    float chance;

	    chance = square2(low, high, &seed);
	    for (i2 = i; i2 < MIN(DIMY, i + size); i2++)
		for (j2 = j; j2 < MIN(DIMX, j + size); j2++)
                    if (ran1(&seed) < chance)
			SET_ARRAY(medium, i2, j2, 0);
	}
}

static inline float square_k(float k, float *vals, long *state)
{
    int idx;

    idx = (int) floorf(ran1(state) * (float) k);
    return vals[idx];
}

typedef enum { LOOSE, FIT } type;

static float *discretize(float rng[2], int k, type type)
{
    int i;
    float *a;

    a = xmalloc(sizeof(*a) * k, HOST);

    switch (type) {
    case LOOSE:
	for (i = 0; i < k; i++)
	    a[i] = rng[0] + ((float) (i + 1)) * (rng[1] - rng[0]) / ((float) (k + 1));
	break;
    case FIT:
	assert(k > 1);
	for (i = 0; i < k; i++)
	    a[i] = rng[0] + ((float) i) * (rng[1] - rng[0]) / ((float) (k - 1));
	break;
    }

    return a;
}

static float range_[2];

static float *range(float mean, float hetero)
{
    float shift;

    shift = hetero * fminf(mean, 1.0f - mean);
    range_[0] = mean - shift;
    range_[1] = mean + shift;

    return range_;
}

static float *fibro_values(float mean, float hetero, int k)
{
    float *rng;

    rng = range(mean, hetero);
    return discretize(rng, k, FIT);
}

void set_square_fibrosis2(medium_t medium, float mean, float hetero,
			  int size, int discr, long seed)
{
    int i, j;
    float *vals;

    assert(mean >= 0.0 && mean <= 1.0);
    assert(hetero >= 0.0 && hetero <= 1.0);
    assert(size > 0);
    assert(discr > 1);

    vals = fibro_values(mean, hetero, discr);

    for (i = 0; i < DIMY; i += size)
	for (j = 0; j < DIMX; j += size) {
	    int i2, j2;
	    float chance;

	    chance = square_k(discr, vals, &seed);

	    for (i2 = i; i2 < MIN(DIMY, i + size); i2++)
		for (j2 = j; j2 < MIN(DIMX, j + size); j2++)
		    if (ran1(&seed) < chance)
			SET_ARRAY(medium, i2, j2, 0);
	}
}

void set_ablation(medium_t medium, int x, int y, int r)
{
    int i, j;

    for (i = 0; i < DIMY; i++)
        for (j = 0; j < DIMX; j++)
            if (SQR(x - j) + SQR(y - i) <= SQR(r))
		SET_ARRAY(medium, i, j, 0);
}
