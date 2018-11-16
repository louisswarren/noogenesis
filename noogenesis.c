#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#define elem(M, I, J) (M->elem[i * M->cols + j])

#define allocate(X, N) X = emalloc(sizeof(*X) * N)

void *emalloc(size_t size)
{
	void *x = malloc(size);
	if (!x) {
		fprintf(stderr, "Out of memory.\n");
		exit(EXIT_FAILURE);
	}
	return x;
}

typedef struct {
	size_t rows;
	size_t cols;
	double elem[];
} Matrix;

typedef struct {
	size_t dim;
	double elem[];
} Vector;

typedef struct {
	size_t depth;
	Matrix **weights;
	Vector **biases;
	Vector **memory;
} Network;

Matrix *new_matrix(size_t rows, size_t cols)
{
	Matrix *m = malloc(sizeof(*m) + rows * cols * sizeof(*m->elem));
	m->rows = rows;
	m->cols = cols;
	return m;
}

Vector *new_vector(size_t dim)
{
	Vector *x = malloc(sizeof(*x) + dim * sizeof(*x->elem));
	x->dim = dim;
	return x;
}

Network *new_network(size_t depth, const size_t layer_sizes[])
{
	Network *n = malloc(sizeof(*n));
	allocate(n->weights, depth);
	allocate(n->biases, depth);
	allocate(n->memory, depth);
	for (size_t i = 0; i < depth; ++i) {
		n->weights[i] = new_matrix(layer_sizes[i], layer_sizes[i + 1]);
		n->biases[i] = new_vector(layer_sizes[i + 1]);
		n->memory[i] = new_vector(layer_sizes[i + 1]);
	}
}

void transform(Vector *y, const Matrix *a, const Vector *x)
{
	assert(y->dim == a->rows);
	assert(a->cols == x->dim);
	for (size_t i = 0; i < y->dim; ++i) {
		y->elem[i] = 0;
		for (size_t j = 0; j < x->dim; ++j) {
			y->elem[i] += elem(a, i, j) * x->elem[j];
		}
	}
}

int main(void)
{
	Matrix *a = new_matrix(2, 3);
	Vector *x = new_vector(3);
	Vector *y = new_vector(2);

	a->elem[0] = 1;
	a->elem[1] = 2;
	a->elem[2] = 3;
	a->elem[3] = 4;
	a->elem[4] = 5;
	a->elem[5] = 6;

	x->elem[0] = 5;
	x->elem[1] = 7;
	x->elem[2] = 11;

	transform(y, a, x);

	printf("%f\n", y->elem[0]);
	printf("%f\n", y->elem[1]);

	return 0;
}
