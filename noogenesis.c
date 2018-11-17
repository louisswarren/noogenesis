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

double random_scalar(double epsilon)
{
	return epsilon * (2 * ((double) rand() / RAND_MAX) - 1);
}

Matrix *random_matrix(size_t rows, size_t cols, double epsilon)
{
	Matrix *m = new_matrix(rows, cols);
	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			elem(m, i, j) = random_scalar(epsilon);
		}
	}
	return m;
}

Vector *random_vector(size_t dim, double epsilon)
{
	Vector *x = new_vector(dim);
	for (size_t i = 0; i < dim; ++i) {
		x->elem[i] = random_scalar(epsilon);
	}
	return x;
}

Network *new_network(size_t depth, const size_t layer_sizes[])
{
	Network *n = malloc(sizeof(*n));
	n->depth = depth;
	allocate(n->weights, depth);
	allocate(n->biases, depth);
	allocate(n->memory, depth);
	for (size_t i = 0; i < depth; ++i) {
		n->weights[i] = new_matrix(layer_sizes[i], layer_sizes[i + 1]);
		n->biases[i] = new_vector(layer_sizes[i + 1]);
		n->memory[i] = new_vector(layer_sizes[i + 1]);
	}
	return n;
}

Network *random_network(size_t depth, const size_t layer_sizes[], double epsilon)
{
	Network *n = malloc(sizeof(*n));
	n->depth = depth;
	allocate(n->weights, depth);
	allocate(n->biases, depth);
	allocate(n->memory, depth);
	for (size_t i = 0; i < depth; ++i) {
		n->weights[i] = random_matrix(layer_sizes[i], layer_sizes[i + 1], epsilon);
		n->biases[i] = random_vector(layer_sizes[i + 1], epsilon);
		n->memory[i] = random_vector(layer_sizes[i + 1], epsilon);
	}
	return n;
}

void delete_network(Network *n)
{
	for (size_t i = 0; i < n->depth; ++i) {
		free(n->weights[i]);
		free(n->biases[i]);
		free(n->memory[i]);
	}
	free(n->weights);
	free(n->biases);
	free(n->memory);
	free(n);
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

void print_vector(const Vector *x)
{
	printf("[ ");
	for (size_t i = 0; i < x->dim; ++i) {
		if (i + 1 < x->dim) {
			printf("%f; ", x->elem[i]);
		} else {
			printf("%f ", x->elem[i]);
		}
	}
	printf("]\n");
}

void print_matrix(const Matrix *m)
{
	for (size_t i = 0; i < m->rows; ++i) {
		if (i == 0) {
			printf("[[ ");
		} else {
			printf(" [ ");
		}
		for (size_t j = 0; j < m->cols; ++j) {
			if (j + 1 < m->cols) {
				printf("%f, ", elem(m, i, j));
			} else {
				printf("%f ", elem(m, i, j));
			}
		}
		if (i + 1 == m->rows) {
			printf("]]\n");
		} else {
			printf("];\n");
		}
	}
}

int main(void)
{
	Matrix *a = random_matrix(2, 3, 1.0);
	Vector *x = random_vector(3, 1.0);
	Vector *y = new_vector(2);

	print_matrix(a);
	print_vector(x);

	transform(y, a, x);

	print_vector(y);

	size_t layer_sizes[] = {7, 8, 3};

	for (int i = 0; i < 1000000000; ++i) {
		Network *n = random_network(2, layer_sizes, 1.0);
		delete_network(n);
	}

	free(a);
	free(x);
	free(y);

	return 0;
}
