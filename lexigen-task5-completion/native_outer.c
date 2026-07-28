#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

static PyObject *parallel_outer(PyObject *self, PyObject *args) {
    PyObject *left_object = NULL;
    PyObject *right_object = NULL;
    int thread_count = 0;
    if (!PyArg_ParseTuple(args, "OOi", &left_object, &right_object, &thread_count)) {
        return NULL;
    }
    if (thread_count < 1) {
        PyErr_SetString(PyExc_ValueError, "thread_count must be positive");
        return NULL;
    }

    PyArrayObject *left = (PyArrayObject *)PyArray_FROM_OTF(
        left_object,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );
    PyArrayObject *right = (PyArrayObject *)PyArray_FROM_OTF(
        right_object,
        NPY_DOUBLE,
        NPY_ARRAY_IN_ARRAY
    );
    if (left == NULL || right == NULL) {
        Py_XDECREF(left);
        Py_XDECREF(right);
        return NULL;
    }
    if (PyArray_NDIM(left) != 1 || PyArray_NDIM(right) != 1) {
        Py_DECREF(left);
        Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "outer-product inputs must be one-dimensional");
        return NULL;
    }

    const npy_intp left_size = PyArray_DIM(left, 0);
    const npy_intp right_size = PyArray_DIM(right, 0);
    npy_intp dimensions[2] = {left_size, right_size};
    PyArrayObject *output = (PyArrayObject *)PyArray_SimpleNew(
        2,
        dimensions,
        NPY_DOUBLE
    );
    if (output == NULL) {
        Py_DECREF(left);
        Py_DECREF(right);
        return NULL;
    }

    const double *left_data = (const double *)PyArray_DATA(left);
    const double *right_data = (const double *)PyArray_DATA(right);
    double *output_data = (double *)PyArray_DATA(output);

    Py_BEGIN_ALLOW_THREADS
    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (npy_intp row = 0; row < left_size; ++row) {
        const double scale = left_data[row];
        double *destination = output_data + row * right_size;
        #pragma omp simd
        for (npy_intp column = 0; column < right_size; ++column) {
            destination[column] = scale * right_data[column];
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(left);
    Py_DECREF(right);
    return (PyObject *)output;
}

static PyMethodDef methods[] = {
    {
        "outer",
        parallel_outer,
        METH_VARARGS,
        "Compute a float64 outer product with a fixed OpenMP thread count."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "lexigen_outer_native",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_lexigen_outer_native(void) {
    import_array();
    return PyModule_Create(&module);
}
