#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <openssl/evp.h>
#include <openssl/sha.h>

static PyObject *sha256_oneshot(PyObject *self, PyObject *object) {
    Py_buffer view;
    unsigned char digest[SHA256_DIGEST_LENGTH];
    unsigned char *result = NULL;
    if (PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) != 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    result = SHA256((const unsigned char *)view.buf, (size_t)view.len, digest);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (result == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "OpenSSL SHA256 one-shot failed");
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)digest, SHA256_DIGEST_LENGTH);
}

static PyObject *evp_q_digest(PyObject *self, PyObject *object) {
    Py_buffer view;
    unsigned char digest[EVP_MAX_MD_SIZE];
    size_t digest_length = 0;
    int success = 0;
    if (PyObject_GetBuffer(object, &view, PyBUF_CONTIG_RO) != 0) {
        return NULL;
    }
    Py_BEGIN_ALLOW_THREADS
    success = EVP_Q_digest(
        NULL,
        "SHA256",
        NULL,
        view.buf,
        (size_t)view.len,
        digest,
        &digest_length
    );
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&view);
    if (success != 1 || digest_length != SHA256_DIGEST_LENGTH) {
        PyErr_SetString(PyExc_RuntimeError, "OpenSSL EVP_Q_digest failed");
        return NULL;
    }
    return PyBytes_FromStringAndSize((const char *)digest, (Py_ssize_t)digest_length);
}

static PyMethodDef methods[] = {
    {"sha256_oneshot", sha256_oneshot, METH_O, "Hash one contiguous buffer with OpenSSL SHA256."},
    {"evp_q_digest", evp_q_digest, METH_O, "Hash one contiguous buffer with OpenSSL EVP_Q_digest."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "lexigen_sha256_native",
    "Frozen exact SHA-256 native candidates.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit_lexigen_sha256_native(void) {
    return PyModule_Create(&module);
}
