#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const unsigned char *key;
    const unsigned char *nonce;
    const unsigned char *input;
    unsigned char *output;
    size_t length;
    uint32_t counter;
    int ok;
} worker_args;

static void store32_le(unsigned char out[4], uint32_t value) {
    out[0] = (unsigned char)value;
    out[1] = (unsigned char)(value >> 8);
    out[2] = (unsigned char)(value >> 16);
    out[3] = (unsigned char)(value >> 24);
}

static void store64_le(unsigned char out[8], uint64_t value) {
    for (int i = 0; i < 8; ++i) out[i] = (unsigned char)(value >> (8 * i));
}

static int chacha_xor(const unsigned char *key, const unsigned char *nonce,
                      uint32_t counter, const unsigned char *input,
                      unsigned char *output, size_t length) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return 0;
    unsigned char iv[16];
    store32_le(iv, counter);
    memcpy(iv + 4, nonce, 12);
    int ok = EVP_EncryptInit_ex(ctx, EVP_chacha20(), NULL, key, iv) == 1;
    size_t offset = 0;
    while (ok && offset < length) {
        size_t remaining = length - offset;
        int chunk = remaining > (size_t)INT_MAX ? INT_MAX : (int)remaining;
        int produced = 0;
        ok = EVP_EncryptUpdate(ctx, output + offset, &produced,
                               input + offset, chunk) == 1 && produced == chunk;
        offset += (size_t)chunk;
    }
    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

static void *worker_main(void *opaque) {
    worker_args *args = (worker_args *)opaque;
    args->ok = chacha_xor(args->key, args->nonce, args->counter,
                          args->input, args->output, args->length);
    return NULL;
}

static int derive_poly1305_key(const unsigned char *key,
                               const unsigned char *nonce,
                               unsigned char poly_key[32]) {
    unsigned char zeros[64] = {0};
    unsigned char block[64];
    if (!chacha_xor(key, nonce, 0, zeros, block, sizeof(block))) return 0;
    memcpy(poly_key, block, 32);
    OPENSSL_cleanse(block, sizeof(block));
    return 1;
}

static int poly1305_tag(const unsigned char poly_key[32],
                        const unsigned char *aad, size_t aad_len,
                        const unsigned char *ciphertext, size_t ciphertext_len,
                        unsigned char tag[16]) {
    EVP_MAC *mac = EVP_MAC_fetch(NULL, "POLY1305", NULL);
    if (!mac) return 0;
    EVP_MAC_CTX *ctx = EVP_MAC_CTX_new(mac);
    EVP_MAC_free(mac);
    if (!ctx) return 0;
    int ok = EVP_MAC_init(ctx, poly_key, 32, NULL) == 1;
    static const unsigned char zeros[16] = {0};
    if (ok && aad_len) ok = EVP_MAC_update(ctx, aad, aad_len) == 1;
    size_t aad_pad = (16 - (aad_len & 15)) & 15;
    if (ok && aad_pad) ok = EVP_MAC_update(ctx, zeros, aad_pad) == 1;
    if (ok && ciphertext_len) ok = EVP_MAC_update(ctx, ciphertext, ciphertext_len) == 1;
    size_t ciphertext_pad = (16 - (ciphertext_len & 15)) & 15;
    if (ok && ciphertext_pad) ok = EVP_MAC_update(ctx, zeros, ciphertext_pad) == 1;
    unsigned char lengths[16];
    store64_le(lengths, (uint64_t)aad_len);
    store64_le(lengths + 8, (uint64_t)ciphertext_len);
    if (ok) ok = EVP_MAC_update(ctx, lengths, sizeof(lengths)) == 1;
    size_t tag_len = 16;
    if (ok) ok = EVP_MAC_final(ctx, tag, &tag_len, 16) == 1 && tag_len == 16;
    EVP_MAC_CTX_free(ctx);
    return ok;
}

static PyObject *parallel_encrypt(PyObject *self, PyObject *args) {
    const unsigned char *key, *nonce, *plaintext, *aad;
    Py_ssize_t key_len, nonce_len, plaintext_len, aad_len;
    int workers;
    if (!PyArg_ParseTuple(args, "y#y#y#y#i",
                          &key, &key_len, &nonce, &nonce_len,
                          &plaintext, &plaintext_len, &aad, &aad_len,
                          &workers)) return NULL;
    if (key_len != 32 || nonce_len != 12) {
        PyErr_SetString(PyExc_ValueError, "ChaCha20-Poly1305 requires a 32-byte key and 12-byte nonce");
        return NULL;
    }
    if (workers < 1 || workers > 32) {
        PyErr_SetString(PyExc_ValueError, "workers must be between 1 and 32");
        return NULL;
    }
    PyObject *ciphertext_obj = PyBytes_FromStringAndSize(NULL, plaintext_len);
    if (!ciphertext_obj) return NULL;
    unsigned char *ciphertext = (unsigned char *)PyBytes_AS_STRING(ciphertext_obj);
    size_t length = (size_t)plaintext_len;
    size_t total_blocks = (length + 63u) / 64u;
    int active = total_blocks == 0 ? 1 : workers;
    if ((size_t)active > total_blocks && total_blocks > 0) active = (int)total_blocks;
    pthread_t threads[32];
    worker_args wargs[32];
    int created = 0;
    int ok = 1;
    unsigned char poly_key[32];
    unsigned char tag[16];

    Py_BEGIN_ALLOW_THREADS
    if (!derive_poly1305_key(key, nonce, poly_key)) ok = 0;
    if (ok && length > 0) {
        size_t base = total_blocks / (size_t)active;
        size_t rem = total_blocks % (size_t)active;
        size_t start_block = 0;
        for (int i = 0; i < active; ++i) {
            size_t blocks = base + ((size_t)i < rem ? 1u : 0u);
            size_t byte_start = start_block * 64u;
            size_t byte_end = (start_block + blocks) * 64u;
            if (byte_end > length) byte_end = length;
            wargs[i].key = key;
            wargs[i].nonce = nonce;
            wargs[i].input = plaintext + byte_start;
            wargs[i].output = ciphertext + byte_start;
            wargs[i].length = byte_end - byte_start;
            wargs[i].counter = (uint32_t)(1u + start_block);
            wargs[i].ok = 0;
            if (pthread_create(&threads[i], NULL, worker_main, &wargs[i]) != 0) {
                ok = 0;
                break;
            }
            ++created;
            start_block += blocks;
        }
        for (int i = 0; i < created; ++i) {
            if (pthread_join(threads[i], NULL) != 0 || !wargs[i].ok) ok = 0;
        }
    }
    if (ok) ok = poly1305_tag(poly_key, aad, (size_t)aad_len,
                               ciphertext, length, tag);
    OPENSSL_cleanse(poly_key, sizeof(poly_key));
    Py_END_ALLOW_THREADS

    if (!ok) {
        Py_DECREF(ciphertext_obj);
        PyErr_SetString(PyExc_RuntimeError, "native ChaCha20-Poly1305 operation failed");
        return NULL;
    }
    PyObject *tag_obj = PyBytes_FromStringAndSize((const char *)tag, 16);
    if (!tag_obj) {
        Py_DECREF(ciphertext_obj);
        return NULL;
    }
    return Py_BuildValue("NN", ciphertext_obj, tag_obj);
}

static PyMethodDef methods[] = {
    {"encrypt", parallel_encrypt, METH_VARARGS, "Encrypt using parallel RFC 8439 ChaCha20-Poly1305."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "lexigen_chacha",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_lexigen_chacha(void) {
    return PyModule_Create(&module);
}
