#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <openssl/evp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct worker_args {
    const unsigned char *key;
    const unsigned char *nonce;
    const unsigned char *input;
    unsigned char *output;
    size_t length;
    uint32_t counter;
    int ok;
};

static int chacha_xor(const unsigned char *key, const unsigned char *nonce,
                      uint32_t counter, const unsigned char *input,
                      unsigned char *output, size_t length) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (ctx == NULL) {
        return 0;
    }

    unsigned char iv[16];
    iv[0] = (unsigned char)counter;
    iv[1] = (unsigned char)(counter >> 8);
    iv[2] = (unsigned char)(counter >> 16);
    iv[3] = (unsigned char)(counter >> 24);
    memcpy(iv + 4, nonce, 12);

    int ok = EVP_EncryptInit_ex(ctx, EVP_chacha20(), NULL, key, iv);
    size_t offset = 0;
    while (ok && offset < length) {
        size_t part = length - offset;
        if (part > ((size_t)1 << 30)) {
            part = ((size_t)1 << 30);
        }
        int output_length = 0;
        ok = EVP_EncryptUpdate(
            ctx,
            output + offset,
            &output_length,
            input + offset,
            (int)part
        );
        if (!ok || output_length != (int)part) {
            ok = 0;
            break;
        }
        offset += part;
    }

    EVP_CIPHER_CTX_free(ctx);
    return ok;
}

static void *worker(void *opaque) {
    struct worker_args *args = (struct worker_args *)opaque;
    args->ok = chacha_xor(
        args->key,
        args->nonce,
        args->counter,
        args->input,
        args->output,
        args->length
    );
    return NULL;
}

static void put_little_endian_u64(unsigned char output[8], uint64_t value) {
    for (int index = 0; index < 8; ++index) {
        output[index] = (unsigned char)(value >> (8 * index));
    }
}

static int poly1305_tag(const unsigned char key[32],
                        const unsigned char *associated_data,
                        size_t associated_data_length,
                        const unsigned char *ciphertext,
                        size_t ciphertext_length,
                        unsigned char tag[16]) {
    EVP_MAC *mac = EVP_MAC_fetch(NULL, "POLY1305", NULL);
    if (mac == NULL) {
        return 0;
    }
    EVP_MAC_CTX *ctx = EVP_MAC_CTX_new(mac);
    EVP_MAC_free(mac);
    if (ctx == NULL) {
        return 0;
    }

    int ok = EVP_MAC_init(ctx, key, 32, NULL);
    static const unsigned char zeros[16] = {0};
    unsigned char lengths[16];
    put_little_endian_u64(lengths, (uint64_t)associated_data_length);
    put_little_endian_u64(lengths + 8, (uint64_t)ciphertext_length);

    if (ok && associated_data_length != 0) {
        ok = EVP_MAC_update(ctx, associated_data, associated_data_length);
    }
    size_t padding = (16 - (associated_data_length & 15)) & 15;
    if (ok && padding != 0) {
        ok = EVP_MAC_update(ctx, zeros, padding);
    }
    if (ok && ciphertext_length != 0) {
        ok = EVP_MAC_update(ctx, ciphertext, ciphertext_length);
    }
    padding = (16 - (ciphertext_length & 15)) & 15;
    if (ok && padding != 0) {
        ok = EVP_MAC_update(ctx, zeros, padding);
    }
    if (ok) {
        ok = EVP_MAC_update(ctx, lengths, sizeof(lengths));
    }

    size_t tag_length = 0;
    if (ok) {
        ok = EVP_MAC_final(ctx, tag, &tag_length, 16);
    }
    EVP_MAC_CTX_free(ctx);
    return ok && tag_length == 16;
}

static int encrypt_parallel(const unsigned char *key,
                            const unsigned char *nonce,
                            const unsigned char *plaintext,
                            size_t plaintext_length,
                            const unsigned char *associated_data,
                            size_t associated_data_length,
                            unsigned char *ciphertext,
                            unsigned char tag[16],
                            int requested_threads) {
    unsigned char zero_block[64] = {0};
    unsigned char first_block[64];
    if (!chacha_xor(key, nonce, 0, zero_block, first_block, 64)) {
        return 0;
    }

    size_t total_blocks = (plaintext_length + 63) / 64;
    int thread_count = requested_threads;
    if (thread_count < 1) {
        thread_count = 1;
    }
    if (total_blocks != 0 && (size_t)thread_count > total_blocks) {
        thread_count = (int)total_blocks;
    }
    if (plaintext_length == 0) {
        thread_count = 0;
    }

    int ok = 1;
    if (thread_count == 1) {
        ok = chacha_xor(
            key,
            nonce,
            1,
            plaintext,
            ciphertext,
            plaintext_length
        );
    } else if (thread_count > 1) {
        pthread_t *thread_ids = calloc((size_t)thread_count, sizeof(*thread_ids));
        struct worker_args *arguments = calloc(
            (size_t)thread_count,
            sizeof(*arguments)
        );
        if (thread_ids == NULL || arguments == NULL) {
            free(thread_ids);
            free(arguments);
            return 0;
        }

        size_t base_blocks = total_blocks / (size_t)thread_count;
        size_t remaining_blocks = total_blocks % (size_t)thread_count;
        size_t start_block = 0;
        int started_threads = 0;

        for (int index = 0; index < thread_count; ++index) {
            size_t block_count = base_blocks;
            if ((size_t)index < remaining_blocks) {
                ++block_count;
            }
            size_t byte_offset = start_block * 64;
            size_t chunk_length = block_count * 64;
            if (byte_offset + chunk_length > plaintext_length) {
                chunk_length = plaintext_length - byte_offset;
            }

            arguments[index].key = key;
            arguments[index].nonce = nonce;
            arguments[index].input = plaintext + byte_offset;
            arguments[index].output = ciphertext + byte_offset;
            arguments[index].length = chunk_length;
            arguments[index].counter = (uint32_t)(1 + start_block);
            arguments[index].ok = 0;

            if (pthread_create(
                    &thread_ids[index],
                    NULL,
                    worker,
                    &arguments[index]
                ) != 0) {
                ok = 0;
                break;
            }
            ++started_threads;
            start_block += block_count;
        }

        for (int index = 0; index < started_threads; ++index) {
            if (pthread_join(thread_ids[index], NULL) != 0 ||
                    !arguments[index].ok) {
                ok = 0;
            }
        }
        free(thread_ids);
        free(arguments);
    }

    if (!ok) {
        return 0;
    }
    return poly1305_tag(
        first_block,
        associated_data,
        associated_data_length,
        ciphertext,
        plaintext_length,
        tag
    );
}

static PyObject *python_encrypt(PyObject *self, PyObject *args) {
    Py_buffer key = {0};
    Py_buffer nonce = {0};
    Py_buffer plaintext = {0};
    Py_buffer associated_data = {0};
    int thread_count = 0;

    if (!PyArg_ParseTuple(
            args,
            "y*y*y*y*i",
            &key,
            &nonce,
            &plaintext,
            &associated_data,
            &thread_count
        )) {
        return NULL;
    }
    if (key.len != 32 || nonce.len != 12) {
        PyErr_SetString(
            PyExc_ValueError,
            "ChaCha20-Poly1305 requires a 32-byte key and 12-byte nonce"
        );
        goto error;
    }

    PyObject *ciphertext_object = PyBytes_FromStringAndSize(NULL, plaintext.len);
    if (ciphertext_object == NULL) {
        goto error;
    }
    unsigned char *ciphertext = (unsigned char *)PyBytes_AS_STRING(
        ciphertext_object
    );
    unsigned char tag[16];
    int ok = 0;

    Py_BEGIN_ALLOW_THREADS
    ok = encrypt_parallel(
        (const unsigned char *)key.buf,
        (const unsigned char *)nonce.buf,
        (const unsigned char *)plaintext.buf,
        (size_t)plaintext.len,
        (const unsigned char *)associated_data.buf,
        (size_t)associated_data.len,
        ciphertext,
        tag,
        thread_count
    );
    Py_END_ALLOW_THREADS

    if (!ok) {
        Py_DECREF(ciphertext_object);
        PyErr_SetString(
            PyExc_RuntimeError,
            "native ChaCha20-Poly1305 execution failed"
        );
        goto error;
    }

    PyObject *tag_object = PyBytes_FromStringAndSize((const char *)tag, 16);
    if (tag_object == NULL) {
        Py_DECREF(ciphertext_object);
        goto error;
    }
    PyObject *result = PyTuple_Pack(2, ciphertext_object, tag_object);
    Py_DECREF(ciphertext_object);
    Py_DECREF(tag_object);

    PyBuffer_Release(&key);
    PyBuffer_Release(&nonce);
    PyBuffer_Release(&plaintext);
    PyBuffer_Release(&associated_data);
    return result;

error:
    if (key.obj != NULL) {
        PyBuffer_Release(&key);
    }
    if (nonce.obj != NULL) {
        PyBuffer_Release(&nonce);
    }
    if (plaintext.obj != NULL) {
        PyBuffer_Release(&plaintext);
    }
    if (associated_data.obj != NULL) {
        PyBuffer_Release(&associated_data);
    }
    return NULL;
}

static PyMethodDef methods[] = {
    {
        "encrypt",
        python_encrypt,
        METH_VARARGS,
        "Encrypt with parallel RFC 8439 ChaCha20-Poly1305."
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "lexigen_chacha_native",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_lexigen_chacha_native(void) {
    return PyModule_Create(&module);
}
