/* File generated with cbindgen from the Rust library -- don't change it directly */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct Offset {
  uint32_t start;
  uint32_t end;
} Offset;

typedef struct Buffer {
  uint32_t *ids;
  uint32_t *type_ids;
  uint32_t *special_tokens_mask;
  uint32_t *attention_mask;
  char **tokens;
  struct Offset *offsets;
  uintptr_t len;
} Buffer;

typedef struct EncodeOptions {
  bool add_special_tokens;
  bool return_type_ids;
  bool return_special_tokens_mask;
  bool return_attention_mask;
  bool return_offsets;
  bool with_offsets_char_mode;
} EncodeOptions;

/**
 * TruncationParameters represents the truncation parameters
 * that can be set with "with_truncation".
 */
typedef struct TruncationParameters {
  uint8_t direction;
  uint8_t strategy;
  uint32_t max_length;
  uint32_t stride;
} TruncationParameters;

/**
 * This function returns a Tokenizer reference to Golang, casted as a C `void*` after reading
 * tokenizer.json to bytes.
 *
 * # Safety
 *
 * The caller has ownership of `bytes` and of the returned `Tokenizer`.
 */
void *from_bytes(const uint8_t *bytes, uint32_t len);

/**
 * This function is return Tokenizer(truncation mode) object to Golang from
 * after read tokenizer.json to bytes
 *
 * # Safety
 *
 */
void *from_bytes_with_truncation(const uint8_t *bytes, uint32_t len, uint32_t max_len, uint8_t dir);

/**
 * # Safety
 *
 * This function is return Tokenizer object to Golang from tokenizer.json
 */
void *from_file(const char *config);

/**
 * # Safety
 *
 * This function is tokenizer single encode function
 */
struct Buffer encode(void *ptr, const char *message, const struct EncodeOptions *options);

/**
 * # Safety
 *
 * This function is tokenizer batch encode function
 */
struct Buffer *encode_batch(void *ptr,
                            const char *const *messages,
                            const struct EncodeOptions *options);

/**
 * # Safety
 *
 * This function is tokenizer decode function
 */
char *decode(void *ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

/**
 * # Safety
 *
 * This function is return vocab size to Golang
 */
uint32_t vocab_size(void *ptr);

/**
 * # Safety
 *
 * This function is release Tokenizer from Rust return to Golang
 */
void free_tokenizer(void *ptr);

/**
 * # Safety
 *
 * This function is release Buffer struct from Rust return to Golang
 */
void free_buffer(struct Buffer buf);

/**
 * # Safety
 *
 * This function is release Vec<Buffer> from Rust return to Golang
 */
void free_batch_buffer(struct Buffer *bufs);

/**
 * # Safety
 *
 * This function is release C.char from Rust return to Golang
 */
void free_string(char *ptr);

/**
 * with_truncation modifies the given tokenizer with the given truncation parameters.
 * It returns a string with an error message (owned by caller) if something went wrong.
 */
char *with_truncation(void *tokenizer_ptr, const struct TruncationParameters *params);

/* File generated with cbindgen from the Rust library -- don't change it directly */
