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
typedef struct TruncationParams {
  uint8_t direction;
  uint8_t strategy;
  uint32_t max_length;
  uint32_t stride;
} TruncationParams;

/**
 * PaddingParams represents the padding parameters: it maps to the values in
 * tokenizers::tokenizer::PaddingParams.
 */
typedef struct PaddingParams {
  uint32_t strategy;
  uint8_t direction;
  uint32_t pad_to_multiple_of;
  uint32_t pad_id;
  uint32_t pad_type_id;
  const char *pad_token;
} PaddingParams;

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
 * Encodes string using given tokenizer and EncodeOptions.
 */
struct Buffer encode(void *tokenizer_ptr, const char *message, const struct EncodeOptions *options);

/**
 * Encode a batch of strings using given tokenizer and EncodeOptions.
 */
struct Buffer *encode_batch(void *tokenizer_ptr,
                            const char *const *messages,
                            const struct EncodeOptions *options);

/**
 * tokenizer.Decode method.
 * The returned string needs to be deallocated with `free_string`.
 */
char *decode(void *tokenizer_ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

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
 * set_truncation modifies the tokenizer with the given truncation parameters.
 * It returns null if ok, or a string with an error message (owned by caller) if something went wrong.
 * The returned string needs to be freed with `free_string`.
 */
char *set_truncation(void *tokenizer_ptr,
                     const struct TruncationParams *params);

/**
 * set_padding modifies the tokenizer with the given padding parameters.
 * It doesn't return anything.
 */
void set_padding(void *tokenizer_ptr, const struct PaddingParams *params);

/* File generated with cbindgen from the Rust library -- don't change it directly */
