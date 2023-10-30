/* File generated with cbindgen from the Rust library -- don't change it directly */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * PointerOrError returns either a `void *` pointer or an error.
 * It can be used by functions interfacing with Rust from other languages (using the C binding).
 *
 * Either `value` or `error` will be defined. The `value` underlying type is defined by the function
 * returning a `PointerOrError`.
 *
 * Ownership of `value` should be documented by the function returning it.
 * Ownership of `error` is transferred back to the caller.
 */
typedef struct PointerOrError {
  void *value;
  char *error;
} PointerOrError;

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
 * Offset of the toke in the sentence.
 * The Go library limits this to u32 -- we don't expect sentences larger than ~4GB.
 */
typedef struct Offset {
  uint32_t start;
  uint32_t end;
} Offset;

/**
 * Buffer represents the result of an encoded sentence.
 * Each of the fields are only filled if they were requested in the corresponding
 * EncodeParams setting.
 */
typedef struct Buffer {
  uint32_t *ids;
  uint32_t *type_ids;
  uint32_t *special_tokens_mask;
  uint32_t *attention_mask;
  char **tokens;
  struct Offset *offsets;
  uint32_t len;
} Buffer;

/**
 * EncodeResult represents the result of encoding one (`encode` function)
 * or more (`encode_batch` function) sentences.
 *
 * It will contain either an error as a C string, or a number of Buffer
 * results, one per sentence encoded -- only one if using `encode` function.
 *
 * Once it is no longer used, free the data with `free_encode_results`.
 */
typedef struct EncodeResults {
  uint32_t len;
  struct Buffer *encoded;
  char *error;
} EncodeResults;

/**
 * EncodeParams specifies what information to return from the
 * encoded sentences.
 * It controls which fields in Buffer are set.
 */
typedef struct EncodeParams {
  bool add_special_tokens;
  bool return_tokens;
  bool return_type_ids;
  bool return_special_tokens_mask;
  bool return_attention_mask;
  bool return_offsets;
  bool with_offsets_char_mode;
} EncodeParams;

/**
 * This function returns a Tokenizer reference to Golang (casted as a C `void*` in the `value` field) or
 * an error.
 *
 * The parameter `bytes` should be the json contents for a `tokenizer.json` file, with its definitions (symbols,
 * truncation parameters, etc.)
 *
 * # Safety
 *
 * The caller has ownership of `bytes` and of the returned `Tokenizer`.
 */
struct PointerOrError from_bytes(const uint8_t *bytes,
                                 uint32_t len);

/**
 * # Safety
 *
 * This function is return Tokenizer object to Golang from tokenizer.json
 */
void *from_file(const char *config);

/**
 * tokenizer.Decode method.
 * The returned string needs to be deallocated with `free_string`.
 */
char *decode(void *tokenizer_ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

/**
 * Returns the vocab size.
 */
uint32_t vocab_size(void *ptr);

/**
 * Frees a Tokenizer allocated by Rust and returned to Golang.
 */
void free_tokenizer(void *ptr);

/**
 * Frees a `*C.char` allocated by Rust and return to Golang.
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
 * get_truncation gets the current Tokenizer's truncation parameters.
 *
 * If there are truncation parameters configured in the Tokenizer, the values are read into the `params` passed,
 * and it returns true.
 *
 * If there are no truncation values configured, it returns false.
 */
bool get_truncation(void *tokenizer_ptr,
                    struct TruncationParams *params);

/**
 * set_padding modifies the tokenizer with the given padding parameters.
 * It doesn't return anything.
 */
void set_padding(void *tokenizer_ptr, const struct PaddingParams *params);

/**
 * get_padding gets the current Tokenizer's padding parameters.
 *
 * If there are padding parameters configured in the Tokenizer, the values are read into the `params` passed,
 * and it returns true. The `params.pad_token` ownership is transferred to the caller, who must free it
 * after use (see `free_string()`).
 *
 * If there are no truncation values configured, it returns false.
 */
bool get_padding(void *tokenizer_ptr,
                 struct PaddingParams *params);

/**
 * Encodes string using given tokenizer and EncodeParams.
 */
struct EncodeResults encode(void *tokenizer_ptr, const char *message, struct EncodeParams options);

/**
 * Encode a batch of strings using given tokenizer and EncodeParams.
 * The
 */
struct EncodeResults encode_batch(void *tokenizer_ptr,
                                  uint32_t num_messages,
                                  const char *const *messages,
                                  struct EncodeParams options);

/**
 * This function is release Vec<Buffer> from Rust returned to Golang by `encode_batch`.
 */
void free_encode_results(struct EncodeResults results);

/* File generated with cbindgen from the Rust library -- don't change it directly */
