// Package rs wraps the Rust tokenizer.
//
// The two parts used by this wrapper are:
//
//   - The linked Rust tokenizer is in [huggingface/tokenizers/tokenizers](https://github.com/huggingface/tokenizers/tree/main/tokenizers).
//   - The Rust wrapper with a C signature (`extern "C"`) is implemented in the subdirectory
//     [github.com/gomlx/tokenizers/rs](https://github.com/gomlx/tokenizers/tree/main/rs).
//
// End users should use the public library in [github.com/gomlx/tokenizers](https://github.com/gomlx/tokenizers) instead.
package rs

// The first lines below link the pre-built library according to the platform configured.
// If adding support for a another platform, please also add the building rules in `magefile.go`, in
// the project's root directory.

/*
#cgo linux&&amd64 LDFLAGS: ${SRCDIR}/../../lib/linux_amd64/libgomlx_tokenizers.a -ldl -lm -lstdc++
#include <stdlib.h>
#include "gomlx_tokenizers.h"
*/
import "C"

import (
	"github.com/pkg/errors"
	"io"
	"runtime"
	"unsafe"
)

type Offset struct {
	Start uint32
	End   uint32
}

type TokenizerResult struct {
	TokenIds          []uint32
	TypeIds           []uint32
	SpecialTokensMask []uint32
	AttentionMask     []uint32
	Tokens            []string
	Offsets           []Offset
}

type EncodeParams = C.EncodeParams
type EncodeOption func(eo *EncodeParams)

func WithReturnAll(withCharMode bool) EncodeOption {
	return func(eo *EncodeParams) {
		*eo = EncodeParams{
			return_type_ids:            true,
			return_tokens:              true,
			return_special_tokens_mask: true,
			return_attention_mask:      true,
			return_offsets:             true,
			with_offsets_char_mode:     C.bool(withCharMode),
		}
	}
}

func WithTokens() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_tokens = true
	}
}

func WithReturnTypeIds() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_type_ids = true
	}
}

func WithReturnSpecialTokensMask() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_special_tokens_mask = true
	}
}

func WithReturnAttentionMask() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_attention_mask = true
	}
}

func WithReturnOffsets() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_offsets = true
	}
}

func WithReturnCharModeOffsets() EncodeOption {
	return func(eo *EncodeParams) {
		eo.return_offsets = C.bool(true)
		eo.with_offsets_char_mode = C.bool(true)
	}
}

// uint vector to golang slice
func uint32VecToSlice(arrPtr *C.uint32_t, arrLen int) []uint32 {
	slice := make([]uint32, arrLen)
	for i, v := range unsafe.Slice(arrPtr, arrLen) {
		slice[i] = uint32(v)
	}

	return slice
}

type Tokenizer struct {
	tokenizer unsafe.Pointer
}

type TruncationDirection int

const (
	TruncationDirectionLeft TruncationDirection = iota
	TruncationDirectionRight
)

var _ io.Closer = (*Tokenizer)(nil)

func FromBytes(data []byte) (*Tokenizer, error) {
	tokenizer := C.from_bytes((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)))

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func FromFile(path string) (*Tokenizer, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	tokenizer, err := C.from_file(cPath)
	if err != nil {
		return nil, err
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func errorFromCStr(cStr *C.char) error {
	if cStr == nil {
		return nil
	}
	err := errors.New(C.GoString(cStr))
	C.free_string(cStr)
	return err
}

// SetTruncation changes the tokenizer truncation.
// - direction: // 0 -> Left (*); 1 -> Right
// - 0 -> LongestFirst (*), 1 -> OnlyFirst, 2 -> OnlySecond,
func (t *Tokenizer) SetTruncation(
	direction uint8, maxLength uint32, strategy uint8, stride uint32) error {
	params := &C.TruncationParams{
		direction:  C.uint8_t(direction),
		max_length: C.uint32_t(maxLength),
		strategy:   C.uint8_t(strategy),
		stride:     C.uint32_t(stride),
	}
	defer runtime.KeepAlive(t)
	return errorFromCStr(
		C.set_truncation(t.tokenizer, params))
}

// SetNoTruncation changes the tokenizer to not use truncation.
func (t *Tokenizer) SetNoTruncation() error {
	defer runtime.KeepAlive(t)
	return errorFromCStr(
		C.set_truncation(t.tokenizer, nil))
}

// SetPadding changes the tokenizer padding configuration.
// - strategy: 0 -> BatchLongest, >0 -> Fixed to the given value.
// - direction: 0 -> Left (*); 1 -> Right.
func (t *Tokenizer) SetPadding(
	strategy uint32, direction uint8, padToMultipleOf, padId, padTypeId uint32, padToken string) {
	var padTokenCStr *C.char
	if padToken != "" {
		padTokenCStr = C.CString(padToken)
	}
	params := &C.PaddingParams{
		strategy:           C.uint32_t(strategy),        // 0 -> BatchLongest, >0 -> Fixed(value)
		direction:          C.uint8_t(direction),        // 0 -> Left, !=0 -> Right
		pad_to_multiple_of: C.uint32_t(padToMultipleOf), // Disabled if 0.
		pad_id:             C.uint32_t(padId),
		pad_type_id:        C.uint32_t(padTypeId),
		pad_token:          padTokenCStr,
	}
	defer runtime.KeepAlive(t)
	C.set_padding(t.tokenizer, params)
	if padTokenCStr != nil {
		C.free(unsafe.Pointer(padTokenCStr))
	}
}

// SetNoPadding changes the tokenizer not to use padding.
func (t *Tokenizer) SetNoPadding() {
	defer runtime.KeepAlive(t)
	C.set_padding(t.tokenizer, nil)
}

func (t *Tokenizer) Close() error {
	defer runtime.KeepAlive(t)
	C.free_tokenizer(t.tokenizer)
	t.tokenizer = nil
	return nil
}

func (t *Tokenizer) Encode(str string, addSpecialTokens bool, opts ...EncodeOption) (*TokenizerResult, error) {
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	encParams := EncodeParams{add_special_tokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encParams)
	}

	// We expected an EncodedResults with only one result.
	res := C.encode(t.tokenizer, cStr, encParams)
	defer C.free_encode_results(res)
	if res.len != 1 || res.error != nil {
		if res.error != nil {
			return nil, errors.New(C.GoString(res.error))
		} else {
			return nil, errors.Errorf("Tokenizer.Encode failed, got %d results, wanted 1.", res.len)
		}
	}

	encodeResult := &TokenizerResult{}
	t.parseResult(encParams, *res.encoded, encodeResult)
	return encodeResult, nil
}

func (t *Tokenizer) EncodeBatch(strArr []string, addSpecialTokens bool, opts ...EncodeOption) ([]TokenizerResult, error) {
	batchLen := len(strArr)
	if batchLen == 0 {
		return nil, errors.New("empty batch given to EncodeBatch")
	}

	// parse encode options
	encParams := EncodeParams{add_special_tokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encParams)
	}

	// Make string vector to Rust
	cStrings := make([]*C.char, batchLen)
	for i, s := range strArr {
		cStrings[i] = C.CString(s)
	}
	defer func() {
		// release c-char
		for i := range cStrings {
			C.free(unsafe.Pointer(cStrings[i]))
		}
	}()

	// EncodeResults with batchLen results.
	results := C.encode_batch(
		t.tokenizer,
		C.uint32_t(batchLen),
		(**C.char)(unsafe.Pointer(&cStrings[0])),
		encParams,
	)
	defer C.free_encode_results(results)
	if int(results.len) != batchLen || results.error != nil {
		if results.error != nil {
			return nil, errors.New(C.GoString(results.error))
		} else {
			return nil, errors.Errorf("Tokenizer.EncodeBatch failed, got %d results, but batch length given was %d.", results.len, batchLen)
		}
	}
	runtime.KeepAlive(encParams)

	// parse tokenizer encode result
	batchResults := make([]TokenizerResult, batchLen)
	buffers := unsafe.Slice((*C.Buffer)(unsafe.Pointer(results.encoded)), batchLen)
	for ii, buffer := range buffers {
		t.parseResult(encParams, buffer, &batchResults[ii])
	}

	return batchResults, nil
}

// parseResult takes a `*C.Buffer` and copies content to the given `*TokenizerResult`.
// It also requires the `C.EncodeParams` used to encode.
func (t *Tokenizer) parseResult(params C.EncodeParams, buffer C.Buffer, output *TokenizerResult) {
	entryLen := int(buffer.len)

	// Tokens
	if buffer.tokens != nil && params.return_tokens {
		output.Tokens = make([]string, entryLen)
		cStrTokens := unsafe.Slice((**C.char)(unsafe.Pointer(buffer.tokens)), entryLen)
		for j, cStr := range cStrTokens {
			output.Tokens[j] = C.GoString(cStr)
		}
	}

	// TokenIds
	output.TokenIds = uint32VecToSlice(buffer.ids, entryLen)

	// Token offsets
	if params.return_offsets && buffer.offsets != nil {
		output.Offsets = make([]Offset, entryLen)
		cOffsets := (*[1 << 30]C.struct_Offset)(unsafe.Pointer(buffer.offsets))
		for j := 0; j < entryLen; j++ {
			output.Offsets[j] = Offset{
				Start: uint32(cOffsets[j].start),
				End:   uint32(cOffsets[j].end),
			}
		}
	}

	// TypeIds
	if params.return_type_ids && buffer.type_ids != nil {
		output.TypeIds = uint32VecToSlice(buffer.type_ids, entryLen)
	}

	// SpecialTokensMask
	if params.return_special_tokens_mask && buffer.special_tokens_mask != nil {
		output.SpecialTokensMask = uint32VecToSlice(buffer.special_tokens_mask, entryLen)
	}

	// AttentionMask
	if params.return_attention_mask && buffer.attention_mask != nil {
		output.AttentionMask = uint32VecToSlice(buffer.attention_mask, entryLen)
	}
}

func (t *Tokenizer) Decode(tokenIDs []uint32, skipSpecialTokens bool) string {
	if len(tokenIDs) == 0 {
		return ""
	}
	res := C.decode(t.tokenizer, (*C.uint)(unsafe.Pointer(&tokenIDs[0])), C.uint(len(tokenIDs)), C.bool(skipSpecialTokens))
	runtime.KeepAlive(tokenIDs)
	defer C.free_string(res)

	return C.GoString(res)
}

func (t *Tokenizer) VocabSize() uint32 {
	return uint32(C.vocab_size(t.tokenizer))
}
