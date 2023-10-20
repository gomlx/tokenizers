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
			return_special_tokens_mask: true,
			return_attention_mask:      true,
			return_offsets:             true,
			with_offsets_char_mode:     C.bool(withCharMode),
		}
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

func (t *Tokenizer) Encode(str string, addSpecialTokens bool, opts ...EncodeOption) *TokenizerResult {
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	encParams := EncodeParams{add_special_tokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encParams)
	}

	// We expected an EncodedResults with only one result.
	res := C.encode(t.tokenizer, cStr, encParams)
	defer C.free_encode_results(res)
	if res.len != 1 || res.encoded.len == 0 {
		return new(TokenizerResult)
	}
	buf := *res.encoded
	bufLen := int(buf.len)
	encodeResult := &TokenizerResult{Tokens: make([]string, bufLen)}
	// TokenIds
	encodeResult.TokenIds = uint32VecToSlice(buf.ids, bufLen)

	// Tokens
	for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(buf.tokens))[:bufLen:bufLen] {
		encodeResult.Tokens[i] = C.GoString(s)
	}

	// Token offsets
	if encParams.return_offsets && buf.offsets != nil {
		encodeResult.Offsets = make([]Offset, bufLen)
		cOffsets := (*[1 << 30]C.struct_Offset)(unsafe.Pointer(buf.offsets))
		for i := 0; i < bufLen; i++ {
			encodeResult.Offsets[i] = Offset{
				Start: uint32(cOffsets[i].start),
				End:   uint32(cOffsets[i].end),
			}
		}
	}

	// TypeIds
	if encParams.return_type_ids && buf.type_ids != nil {
		encodeResult.TypeIds = uint32VecToSlice(buf.type_ids, bufLen)
	}

	// SpecialTokensMask
	if encParams.return_special_tokens_mask && buf.special_tokens_mask != nil {
		encodeResult.SpecialTokensMask = uint32VecToSlice(buf.special_tokens_mask, bufLen)
	}

	// AttentionMask
	if encParams.return_attention_mask && buf.attention_mask != nil {
		encodeResult.AttentionMask = uint32VecToSlice(buf.attention_mask, bufLen)
	}

	return encodeResult
}

func (t *Tokenizer) EncodeBatch(strArr []string, addSpecialTokens bool, opts ...EncodeOption) []*TokenizerResult {
	batchLen := len(strArr)
	if batchLen == 0 {
		return nil
	}

	// parse encode options
	encParams := EncodeParams{add_special_tokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encParams)
	}

	// Make string vector to Rust
	cStrings := make([]*C.char, len(strArr))
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
		(**C.char)(unsafe.Pointer(&cStrings[0])),
		encParams,
	)
	defer C.free_encode_results(results)
	if int(results.len) != batchLen {
		// Something went wrong. TODO: add errors.
		return nil
	}
	runtime.KeepAlive(encParams)

	// parse tokenizer encode result
	batchResult := make([]*TokenizerResult, batchLen)
	buffers := unsafe.Slice((*C.Buffer)(unsafe.Pointer(results.encoded)), batchLen)
	for i, encodeResult := range buffers {
		subResLen := int(encodeResult.len)
		subTokenizerResult := &TokenizerResult{Tokens: make([]string, subResLen)}

		// TokenIds
		subTokenizerResult.TokenIds = uint32VecToSlice(encodeResult.ids, subResLen)

		// Tokens
		for j, s := range (*[1 << 30]*C.char)(unsafe.Pointer(encodeResult.tokens))[:subResLen:subResLen] {
			subTokenizerResult.Tokens[j] = C.GoString(s)
		}

		// Token offsets
		if encParams.return_offsets && encodeResult.offsets != nil {
			subTokenizerResult.Offsets = make([]Offset, subResLen)
			cOffsets := (*[1 << 30]C.struct_Offset)(unsafe.Pointer(encodeResult.offsets))
			for j := 0; j < subResLen; j++ {
				subTokenizerResult.Offsets[j] = Offset{
					Start: uint32(cOffsets[j].start),
					End:   uint32(cOffsets[j].end),
				}
			}
		}

		// TypeIds
		if encParams.return_type_ids && encodeResult.type_ids != nil {
			subTokenizerResult.TypeIds = uint32VecToSlice(encodeResult.type_ids, subResLen)
		}

		// SpecialTokensMask
		if encParams.return_special_tokens_mask && encodeResult.special_tokens_mask != nil {
			subTokenizerResult.SpecialTokensMask = uint32VecToSlice(encodeResult.special_tokens_mask, subResLen)
		}

		// AttentionMask
		if encParams.return_attention_mask && encodeResult.attention_mask != nil {
			subTokenizerResult.AttentionMask = uint32VecToSlice(encodeResult.attention_mask, subResLen)
		}

		batchResult[i] = subTokenizerResult
	}

	return batchResult
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
