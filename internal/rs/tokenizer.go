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

type EncodeOptions struct {
	AddSpecialTokens C.bool

	ReturnTypeIds           C.bool
	ReturnSpecialTokensMask C.bool
	ReturnAttentionMask     C.bool
	ReturnOffsets           C.bool

	WithOffsetsCharMode C.bool
}

type EncodeOption func(eo *EncodeOptions)

func WithReturnAll(withCharMode bool) EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnTypeIds = C.bool(true)
		eo.ReturnSpecialTokensMask = C.bool(true)
		eo.ReturnAttentionMask = C.bool(true)
		eo.ReturnOffsets = C.bool(true)
		eo.WithOffsetsCharMode = C.bool(withCharMode)
	}
}

func WithReturnTypeIds() EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnTypeIds = C.bool(true)
	}
}

func WithReturnSpecialTokensMask() EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnSpecialTokensMask = C.bool(true)
	}
}

func WithReturnAttentionMask() EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnAttentionMask = C.bool(true)
	}
}

func WithReturnOffsets() EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnOffsets = C.bool(true)
	}
}

func WithReturnCharModeOffsets() EncodeOption {
	return func(eo *EncodeOptions) {
		eo.ReturnOffsets = C.bool(true)
		eo.WithOffsetsCharMode = C.bool(true)
	}
}

// uint vector to golang slice
func uintVecToSlice(arrPtr *C.uint, arrLen int) []uint32 {
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

func FromBytesWithTruncation(data []byte, maxLen uint32, dir TruncationDirection) (*Tokenizer, error) {
	tokenizer := C.from_bytes_with_truncation((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)), C.uint(maxLen), C.uchar(dir))

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

func (t *Tokenizer) Close() error {
	C.free_tokenizer(t.tokenizer)
	t.tokenizer = nil

	return nil
}

func (t *Tokenizer) Encode(str string, addSpecialTokens bool, opts ...EncodeOption) *TokenizerResult {
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	encOptions := EncodeOptions{AddSpecialTokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encOptions)
	}

	res := C.encode(t.tokenizer, cStr, (*C.struct_EncodeOptions)(unsafe.Pointer(&encOptions)))
	resLen := int(res.len)
	if resLen == 0 {
		return new(TokenizerResult)
	}
	defer C.free_buffer(res)

	encodeResult := &TokenizerResult{Tokens: make([]string, resLen)}
	// TokenIds
	encodeResult.TokenIds = uintVecToSlice(res.ids, resLen)

	// Tokens
	for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:resLen:resLen] {
		encodeResult.Tokens[i] = C.GoString(s)
	}

	// Token offsets
	if encOptions.ReturnOffsets && res.offsets != nil {
		encodeResult.Offsets = make([]Offset, resLen)
		cOffsets := (*[1 << 30]C.struct_Offset)(unsafe.Pointer(res.offsets))
		for i := 0; i < resLen; i++ {
			encodeResult.Offsets[i] = Offset{
				Start: uint32(cOffsets[i].start),
				End:   uint32(cOffsets[i].end),
			}
		}
	}

	// TypeIds
	if encOptions.ReturnTypeIds && res.type_ids != nil {
		encodeResult.TypeIds = uintVecToSlice(res.type_ids, resLen)
	}

	// SpecialTokensMask
	if encOptions.ReturnSpecialTokensMask && res.special_tokens_mask != nil {
		encodeResult.SpecialTokensMask = uintVecToSlice(res.special_tokens_mask, resLen)
	}

	// AttentionMask
	if encOptions.ReturnAttentionMask && res.attention_mask != nil {
		encodeResult.AttentionMask = uintVecToSlice(res.attention_mask, resLen)
	}

	return encodeResult
}

func (t *Tokenizer) EncodeBatch(strArr []string, addSpecialTokens bool, opts ...EncodeOption) []*TokenizerResult {
	batchLen := len(strArr)
	if batchLen == 0 {
		return nil
	}

	// parse encode options
	encOptions := EncodeOptions{AddSpecialTokens: C.bool(addSpecialTokens)}
	for _, opt := range opts {
		opt(&encOptions)
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

	// call encode batch
	batchRes := C.encode_batch(
		t.tokenizer,
		(**C.char)(unsafe.Pointer(&cStrings[0])),
		(*C.struct_EncodeOptions)(unsafe.Pointer(&encOptions)),
	)
	runtime.KeepAlive(encOptions)

	// parse tokenizer encode result
	batchResult := make([]*TokenizerResult, batchLen)
	if batchLen > 0 {
		defer C.free_batch_buffer(batchRes)
	}
	for i, encodeResult := range (*[1 << 30]C.struct_Buffer)(unsafe.Pointer(batchRes))[:batchLen:batchLen] {
		subResLen := int(encodeResult.len)
		subTokenizerResult := &TokenizerResult{Tokens: make([]string, subResLen)}

		// TokenIds
		subTokenizerResult.TokenIds = uintVecToSlice(encodeResult.ids, subResLen)

		// Tokens
		for j, s := range (*[1 << 30]*C.char)(unsafe.Pointer(encodeResult.tokens))[:subResLen:subResLen] {
			subTokenizerResult.Tokens[j] = C.GoString(s)
		}

		// Token offsets
		if encOptions.ReturnOffsets && encodeResult.offsets != nil {
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
		if encOptions.ReturnTypeIds && encodeResult.type_ids != nil {
			subTokenizerResult.TypeIds = uintVecToSlice(encodeResult.type_ids, subResLen)
		}

		// SpecialTokensMask
		if encOptions.ReturnSpecialTokensMask && encodeResult.special_tokens_mask != nil {
			subTokenizerResult.SpecialTokensMask = uintVecToSlice(encodeResult.special_tokens_mask, subResLen)
		}

		// AttentionMask
		if encOptions.ReturnAttentionMask && encodeResult.attention_mask != nil {
			subTokenizerResult.AttentionMask = uintVecToSlice(encodeResult.attention_mask, subResLen)
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
