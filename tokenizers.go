// Package tokenizers provides an implementation of today's most used tokenizers, with a focus on performance and versatility.
//
// It is currently a wrapper around the Rust implementation in
// https://github.com/huggingface/tokenizers/tree/main/tokenizers.
//
// For now, it only provides the encoding and decoding functionality -- not training a new tokenizers.
// It includes reading from [HuggingFace's pretrained tokenizers](https://huggingface.co/docs/tokenizers/index)
// using `FromPretrained`.
package tokenizers

import "C"
import (
	"fmt"
	"github.com/gomlx/tokenizers/internal/rs"
	"github.com/pkg/errors"
	"os"
	"strings"
)

// Tokenizer represents an initialized Tokenizer, including various configurations
// for truncation, padding, and how to encode.
//
// It can be used to encode (`Encode` and `EncodeBatch`) strings to token ids and other optional fields,
// and to decode (`Decode` and `DecodeBatch`) token ids back to strings.
//
// To build a new Tokenizer from a JSon configuration, see `FromFile` or `FromBytes`.
// To automatically load the JSon configuration from HuggingFace, use `FromPretrained`.
type Tokenizer struct {
	tokenizer *rs.Tokenizer

	encodeParams                  rs.EncodeParams
	isTruncationSet, isPaddingSet bool

	// All of these are only valid if `isTruncationSet` is true.
	truncationDirection                   Direction
	truncationMaxLength, truncationStride uint32
	truncationStrategy                    TruncationStrategy

	// All of these are only valid if `isPaddingSet` is true.
	paddingDirection                                 Direction
	paddingStrategy                                  PaddingStrategy
	paddingLength, padToMultipleOf, padId, padTypeId uint32
	padToken                                         string
}

// Direction is used in truncation and padding configuration.
type Direction uint8

const (
	Left  Direction = 0
	Right Direction = 1
)

// TruncationStrategy generally affects how truncation is applied and the inputs are pairs of
// sentences. It is very dependent on the truncation model used, and usually set by the preloaded
// tokenization model.
type TruncationStrategy uint8 // Values must match the underlying Rust library.

const (
	TruncateLongestFirst TruncationStrategy = iota
	TruncateOnlyFirst
	TruncateOnlySecond
)

// PaddingStrategy usually is defined by the preloaded tokenization model (since it should match
// the LLM model). But it can be manipulated.
//
// It can be set to PadLongest, which pads the tokenization to the longest sequence in the batch,
// or PadFixed when it pads to a fixed length.
type PaddingStrategy uint8 // Values must match the underlying Rust library.

const (
	PadLongest PaddingStrategy = iota
	PadFixed
)

// OffsetsCharMode defines how to encode the offset positions when encoding.
// - `OffsetsCharModeByte`: Offsets are calculated on a byte basis.
// - `OffsetsCharModeUnicode` (default): Offsets are calculated on a Unicode code point basis.
type OffsetsCharMode uint8

const (
	OffsetsCharModeByte    OffsetsCharMode = 0
	OffsetsCharModeUnicode OffsetsCharMode = 1
)

//go:generate stringer -type=Direction,TruncationStrategy,PaddingStrategy,OffsetsCharMode -output=types_string.go .

// panicf generates an error message and panics with it, in one function.
func panicf(format string, args ...any) {
	err := errors.Errorf(format, args...)
	panic(err)
}

// FromFile creates a Tokenizer from the tokenizer model stored as JSon in filePath.
// It is the same format as [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers).
func FromFile(filePath string) (*Tokenizer, error) {
	contents, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrap(err, "can't read tokenizer file:")
	}
	return FromBytes(contents)
}

// FromBytes is the same as FromFile, but instead takes the JSon `data` and returns a Tokenizer,
// or an error.
// It is the same format as [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers).
func FromBytes(data []byte) (*Tokenizer, error) {
	t := &Tokenizer{}
	var err error
	t.setDefaultEncodeParams()

	t.tokenizer, err = rs.FromBytes(data)
	if err != nil {
		return nil, errors.WithMessage(err, "Tokenizer.FromBytes(<json-data>):")
	}

	// Parse truncation and padding:
	var direction, truncStrategy uint8
	t.isTruncationSet, direction, t.truncationMaxLength, truncStrategy, t.truncationStride = t.tokenizer.GetTruncation()
	t.truncationDirection = Direction(direction)
	t.truncationStrategy = TruncationStrategy(truncStrategy)
	if !t.isTruncationSet {
		t.setDefaultTruncation() // Not used, but it's safe to reset to the default.
	}

	var padStrategy uint32
	t.isPaddingSet, padStrategy, direction, t.padToMultipleOf, t.padId, t.padTypeId, t.padToken = t.tokenizer.GetPadding()
	t.paddingDirection = Direction(direction)
	if padStrategy == 0 {
		t.paddingStrategy = PadLongest
	} else {
		t.paddingStrategy = PadFixed
		t.paddingLength = padStrategy
	}
	if !t.isPaddingSet {
		t.setDefaultPadding() // Not used, but it's safe to reset to the default.
	}

	return t, nil
}

// Finalize is optional, and will release immediately the memory associated with the Tokenizer, not waiting for the
// garbage collection.
// After calling this function, the Tokenizer is no longer valid, and any calls to it will panic.
func (t *Tokenizer) Finalize() {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.tokenizer.Finalize()
	t.tokenizer = nil
}

// String implements fmt.Stringer.
func (t *Tokenizer) String() string {
	if t.tokenizer == nil {
		return "nil"
	}
	var parts []string
	parts = append(parts, fmt.Sprintf("  Truncation: IsTruncationSet=%v", t.isTruncationSet))
	parts = append(parts, fmt.Sprintf("    TruncationDirection=%v", t.truncationDirection))
	parts = append(parts, fmt.Sprintf("    TruncationMaxLength=%v", t.truncationMaxLength))
	parts = append(parts, fmt.Sprintf("    TruncationStride=%v", t.truncationStride))
	parts = append(parts, fmt.Sprintf("    TruncationStrategy=%v", t.truncationStrategy))
	parts = append(parts, fmt.Sprintf("  Padding: IsPaddingSet=%v", t.isPaddingSet))
	parts = append(parts, fmt.Sprintf("    PaddingDirection=%v", t.paddingDirection))
	parts = append(parts, fmt.Sprintf("    PaddingStrategy=%v", t.paddingStrategy))
	parts = append(parts, fmt.Sprintf("    PaddingLength=%v", t.paddingLength))
	parts = append(parts, fmt.Sprintf("    PadToMultipleOf=%v", t.padToMultipleOf))
	parts = append(parts, fmt.Sprintf("    PadId=%v", t.padId))
	parts = append(parts, fmt.Sprintf("    PadTypeId=%v", t.padTypeId))
	parts = append(parts, fmt.Sprintf("    PadToken=%q", t.padToken))
	parts = append(parts, "  Encode Parameters:")
	parts = append(parts, fmt.Sprintf("    AddSpecialTokens=%v", t.encodeParams.AddSpecialTokens))
	parts = append(parts, fmt.Sprintf("    ReturnTokens=%v", t.encodeParams.ReturnTokens))
	parts = append(parts, fmt.Sprintf("    ReturnTypeIds=%v", t.encodeParams.ReturnTypeIds))
	parts = append(parts, fmt.Sprintf("    ReturnSpecialTokensMask=%v", t.encodeParams.ReturnSpecialTokensMask))
	parts = append(parts, fmt.Sprintf("    ReturnAttentionMask=%v", t.encodeParams.ReturnAttentionMask))
	parts = append(parts, fmt.Sprintf("    ReturnOffsets=%v", t.encodeParams.ReturnOffsets))
	var offsetCharMode OffsetsCharMode
	if t.encodeParams.WithOffsetsCharMode {
		offsetCharMode = OffsetsCharModeUnicode
	}
	parts = append(parts, fmt.Sprintf("    WithOffsetsCharMode=%s", offsetCharMode))
	return fmt.Sprintf("Tokenizer(\n%s\n)\n", strings.Join(parts, "\n"))
}

// setTruncation updates the underlying (Rust) truncation parameters according to parameters set.
// This is needed because they are configured as a block, while the Go API uses a fine-grained approach.
// It panics on error -- only happens with invalid parameters.
func (t *Tokenizer) setTruncation() {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	var err error
	if !t.isTruncationSet {
		err = t.tokenizer.SetNoTruncation()
		if err != nil {
			err = errors.WithMessage(err, "while disabling truncation:")
			panic(err)
		}
		return
	}

	err = t.tokenizer.SetTruncation(uint8(t.truncationDirection), t.truncationMaxLength, uint8(t.truncationStrategy), t.truncationStride)
	if err != nil {
		err = errors.WithMessage(err, "while disabling truncation:")
		panic(err)
	}
}

// setDefaultTruncation sets the default values of truncation.
func (t *Tokenizer) setDefaultTruncation() {
	t.truncationDirection = Left
	t.truncationStrategy = TruncateLongestFirst
	t.truncationMaxLength = 512
	t.truncationStride = 0
}

// WithTruncation enables truncation and changes the truncation to the given length.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithTruncation(length int) *Tokenizer {
	if length <= 0 {
		panicf("Tokenizer.WithTruncation(length=%d): length must be > 0", length)
	}
	t.isTruncationSet = true
	t.truncationMaxLength = uint32(length)
	t.setTruncation()
	return t
}

// WithTruncationStrategy enables truncation (if not already) and sets the truncation strategy.
// This affects how truncation behaves when encoding sentence pairs, and
// is usually defined by the tokenization model that is loaded, and not directly by the user.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithTruncationStrategy(strategy TruncationStrategy) *Tokenizer {
	t.isTruncationSet = true
	t.truncationStrategy = strategy
	t.setTruncation()
	return t
}

// WithTruncationStride enables truncation (if not already) and sets the truncation stride.
// From HuggingFace: "The length of the previous first sequence to be included in the overflowing sequence",
// but I'm not sure what they mean with that.
//
// This is usually defined by the tokenization model that is loaded, and not directly by the user.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithTruncationStride(stride int) *Tokenizer {
	if stride < 0 {
		panicf("Tokenizer.WithTruncationStride(stride=%d): stride must be >= 0", stride)
	}
	t.isTruncationSet = true
	t.truncationStride = uint32(stride)
	t.setTruncation()
	return t
}

// WithTruncationDirection enables truncation (if not already) and sets the truncation to happen in the given direction.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithTruncationDirection(direction Direction) *Tokenizer {
	t.isTruncationSet = true
	t.truncationDirection = direction
	t.setTruncation()
	return t
}

// WithNoTruncation disables truncation and resets all truncation parameters.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) WithNoTruncation() *Tokenizer {
	t.isTruncationSet = false
	t.setDefaultTruncation()
	t.setTruncation()
	return t
}

// setPadding updates the underlying (Rust) padding parameters according to the parameters set.
// This is needed because they are configured as a block, while the Go API uses a fine-grained approach.
// It panics on error -- only happens with invalid parameters.
func (t *Tokenizer) setPadding() {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	if !t.isPaddingSet {
		t.tokenizer.SetNoPadding()
		return
	}
	var strategy uint32
	if t.paddingStrategy != PadLongest {
		if t.paddingLength == 0 {
			panicf("Tokenizer.setPadding() with strategy PadFixed, but length of 0 is invalid")
		}
		strategy = t.paddingLength
	}
	t.tokenizer.SetPadding(strategy, uint8(t.paddingDirection), t.padToMultipleOf, t.padId, t.padTypeId, t.padToken)
}

// setDefaultPadding sets the default values for padding.
func (t *Tokenizer) setDefaultPadding() {
	t.paddingStrategy = PadLongest
	t.paddingDirection = Right
	t.paddingLength = 0 // Ignored with PadLongest.
	t.padId = 0
	t.padTypeId = 0
	t.padToMultipleOf = 0
	t.padToken = "[PAD]"
}

// WithPadToLongest enables padding (if not already) and sets the padding to the longest sequence in the batch.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length <= 0).
func (t *Tokenizer) WithPadToLongest() *Tokenizer {
	t.isPaddingSet = true
	t.paddingStrategy = PadLongest
	t.paddingLength = 0
	t.setPadding()
	return t
}

// WithPadToLength enables padding (if not already) and sets the padding to the fixed given length.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length == 0).
func (t *Tokenizer) WithPadToLength(length uint32) *Tokenizer {
	t.isPaddingSet = true
	t.paddingStrategy = PadFixed
	t.paddingLength = length
	t.setPadding()
	return t
}

// WithPadId enables padding (if not already) and sets the id of the token to use for padding.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length == 0).
func (t *Tokenizer) WithPadId(id uint32) *Tokenizer {
	t.isPaddingSet = true
	t.padId = id
	t.setPadding()
	return t
}

// WithPadTypeId enables padding (if not already) and sets the type id of the token to use for padding.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length == 0).
func (t *Tokenizer) WithPadTypeId(typeId uint32) *Tokenizer {
	t.isPaddingSet = true
	t.padTypeId = typeId
	t.setPadding()
	return t
}

// WithPadToken enables padding (if not already) and sets the token to use for padding.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length == 0).
func (t *Tokenizer) WithPadToken(token string) *Tokenizer {
	t.isPaddingSet = true
	t.padToken = token
	t.setPadding()
	return t
}

// WithPaddingToMultipleOf enables padding (if not already) and sets the multiple of value.
// If specified, the padding length should always snap to the next multiple of the given value.
// For example, if we were going to pad with a length of 250 but pad_to_multiple_of=8 then we will pad to 256.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (e.g.: if padding length == 0).
func (t *Tokenizer) WithPaddingToMultipleOf(multiple uint32) *Tokenizer {
	t.isPaddingSet = true
	t.padToMultipleOf = multiple
	t.setPadding()
	return t
}

// WithPaddingDirection enables padding (if not already) and sets the padding to happen in the given direction.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithPaddingDirection(direction Direction) *Tokenizer {
	t.isPaddingSet = true
	t.paddingDirection = direction
	t.setPadding()
	return t
}

// WithNoPadding disables padding and resets all padding parameters.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
//
// It may panic is an invalid value is used (negative length, etc.).
func (t *Tokenizer) WithNoPadding() *Tokenizer {
	t.isPaddingSet = false
	t.setDefaultPadding()
	t.setPadding()
	return t
}

func (t *Tokenizer) setDefaultEncodeParams() {
	t.encodeParams = rs.EncodeParams{
		AddSpecialTokens:        false,
		ReturnTokens:            true,
		ReturnTypeIds:           false,
		ReturnSpecialTokensMask: false,
		ReturnAttentionMask:     false,
		ReturnOffsets:           false,
		WithOffsetsCharMode:     true, // == OffsetsCharModeUnicode
	}
}

// AddSpecialTokens sets whether Encode (and EncodeBatch) should add the special tokens (start and end of sentence,
// etc.).
// Default is false.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) AddSpecialTokens(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.AddSpecialTokens = value
	return t
}

// ReturnTokens sets whether Encode (and EncodeBatch) should also return the textual tokens separated.
// Default is true.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) ReturnTokens(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.ReturnTokens = value
	return t
}

// ReturnTypeIds sets whether Encode (and EncodeBatch) should also return the token IDs.
// Default is false.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) ReturnTypeIds(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.ReturnTypeIds = value
	return t
}

// ReturnSpecialTokensMask sets whether Encode (and EncodeBatch) should also return a special tokens mask.
// The special tokens mask is a binary vector indicating whether each token is a special token (e.g., padding, CLS, SEP).
// Default is false.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) ReturnSpecialTokensMask(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.ReturnSpecialTokensMask = value
	return t
}

// ReturnAttentionMask sets whether Encode (and EncodeBatch) should also return an attention mask.
// The attention mask is a binary matrix indicating which tokens can attend to each other.
// It is used in transformer models to prevent the model from attending to padding tokens.
// Default is false.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) ReturnAttentionMask(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.ReturnAttentionMask = value
	return t
}

// ReturnOffsets sets whether Encode (and EncodeBatch) should also return the byte offsets of the tokens in the original text.
// Default is false.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) ReturnOffsets(value bool) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.ReturnOffsets = value
	return t
}

// WithOffsetsCharMode sets the character-level offset mode for the token offsets.
// The possible values are:
//
// - `OffsetsCharModeByte`: Offsets are calculated on a byte basis.
// - `OffsetsCharModeUnicode` (default): Offsets are calculated on a Unicode code point basis.
//
// Notice that to enable returning of the offsets you need to configure `t.ReturnOffsets(true)`.
//
// It returns itself (the Tokenizer), to allow cascaded configuration calls.
func (t *Tokenizer) WithOffsetsCharMode(value OffsetsCharMode) *Tokenizer {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	t.encodeParams.WithOffsetsCharMode = value == OffsetsCharModeUnicode
	return t
}

// Encoding is the result of a Tokenizer.Encode.
//
// Only TokenIds is always present, all other fields
// are only set if configured in the Tokenizer.
//
// The SpecialTokensMask indicates which tokens are special tokens (e.g., padding, CLS, SEP).
//
// The AttentionMask indicates which tokens are padding and should be ignored.
type Encoding = rs.Encoding

// Encode given sentence.
//
// The returned Encoding object will have fields filled according to Tokenizer fields configured to be returned.
func (t *Tokenizer) Encode(sentence string) (*Encoding, error) {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	return t.tokenizer.Encode(sentence, t.encodeParams)
}

// EncodeBatch list of strings.
//
// The returned Encoding object will have fields filled according to Tokenizer fields configured to be returned.
func (t *Tokenizer) EncodeBatch(sentences []string) ([]Encoding, error) {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	return t.tokenizer.EncodeBatch(sentences, t.encodeParams)
}

// Decode is the reverse of encode, and converts the list of tokens back to a "sentence" (string).
func (t *Tokenizer) Decode(tokenIds []uint32, skipSpecialTokens bool) string {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	if len(tokenIds) == 0 {
		return ""
	}
	return t.tokenizer.Decode(tokenIds, skipSpecialTokens)
}

// VocabSize returns the number of known tokens.
func (t *Tokenizer) VocabSize() uint32 {
	if t.tokenizer == nil {
		panicf("Tokenizer already finalized, one cannot change or use it any longer")
	}
	return t.tokenizer.VocabSize()
}
