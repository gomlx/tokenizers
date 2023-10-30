package rs_test

import (
	_ "embed"
	"runtime"
	"testing"

	"github.com/gomlx/tokenizers/internal/rs"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed test-sentence-transformers-labse.json
var embeddedBytes []byte

const (
	bertJson        = "../../examples/bert/bert-base-uncased.json"
	bertPaddingJson = "../../examples/bert/bert-base-uncased-padding.json"
)

// TODO test for leaks

func TestInvalidConfigPath(t *testing.T) {
	_, err := rs.FromFile("./non-existent.json")
	require.Error(t, err)
}

func TestInvalidBytes(t *testing.T) {
	contents := "I_am_not_json"
	tk, err := rs.FromBytes([]byte(contents))
	assert.Error(t, err)
	if err == nil {
		tk.Finalize()
	}
}

func TestEmbeddingConfig(t *testing.T) {
	tk, err := rs.FromBytes(embeddedBytes)
	require.NoError(t, err)
	defer tk.Finalize()

	tests := []struct {
		name       string
		str        string
		addSpecial bool
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "without special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			wantIDs:    []uint32{0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:       "with special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			wantIDs:    []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4, 0x66},
			wantTokens: []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			encodeRes, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)

			encParams.ReturnTokens = false
			encodeRes, err = tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Empty(t, encodeRes.Tokens)
		})
	}
}

func TestEncode(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	tests := []struct {
		name       string
		str        string
		addSpecial bool
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "without special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			wantIDs:    []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:       "with special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			wantIDs:    []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			wantTokens: []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"},
		},
		{
			name:       "empty string",
			str:        "",
			wantIDs:    []uint32{},
			wantTokens: []string{},
			addSpecial: false,
		},
		{
			name:       "empty string with special tokens",
			str:        "",
			wantIDs:    []uint32{101, 102},
			wantTokens: []string{"[CLS]", "[SEP]"},
			addSpecial: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			encodeRes, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}

	// Checks that the tokenizer is properly finalized.
	tk.Finalize()
	assert.Equal(t, int64(0), rs.CountTokenizerAllocs.Load())
}

func TestEncodeOptions(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Finalize()
	tests := []struct {
		name                  string
		str                   string
		addSpecial            bool
		wantIDs               []uint32
		wantTypeIDs           []uint32
		wantTokens            []string
		wantSpecialTokensMask []uint32
		wantAttentionMask     []uint32
	}{
		{
			name:                  "without special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            false,
			wantIDs:               []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
			wantSpecialTokensMask: []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			encoding, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encParams.ReturnTypeIds = true
			encoding, err = tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encParams = rs.EncodeParams{
				AddSpecialTokens:        tt.addSpecial,
				ReturnTokens:            true,
				ReturnSpecialTokensMask: true,
			}
			encoding, err = tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encParams = rs.EncodeParams{
				AddSpecialTokens:    tt.addSpecial,
				ReturnAttentionMask: true,
			}
			encoding, err = tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "no tokens requested, should be empty")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")

			encParams = rs.ReturnAll(tt.addSpecial, false)
			encoding, err = tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")
		})
	}
}

func TestEncodeOffsets(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Finalize()

	encParams := rs.EncodeParams{
		AddSpecialTokens: false,
		ReturnOffsets:    true,
	}
	encodeRes, err := tk.Encode("Ohne UTF-8, ist alles Käse!", encParams)
	require.NoError(t, err)
	expected := []rs.Offset{
		rs.Offset{Start: 0, End: 2},
		rs.Offset{Start: 2, End: 4},
		rs.Offset{Start: 5, End: 7},
		rs.Offset{Start: 7, End: 8},
		rs.Offset{Start: 8, End: 9},
		rs.Offset{Start: 9, End: 10},
		rs.Offset{Start: 10, End: 11},
		rs.Offset{Start: 12, End: 15},
		rs.Offset{Start: 16, End: 19},
		rs.Offset{Start: 19, End: 21},
		rs.Offset{Start: 22, End: 25},
		rs.Offset{Start: 25, End: 27},
		rs.Offset{Start: 27, End: 28}}
	assert.Equal(t, encodeRes.Offsets, expected)

	encParams.WithOffsetsCharMode = true
	encodeResV1, err := tk.Encode("Ohne UTF-8, ist alles Käse!", encParams)
	require.NoError(t, err)
	expectedV1 := []rs.Offset{
		rs.Offset{Start: 0x0, End: 0x2},
		rs.Offset{Start: 0x2, End: 0x4},
		rs.Offset{Start: 0x5, End: 0x7},
		rs.Offset{Start: 0x7, End: 0x8},
		rs.Offset{Start: 0x8, End: 0x9},
		rs.Offset{Start: 0x9, End: 0xa},
		rs.Offset{Start: 0xa, End: 0xb},
		rs.Offset{Start: 0xc, End: 0xf},
		rs.Offset{Start: 0x10, End: 0x13},
		rs.Offset{Start: 0x13, End: 0x15},
		rs.Offset{Start: 0x16, End: 0x18},
		rs.Offset{Start: 24, End: 26},
		rs.Offset{Start: 26, End: 27}}
	assert.Equal(t, encodeResV1.Offsets, expectedV1)
}

func TestEncodeBatch(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Finalize()

	tests := []struct {
		name       string
		str        string
		addSpecial bool
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "without special tokens-1",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			wantIDs:    []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:       "with special tokens-1",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			wantIDs:    []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			wantTokens: []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			results, err := tk.EncodeBatch([]string{tt.str, tt.str}, encParams)
			require.NoError(t, err)
			require.Equal(t, 2, len(results))
			for ii, res := range results {
				assert.Equalf(t, tt.wantIDs, res.TokenIds, "Sentence %d", ii)
				assert.Equalf(t, tt.wantTokens, res.Tokens, "Sentence %d", ii)
			}
		})
	}
}

// TestEncodeWithTruncation tests truncation, but it's also used to verify that GC is properly finalizing
// the Tokenizers.
func TestEncodeWithTruncation(t *testing.T) {
	tests := []struct {
		name       string
		str        string
		addSpecial bool
		maxLen     int
		dir        rs.TruncationDirection
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "without special tokens, left truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			maxLen:     5,
			dir:        rs.TruncationDirectionLeft,
			wantIDs:    []uint32{0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
			wantTokens: []string{"jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:       "without special tokens, right truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			maxLen:     5,
			dir:        rs.TruncationDirectionRight,
			wantIDs:    []uint32{0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the"},
		},
		{
			name:       "with special tokens, left truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			maxLen:     5,
			dir:        rs.TruncationDirectionLeft,
			wantIDs:    []uint32{0x65, 0x3a89, 0x35fc3, 0x57b4, 0x66},
			wantTokens: []string{"[CLS]", "the", "lazy", "dog", "[SEP]"},
		},
		{
			name:       "with special tokens, right truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			maxLen:     5,
			dir:        rs.TruncationDirectionRight,
			wantIDs:    []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x66},
			wantTokens: []string{"[CLS]", "brown", "fox", "jumps", "[SEP]"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tk, err := rs.FromBytes(embeddedBytes)
			require.NoError(t, err)

			isSet, _, _, _, _ := tk.GetTruncation()
			assert.False(t, isSet)

			err = tk.SetTruncation(uint8(tt.dir), uint32(tt.maxLen), 0, 0)
			require.NoError(t, err)

			isSet, direction, maxLength, strategy, stride := tk.GetTruncation()
			assert.True(t, isSet)
			assert.Equal(t, uint8(tt.dir), direction)
			assert.Equal(t, uint32(tt.maxLen), maxLength)
			assert.Equal(t, uint8(0), strategy)
			assert.Equal(t, uint32(0), stride)

			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			encodeRes, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)

			// Checks reset of truncation.
			err = tk.SetNoTruncation()
			require.NoError(t, err)
			isSet, _, _, _, _ = tk.GetTruncation()
			assert.False(t, isSet)

		})
	}

	// Check tokenizers are properly finalized.
	for ii := 0; ii < 3; ii++ {
		runtime.GC()
	}
	assert.Equal(t, int64(0), rs.CountTokenizerAllocs.Load())
}

func TestEncodeWithPadding(t *testing.T) {
	tests := []struct {
		name            string
		str             string
		addSpecial      bool
		padLen          uint32
		dir             uint8 // 0 -> Left, 1 -> Right
		padId           uint32
		padToken        string
		padToMultipleOf uint32
		wantIDs         []uint32
		wantTokens      []string
	}{
		{
			name:            "without special tokens, left padding",
			str:             "brown fox jumps over the lazy dog",
			addSpecial:      false,
			padLen:          8,
			dir:             0,
			padId:           0,
			padToken:        "[PAD]",
			padToMultipleOf: 0,
			wantIDs:         []uint32{0x0, 0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
			wantTokens:      []string{"[PAD]", "brown", "fox", "jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:            "with special tokens, right padding, multiple of 16",
			str:             "brown fox jumps over the lazy dog",
			addSpecial:      true,
			padLen:          0,
			dir:             1,
			padId:           0,
			padToken:        "",
			padToMultipleOf: 4, // Since it tokenizes to 9 elements, it will pad to 12 (next multiple of 4).
			wantIDs:         []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4, 0x66, 0x0, 0x0, 0x0},
			wantTokens:      []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]", "", "", ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tk, err := rs.FromBytes(embeddedBytes)
			require.NoError(t, err)
			isSet, _, _, _, _, _, _ := tk.GetPadding()
			assert.False(t, isSet)

			defer tk.Finalize()
			tk.SetPadding(tt.padLen, tt.dir, tt.padToMultipleOf, tt.padId, 0, tt.padToken)
			isSet, strategy, direction, padToMultipleOf, padId, padTypeId, padToken := tk.GetPadding()
			assert.Equal(t, tt.padLen, strategy)
			assert.Equal(t, tt.dir, direction)
			assert.Equal(t, tt.padToMultipleOf, padToMultipleOf)
			assert.Equal(t, tt.padId, padId)
			assert.Equal(t, uint32(0), padTypeId)
			assert.Equal(t, tt.padToken, padToken)

			encParams := rs.EncodeParams{
				AddSpecialTokens: tt.addSpecial,
				ReturnTokens:     true,
			}
			encodeRes, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)

			// Checks reset of padding.
			tk.SetNoPadding()
			isSet, _, _, _, _, _, _ = tk.GetPadding()
			assert.False(t, isSet)
		})
	}
}

func TestEncodeWithPaddingBert(t *testing.T) {
	tests := []struct {
		name       string
		str        string
		addSpecial bool
		maxLen     int
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "padding len 10",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			maxLen:     10,
			wantIDs:    []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899, 0, 0, 0},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog", "", "", ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tk, err := rs.FromFile(bertPaddingJson) // Padding pre-configured in Json file.
			require.NoError(t, err)
			defer tk.Finalize()

			encParams := rs.ReturnAll(false, false)
			encodeRes, err := tk.Encode(tt.str, encParams)
			require.NoError(t, err)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}
}

func TestDecode(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Finalize()
	tests := []struct {
		name        string
		tokens      []uint32
		skipSpecial bool
		want        string
	}{
		{
			name:        "without special tokens, skip special tokens",
			tokens:      []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			skipSpecial: true,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "with special tokens, skip special tokens",
			tokens:      []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			skipSpecial: true,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "without special tokens, don't skip special tokens",
			tokens:      []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			skipSpecial: false,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "with special tokens, don't skip special tokens",
			tokens:      []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			skipSpecial: false,
			want:        "[CLS] brown fox jumps over the lazy dog [SEP]",
		},
		{
			name:        "no tokens",
			tokens:      []uint32{},
			skipSpecial: false,
			want:        "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tk.Decode(tt.tokens, tt.skipSpecial)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestVocabSize(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Finalize()
	assert.Equal(t, uint32(30522), tk.VocabSize())
}

func BenchmarkEncodeNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Finalize()
	expected := []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}
	b.ResetTimer()
	encParams := rs.EncodeParams{
		AddSpecialTokens: false,
		ReturnTokens:     true,
	}
	for i := 0; i < b.N; i++ {
		encodeRes, err := tk.Encode("brown fox jumps over the lazy dog", encParams)
		if err != nil {
			require.NoError(b, err)
		}
		assert.Equal(b, expected, encodeRes.TokenIds)
	}
}

func BenchmarkEncodeWithOptionNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Finalize()
	encParams := rs.ReturnAll(false, false)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err = tk.Encode("brown fox jumps over the lazy dog", encParams)
		if err != nil {
			require.NoError(b, err)
		}
	}
}

func BenchmarkDecodeNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Finalize()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true)
		assert.Equal(b, "brown fox jumps over the lazy dog", str)
	}
}
