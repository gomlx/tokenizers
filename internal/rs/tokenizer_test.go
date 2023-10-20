package rs_test

import (
	_ "embed"
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

func TestEmbeddingConfig(t *testing.T) {
	tk, err := rs.FromBytes(embeddedBytes)
	require.NoError(t, err)
	defer tk.Close()

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
			encodeRes := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}
}

func TestEncode(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Close()
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
			addSpecial: false,
		},
		{
			name:       "empty string with special tokens",
			str:        "",
			addSpecial: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encodeRes := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}
}

func TestEncodeOptions(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Close()
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
			encoding := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encoding = tk.Encode(tt.str, tt.addSpecial, rs.WithReturnTypeIds())
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encoding = tk.Encode(tt.str, tt.addSpecial, rs.WithReturnSpecialTokensMask())
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")

			encoding = tk.Encode(tt.str, tt.addSpecial, rs.WithReturnAttentionMask())
			assert.Equal(t, tt.wantIDs, encoding.TokenIds, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIds, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")

			encoding = tk.Encode(tt.str, tt.addSpecial, rs.WithReturnAll(false))
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
	defer tk.Close()

	encodeRes := tk.Encode("brown fox jumps over the lazy dog", false, rs.WithReturnOffsets())
	expected := []rs.Offset{
		{Start: 0, End: 5},
		{Start: 6, End: 9},
		{Start: 10, End: 15},
		{Start: 16, End: 20},
		{Start: 21, End: 24},
		{Start: 25, End: 29},
		{Start: 30, End: 33},
	}
	assert.Equal(t, encodeRes.Offsets, expected)

	encodeResV1 := tk.Encode("brown fox jumps over the lazy dog", false, rs.WithReturnCharModeOffsets())
	expectedV1 := []rs.Offset{
		{Start: 0, End: 5},
		{Start: 6, End: 9},
		{Start: 10, End: 15},
		{Start: 16, End: 20},
		{Start: 21, End: 24},
		{Start: 25, End: 29},
		{Start: 30, End: 33},
	}
	assert.Equal(t, encodeResV1.Offsets, expectedV1)
}

func TestEncodeBatch(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Close()

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
			results := tk.EncodeBatch([]string{tt.str, tt.str}, tt.addSpecial)
			require.Equal(t, 2, len(results))
			for ii, res := range results {
				assert.Equalf(t, tt.wantIDs, res.TokenIds, "Sentence %d", ii)
				assert.Equalf(t, tt.wantTokens, res.Tokens, "Sentence %d", ii)
			}
		})
	}
}

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
			defer func() { _ = tk.Close() }()

			err = tk.SetTruncation(uint8(tt.dir), uint32(tt.maxLen), 0, 0)
			require.NoError(t, err)

			tk.Encode(tt.str, tt.addSpecial)
			encodeRes := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}
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
			defer func() { _ = tk.Close() }()
			tk.SetPadding(tt.padLen, tt.dir, tt.padToMultipleOf, tt.padId, 0, tt.padToken)

			tk.Encode(tt.str, tt.addSpecial)
			encodeRes := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
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
			defer tk.Close()

			encodeRes := tk.Encode(tt.str, tt.addSpecial, rs.WithReturnAll(false))
			assert.Equal(t, tt.wantIDs, encodeRes.TokenIds)
			assert.Equal(t, tt.wantTokens, encodeRes.Tokens)
		})
	}
}

func TestDecode(t *testing.T) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(t, err)
	defer tk.Close()
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
	defer tk.Close()
	assert.Equal(t, uint32(30522), tk.VocabSize())
}

func BenchmarkEncodeNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Close()
	expected := []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encodeRes := tk.Encode("brown fox jumps over the lazy dog", false)
		assert.Equal(b, expected, encodeRes.TokenIds)
	}
}

func BenchmarkEncodeWithOptionNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tk.Encode("brown fox jumps over the lazy dog", false, rs.WithReturnAll(false))
	}
}

func BenchmarkDecodeNTimes(b *testing.B) {
	tk, err := rs.FromFile(bertJson)
	require.NoError(b, err)
	defer tk.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true)
		assert.Equal(b, "brown fox jumps over the lazy dog", str)
	}
}
