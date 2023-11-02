package tokenizers

import (
	"context"
	"fmt"
	"github.com/pkg/errors"
	"net/http"
	"os"
)

// This file handles loading a Tokenizer vocabulary and configuration from
// a pretrained model, including downloading from HuggingFace.

// Filenames used for tokenizers
const (
	specialTokensMapFileName = "special_tokens_map.json"
	addedTokensFileName      = "added_tokens.json"
	tokenizerConfigFileName  = "tokenizer_config.json"
)

// PretrainedConfig for how to download (or load from disk) a pretrained Tokenizer.
// It can be configured in different ways (see methods below), and when finished configuring,
// call Done to actually download (or load from disk) the pretrained tokenizer.
type PretrainedConfig struct {
	name, cacheDir, authToken string
	isTemporaryCache          bool

	client *http.Client
	ctx    context.Context
}

// FromPretrainedWith creates a new Tokenizer by downloading the pretrained tokenizer corresponding
// to the name.
//
// There are several options that can be configured.
// After that one calls Done, and it will return the Tokenizer object (or an error).
//
// If anything goes wrong, an error is returned instead.
func FromPretrainedWith(name string) *PretrainedConfig {
	pt := &PretrainedConfig{
		name:     name,
		cacheDir: DefaultCacheDir(),
		ctx:      context.Background(),
	}

	// cacheDir defaults to the same used by pytorch transformers.
	return pt
}

// CacheDir configures cacheDir as directory to store a cache of the downloaded files.
// If the tokenizer has already been downloaded in the directory, it will be read from disk
// instead of the network.
//
// The default value is `~/.cache/huggingface/hub/`, the same used by the original Transformers library.
// The cache home is overwritten by `$XDG_CACHE_HOME` if it is set.
func (pt *PretrainedConfig) CacheDir(cacheDir string) *PretrainedConfig {
	pt.cacheDir = cacheDir
	return pt
}

// NoCache to be used, no copy is kept of the downloaded tokenizer.
func (pt *PretrainedConfig) NoCache() *PretrainedConfig {
	pt.cacheDir = ""
	return pt
}

// AuthToken sets the authentication token to use.
// The default is to use no token, which works for simply downloading most tokenizers.
// TODO: not implemented yet, it will lead to an error when calling Done.
func (pt *PretrainedConfig) AuthToken(token string) *PretrainedConfig {
	pt.authToken = token
	return pt
}

// HttpClient configures an http.Client to use to connect to HuggingFace Hub.
// The default is `nil`, in which case one will be created for the requests.
func (pt *PretrainedConfig) HttpClient(client *http.Client) *PretrainedConfig {
	pt.client = client
	return pt
}

// Context configures the given context to download content from the internet.
// The default is to use `context.Background()` with no timeout.
func (pt *PretrainedConfig) Context(ctx context.Context) *PretrainedConfig {
	pt.ctx = ctx
	return pt
}

// Done concludes the configuration of FromPretrainedWith and actually downloads (or loads from disk)
// the tokenizer.
func (pt *PretrainedConfig) Done() (*Tokenizer, error) {
	if pt.client == nil {
		// Default HTTP client: no timeout, empty cookie jar.
		pt.client = &http.Client{}
	}

	// Create a temporary cacheDir is one was not configured.
	if pt.cacheDir == "" {
		pt.isTemporaryCache = true
		// No cache directory, create a temporary file to store vocabulary.
		f, err := os.CreateTemp("", "gomlx_tokenizers")
		if err != nil {
			return nil, errors.Wrap(err, "failed to create temporary directory")
		}
		pt.cacheDir = f.Name()
		_ = f.Close()
		if err := os.Remove(pt.cacheDir); err != nil {
			return nil, errors.Wrap(err, "failed to remove temporary file where the downloading directory would be created")
		}
	}

	// Read Tokenizer configuration.
	configPath, err := Download(pt.ctx, pt.client, pt.name, tokenizerConfigFileName, pt.cacheDir, pt.token)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to download %q")

	}
	fmt.Printf("> configPath=%s\n", configPath)

	/*
		// Download the tokenizer if it does not exist.
		if _, err := os.Stat(vocabPath); os.IsNotExist(err) {
			err := downloadPretrainedTokenizer(pt.name, vocabPath)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to download vocabulary for tokenizer %q", pt.name)
			}
		}

		// Load the tokenizer from the cache.
		data, err := os.ReadFile(vocabPath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to load vocabulary file %q for tokenizer", vocabPath)
		}

		fmt.Printf("Loaded file:\n%s\n", string(data))
		// Create a new Tokenizer from the JSON data.
		return FromBytes(data)

	*/
	return nil, errors.New("not implemented")
}
