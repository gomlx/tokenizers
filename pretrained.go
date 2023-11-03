package tokenizers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"github.com/pkg/errors"
	progressbar "github.com/schollz/progressbar/v3"
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
	name, cacheDir, authToken                   string
	isTemporaryCache, forceDownload, forceLocal bool
	showProgressbar                             bool

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

// ForceDownload will ignore previous files in cache and force (re-)download of contents.
func (pt *PretrainedConfig) ForceDownload() *PretrainedConfig {
	pt.forceDownload = true
	return pt
}

// ForceLocal won't use the internet, and will only read from the local disk.
// Notice this prevents even reaching out for the metadata.
func (pt *PretrainedConfig) ForceLocal() *PretrainedConfig {
	pt.forceLocal = true
	return pt
}

// ProgressBar will display a progress bar when downloading files from the network.
// Only displayed if not reading from cache.
func (pt *PretrainedConfig) ProgressBar() *PretrainedConfig {
	pt.showProgressbar = true
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

// makeProgressBar and returns that ProgressFn that updates it.
// It will only display at the first call to the ProgressFn function, and it will automatically close and clean up
// when ProgressFn is called with `eof==true`.
// In case of error, to interrupt it, just call it with `ProgressFn(0, 0, /*eof=*/ true)`
func makeProgressBar(name string) ProgressFn {
	var data = &struct {
		name          string
		bar           *progressbar.ProgressBar
		started, done bool
	}{
		name:    name,
		started: false,
		done:    false,
	}

	return func(progress, downloaded, total int, eof bool) {
		if data.done {
			return
		}
		if eof && !data.started {
			// Do nothing, since we never actually created the progressbar.
			data.done = true
			return
		}
		if !data.started {
			data.bar = progressbar.DefaultBytes(int64(total), data.name)
			data.started = true
		}
		if progress != 0 {
			_ = data.bar.Add64(int64(progress))
		}
		if eof {
			_ = data.bar.Close()
			data.done = true
		}
	}
}

// Done concludes the configuration of FromPretrainedWith and actually downloads (or loads from disk)
// the tokenizer.
func (pt *PretrainedConfig) Done() (*Tokenizer, error) {
	// Sanity checking.
	if pt.forceDownload && pt.forceLocal {
		return nil, errors.New("cannot use ForceLocal and ForceDownload at the same time, one or the other (or none)")
	}

	// Initialize unset attributes.
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
	repoType := "model"
	revision := "main"
	var progressFn ProgressFn
	if pt.showProgressbar {
		progressFn = makeProgressBar(tokenizerConfigFileName)
	}
	configPath, commitHash, err := Download(
		pt.ctx, pt.client,
		pt.name, repoType, revision, tokenizerConfigFileName, pt.cacheDir, pt.authToken,
		pt.forceDownload, pt.forceLocal, progressFn)
	if err != nil {
		if progressFn != nil {
			progressFn(0, 0, 0, true)
		}
		return nil, errors.WithMessagef(err, "tokenizers.FromPretrainedWith() failed to download %q", tokenizerConfigFileName)
	}
	var contents []byte
	contents, err = os.ReadFile(configPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read downloaded tokenizer configuration file in %q", configPath)
	}
	dec := json.NewDecoder(bytes.NewReader(contents))
	var config = map[string]any{}
	if err = dec.Decode(&config); err != nil {
		return nil, errors.Wrapf(err, "failed to parse JSON from tokenizer configuration file in %q", configPath)
	}

	fmt.Printf("configuration: %q\n", config)
	_ = commitHash
	return nil, errors.New("not implemented")
}
