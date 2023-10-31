package tokenizers

import (
	"fmt"
	"github.com/pkg/errors"
	"io"
	"net/http"
	"os"
	"path"
)

// This file handles loading a Tokenizer vocabulary and configuration from
// a pretrained model, including downloading from HuggingFace.

// FromPretrained creates a new Tokenizer by downloading the pretrained tokenizer corresponding
// to the name -- or reading it from disk if it's already been downloaded.
//
// If `cacheDir` is empty, it will download to a temporary file and it will be deleted after
// being used.
// Otherwise, if the tokenizer referred by `name` will be cached in `cacheDir` after download.
//
// If anything goes wrong, an error is returned instead.
func FromPretrained(name string, cacheDir string) (*Tokenizer, error) {
	// Find file location to download (if needed) the vocabulary json file.
	var vocabPath string
	if cacheDir == "" {
		// No cache directory, create a temporary file to store vocabulary.
		f, err := os.CreateTemp("", "tokenizers_vocab.json")
		if err != nil {
			return nil, errors.Wrap(err, "failed to create temporary directory")
		}
		vocabPath = f.Name()
		_ = f.Close()
	} else {
		vocabDir := path.Join(cacheDir, name)
		err := os.MkdirAll(vocabDir, 0770)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create cache directory %q for tokenizer:", vocabDir)
		}
		vocabPath = path.Join(vocabDir, "vocab.json")
	}

	// Download the tokenizer if it does not exist.
	if _, err := os.Stat(vocabPath); os.IsNotExist(err) {
		err := downloadPretrainedTokenizer(name, vocabPath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to download vocabulary for tokenizer %q", name)
		}
	}

	// Load the tokenizer from the cache.
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load vocabulary file %q for tokenizer", vocabPath)
	}

	// Create a new Tokenizer from the JSON data.
	return FromBytes(data)
}

// downloadPretrainedTokenizer from the HuggingFace Hub, into given path.
func downloadPretrainedTokenizer(name string, targetPath string) error {
	// Create a client to the HuggingFace Hub.
	client := http.Client{}

	// Create a request to download the tokenizer.
	req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("https://huggingface.co/%s/resolve/main/vocab.json", name), nil)
	if err != nil {
		return err
	}

	// Make the request and download the tokenizer.
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	// Create the output file.
	outputFile, err := os.Create(targetPath)
	if err != nil {
		return err
	}
	defer func() { _ = outputFile.Close() }()

	// Copy the tokenizer data from the response to the output file.
	_, err = io.Copy(outputFile, resp.Body)
	if err != nil {
		return err
	}

	return nil
}
