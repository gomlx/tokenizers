package tokenizers

// HuggingFace Hub related functionality.

import (
	"bytes"
	"context"
	"fmt"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"io"
	"net/http"
	"os"
	"path"
	"runtime"
	"strconv"
	"strings"
	"text/template"
)

var SessionId string

func init() {
	sessionUUID, err := uuid.NewRandom()
	if err != nil {
		panicf("failed generating UUID for SessionId: %v", err)
	}
	SessionId = strings.Replace(sessionUUID.String(), "-", "", -1)
}

const (
	// Versions, as of the time of this writing.
	transformersVersion = "4.34.1"
	hubVersion          = "0.17.3"
	tokenizersVersion   = "0.0.1"
)

const (
	HeaderXRepoCommit = "X-Repo-Commit"
	HeaderXLinkedETag = "X-Linked-Etag"
	HeaderXLinkedSize = "X-Linked-Size"
)

func getEnvOr(key, defaultValue string) string {
	v := os.Getenv(key)
	if v == "" {
		return defaultValue
	}
	return v
}

// DefaultCacheDir for HuggingFace Hub, same used by the python library.
//
// Its prefix is either `${XDG_CACHE_HOME}` if set, or `~/.cache` otherwise. Followed by `/huggingface/hub/`.
// So typically: `~/.cache/huggingface/hub/`.
func DefaultCacheDir() string {
	cacheDir := getEnvOr("XDG_CACHE_HOME", path.Join(os.Getenv("HOME"), ".cache"))
	cacheDir = path.Join(cacheDir, "huggingface", "hub")
	return cacheDir
}

// HttpUserAgent returns a user agent to use with HuggingFace Hub API.
// Loosely based on https://github.com/huggingface/transformers/blob/main/src/transformers/utils/hub.py#L198.
func HttpUserAgent() string {
	return fmt.Sprintf("transfomers/%v; golang/%s; session_id/%s; gomlx_tokenizers/%v; hf_hub/%s",
		transformersVersion, runtime.Version(), SessionId, tokenizersVersion, hubVersion)
}

// RepoIdSeparator is used to separate repository/model names parts when mapping to file names.
// Likely only for internal use.
const RepoIdSeparator = "--"

// RepoFolderName returns a serialized version of a hf.co repo name and type, safe for disk storage
// as a single non-nested folder.
//
// Based on github.com/huggingface/huggingface_hub repo_folder_name.
func RepoFolderName(repoId, repoType string) string {
	parts := []string{repoType + "s"}
	parts = append(parts, strings.Split(repoId, "/")...)
	return strings.Join(parts, RepoIdSeparator)
}

var (
	RepoTypesUrlPrefixes = map[string]string{
		"dataset": "datasets/",
		"space":   "spaces/",
	}

	DefaultRevision = "main"

	HuggingFaceUrlTemplate = template.Must(template.New("hf_url").Parse(
		"https://huggingface.co/{{.RepoId}}/resolve/{{.Revision}}/{{.Filename}}"))
)

// GetUrl is based on the `hf_hub_url` function defined in the [huggingface_hub](https://github.com/huggingface/huggingface_hub) library.
func GetUrl(repoId, fileName, repoType, revision string) string {
	if prefix, found := RepoTypesUrlPrefixes[repoType]; found {
		repoId = prefix + repoId
	}
	if revision == "" {
		revision = DefaultRevision
	}
	var buf bytes.Buffer
	err := HuggingFaceUrlTemplate.Execute(&buf,
		struct{ RepoId, Revision, Filename string }{repoId, revision, fileName})
	if err != nil {
		panicf("HuggingFaceUrlTemplate failed (!? pls report the bug, this shouldn't happen) with %+v", err)
	}
	url := buf.String()
	return url
}

// GetHeaders is based on the `build_hf_headers` function defined in the [huggingface_hub](https://github.com/huggingface/huggingface_hub) library.
// TODO: add support for authentication token.
func GetHeaders(userAgent, token string) map[string]string {
	return map[string]string{
		"user-agent": userAgent,
	}
}

// Download returns file either from cache or by downloading from HuggingFace Hub.
//
// Args:
//
// * `ctx` for the requests. There may be more than one request, the first being an `HEAD` HTTP.
// * `client` used to make HTTP requests. I can be created with `&httpClient{}`.
// * `repoId` and `fileName`: define the file and repository (model) name to download.
// * `cacheDir`: directory where to store the downloaded files, or reuse if previously downloaded.
// * `token`: used for authentication. TODO: not implemented yet.
//
// On success it returns the `filePath` to the downloaded file. Otherwise it returns an error.
func Download(ctx context.Context, client *http.Client, repoId, fileName, cacheDir, token string) (filePath string, err error) {
	if cacheDir == "" {
		err = errors.New("Download() requires a cacheDir, even if temporary, to store the results of the download")
		return
	}
	repoType := "model"         // TODO, for now only "model", the default.
	revision := DefaultRevision // commit hashes not accepted yet.
	userAgent := HttpUserAgent()
	if token != "" {
		// TODO, for now no token support.
		err = errors.Errorf("no support yet for authentication token while attemption to download %q from %q",
			fileName, repoId)
		return
	}
	folderName := RepoFolderName(repoId, repoType)

	// Find and if necessary create local file on disk.
	storageDir := path.Join(cacheDir, folderName)
	err = os.MkdirAll(storageDir, 0770)
	if err != nil {
		err = errors.Wrapf(err, "failed to create cache directory %q:", storageDir)
		return
	}
	fmt.Println("storageDir:", storageDir)

	filePath = path.Join(strings.Split(fileName, "/")...) // Join path parts using current OS separator.
	fmt.Println("filePath:", filePath)

	url := GetUrl(repoId, fileName, repoType, revision)
	fmt.Println("URL:", url)

	headers := GetHeaders(userAgent, token)
	fmt.Printf("%q\n", headers)

	var metadata *HFFileMetadata
	metadata, err = getFileMetadata(ctx, client, url, token, headers)
	fmt.Printf("Metadata:\n\t%#v\n", metadata)
	if err != nil {
		return
	}
	commitHash := metadata.CommitHash
	if commitHash == "" {
		err = errors.Errorf("resource %q for %q doesn't seem to be on huggingface.co (missing commit header)",
			fileName, repoId)
		return
	}
	etag := metadata.ETag
	if etag == "" {
		err = errors.Errorf("resource %q for %q doesn't have an ETag, not able to ensure reproduceability",
			fileName, repoId)
		return
	}

	var urlToDownload = url
	if metadata.Location != url {
		// In the case of a redirect, remove authorization header when downloading blob
		delete(headers, "authorization")
	}

	blobPath := path.Join(storageDir, "blobs", etag)
	pointerPath := path.Join(storageDir, revision, commitHash)

	_ = blobPath
	_ = pointerPath
	_ = urlToDownload
	_ = cacheDir
	return
}

// HFFileMetadata used by HuggingFace Hub.
type HFFileMetadata struct {
	CommitHash, ETag, Location string
	Size                       int
}

func removeQuotes(str string) string {
	return strings.TrimRight(strings.TrimLeft(str, "\""), "\"")
}

// getFileMetadata: make a "HEAD" HTTP request and return the response with the header.
func getFileMetadata(ctx context.Context, client *http.Client, url, token string, headers map[string]string) (metadata *HFFileMetadata, err error) {
	// Create a request to download the tokenizer.
	var req *http.Request
	req, err = http.NewRequestWithContext(ctx, http.MethodHead, url, nil)
	if err != nil {
		err = errors.Wrap(err, "failed request for metadata: ")
		return
	}

	// Include requested headers, plus prevent any compression => we want to know the real size of the file.
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	req.Header.Set("Accept-Encoding", "identity")

	// Make the request and download the tokenizer.
	resp, err := client.Do(req)
	if err != nil {
		err = errors.Wrap(err, "failed request for metadata: ")
		return
	}

	// TODO: handle redirects.
	defer func() { _ = resp.Body.Close() }()
	var contents []byte
	contents, err = io.ReadAll(resp.Body)
	if err != nil {
		err = errors.Wrapf(err, "failed reading response (%d) for metadata: ", resp.StatusCode)
		return
	}

	// Check status code.
	if resp.StatusCode != 200 {
		err = errors.Errorf("request for metadata from %q failed with the following message: %q",
			url, contents)
		return
	}

	metadata = &HFFileMetadata{
		CommitHash: resp.Header.Get(HeaderXRepoCommit),
	}
	metadata.ETag = resp.Header.Get(HeaderXLinkedETag)
	if metadata.ETag == "" {
		metadata.ETag = resp.Header.Get("ETag")
	}
	metadata.ETag = removeQuotes(metadata.ETag)
	metadata.Location = resp.Header.Get("Location")
	if metadata.Location == "" {
		metadata.Location = resp.Request.URL.String()
	}

	if sizeStr := resp.Header.Get(HeaderXLinkedSize); sizeStr != "" {
		metadata.Size, err = strconv.Atoi(sizeStr)
		if err != nil {
			err = nil // Discard
			metadata.Size = 0
		}
	}
	if metadata.Size == 0 {
		metadata.Size = int(resp.ContentLength)
	}
	return
}

func 