//go:build linux && amd64

package rs

// Empty dependency, just make sure the directory is retrieved with `go get`,
// since it will hold the `libgomlx_tokenizers.a` file, needed by CGO.
import _ "github.com/gomlx/tokenizers/lib/linux_amd64"
