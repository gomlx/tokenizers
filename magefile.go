//go:build mage

// Builds the underlying Rust dependencies for the GoMLX Tokenizers project.

package main

import (
	"fmt"
	"github.com/magefile/mage/mg"
	"github.com/magefile/mage/sh"
	"github.com/magefile/mage/target"
	"github.com/pkg/errors"
	"os"
	"os/exec"
	"path"
	"strings"
)

// Default sets default action of mage to be build the library for current platform.
var Default = Build

// must panics if an error is passed.
func must(err error) {
	if err != nil {
		panic(err)
	}
}

// must1 panics if an error is passed, otherwies returns t.
func must1[T any](t T, err error) T {
	must(err)
	return t
}

var (
	// Please fill out this mapping if adding support for a different platform.
	// The Go platform name is created with `$GOOS/$GOARCH`, e.g. `linux/amd64`.
	// The Rust platform name is from the list returned by `rustup target list`.
	mapGoPlatformToRustPlatform = map[string]string{
		"linux/amd64":  "x86_64-unknown-linux-gnu",
		"darwin/arm64": "aarch64-apple-darwin",
		"darwin/amd64": "x86_64-apple-darwin",
	}
)

const (
	libraryName = "libgomlx_tokenizers.a"
	headerName  = "gomlx_tokenizers.h"
)

// Builds the Rust library `libgomlx_tokenizers.a` for the current platform.
// It uses the `mapGoPlatformToFunction` to map the platform to the corresponding target function.
func Build() error {
	mg.Deps(Header)
	return rustBuild(getGoPlatform())
}

// Builds the Rust library `libgomlx_tokenizers.a` for each of the platforms included for release by default --
// the most popular ones.
//
// TODO: Rust cross-compilation with C/C++ dependencies not working for now, see details in
// TODO: https://github.com/rust-lang/rust/issues/84984 and https://github.com/briansmith/ring/issues/1442
func Release() error {
	// Trying to parallelize the building Rust code will probably be slower, since each one will already be parallelized
	// by `cargo`.
	//mg.SerialDeps(Linux_amd64, Darwin_arm64, Darwin_amd64)
	mg.SerialDeps(Header, Build)
	return nil
}

// Builds the Rust library `libgomlx_tokenizers.a` for linux/amd64 platform.
func Linux_amd64() error {
	mg.Deps(Header)
	return rustBuild("linux/amd64")
}

// Builds the Rust library `libgomlx_tokenizers.a` for darwin/amd64 platform.
func Darwin_amd64() error {
	mg.Deps(Header)
	return rustBuild("darwin/amd64")
}

// Builds the Rust library `libgomlx_tokenizers.a` for darwin/arm64 platform.
func Darwin_arm64() error {
	mg.Deps(Header)
	return rustBuild("darwin/arm64")
}

// Header builds the `internal/rs/gomlx_tokenizers.h` header file from the Rust sources, using `cbindgen`.
func Header() error {
	// Check whether target is up-to-date.
	pwd := must1(os.Getwd())
	dst := path.Join(pwd, "internal", "rs", headerName)
	modified, err := target.Glob(dst, "rs/cbindgen.toml", "rs/Cargo.toml", "rs/src/*.rs")
	if err != nil {
		return err
	}
	if !modified {
		return nil
	}

	// Make sure cbindgen is installed.
	if _, err := exec.LookPath("cbindgen"); err != nil {
		return errors.WithMessage(err,
			"can't find `cbindgen`, a program that converts Rust signatures to C, needed for "+
				"binding with Go -- it can usually be installed with `cargo install cbindgen`")
	}

	// Build header file.
	if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
		return errors.WithMessagef(err, "removing previous copy of %q", dst)
	}
	must(os.Chdir("rs"))
	fmt.Printf("Building header file %q\n", dst)
	err = sh.Run("cbindgen", "--config", "cbindgen.toml", "--output", dst)
	must(os.Chdir(".."))
	return err
}

// rustBuild builds the rust library `libgomlx_tokenizers.a` for the corresponding Go platform.
// The resulting binary library is stored in `lib/<goPlatform>/` subdirectory.
func rustBuild(goPlatform string) error {
	rustPlatform, found := mapGoPlatformToRustPlatform[goPlatform]
	if !found {
		return fmt.Errorf("platform %q in Rust is not configured -- "+
			"check whether $GOOS or $GOARCH are correctly set, or alternative create a new target "+
			"rule for the unknown platform in `magefile.go`, it's usually very simple", goPlatform)
	}

	// Creates target directory if needed.
	platformDir := strings.Replace(goPlatform, "/", "_", -1)
	dstPath := path.Join("lib", platformDir)
	err := os.MkdirAll(dstPath, 0770)
	if err != nil {
		return errors.WithMessagef(err, "creating target directory %q", dstPath)
	}

	// Checks whether compilation is needed.
	dst := path.Join(dstPath, libraryName)
	modified, err := target.Glob(dst, "rs/Cargo.toml", "rs/src/*.rs")
	if err != nil {
		return errors.WithMessagef(err, "checking whether recompilation needed")
	}
	if !modified {
		// Nothing to do.
		return nil
	}

	// Build from rust directory `rs`.
	must(os.Chdir("rs"))
	fmt.Printf("Building for platform %q\n", goPlatform)
	err = sh.Run("cargo", "build", "--release", "--target", rustPlatform)
	must(os.Chdir(".."))
	if err != nil {
		return err
	}
	return sh.Copy(dst, path.Join("rs", "target", rustPlatform, "release", libraryName))
}

// getGoPlatform return `$GOOS/$GOARCH`.
// If environment GOOS and GOARCH are not set, it uses instead the output of `go env GOOS` and `go env GOARCH`.
func getGoPlatform() string {
	return fmt.Sprintf("%s/%s", getGoEnv("GOOS"), getGoEnv("GOARCH"))
}

// getGoEnv gets the value associated with the environment variable `key`.
// If it is not set, it attempts to get the value returned by `go env <key>`.
func getGoEnv(key string) string {
	value := os.Getenv(key)
	if value != "" {
		return value
	}
	var err error
	value, err = sh.Output("go", "env", key)
	if err != nil {
		panic(errors.WithMessagef(err, "getting value of %q with `go env`", key))
	}
	return value
}
