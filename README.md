# Tokenizers for Go 

<img align="right" src="https://github.com/gomlx/gopjrt/assets/7460115/0f2869be-f64e-48b8-b2fa-1f6cbe703204" alt="Under Construction" width="480"/>

**UNDER CONSTRUCTION** 

**Not functional yet**, but for Gemma/Gemini/T5 and other Google models, see https://github.com/eliben/go-sentencepiece/.

## About

Tokenizers for Language Models - Go API for [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)

## Highlights

> [!IMPORTANT]  
> TODO: nothing implemented yet.

* Allow customization to various LLMs, exposing most of the functionality of the HuggingFace Tokenizers library.
* Provide a `from_pretrained` API, that downloads parameters to various known models -- levaraging HuggingFace Hub

## Installation

This library is a wrapper around the Rust implementation by HuggingFace, and it requires the compiled Rust
code available as a `libgomlx_tokenizers.a`. 

To make that easy, the project provides a prebuilt `libgomlx_tokenizers.a` in the git repository (for the popular
platforms), so for many nothing is needed (except having CGO enabled -- for cross-compilation set `CGO_ENABLED=1`),
and it can be simply included as any other Go library.

If you want to build the underlying Rust wrapper and dependencies yourselves for any reason (including maybe 
to add support for a different platform), it uses the [Mage](https://magefile.org/) build system -- an improved
`Makefile`-like that uses Go. 

If you create a new rule for a different platform, please consider contributing it back :smile:

> [!IMPORTANT]  
> TODO

## Thank You

* [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers): most of the work should be credited to them, this is just a nice (hopefully) wrapper to Go.
* @sunhailin-Leo, this project is a fork of [sunhailin-Leo/tokenizers](https://github.com/sunhailin-Leo/tokenizers/).
* Original [daulet/tokenizers](https://github.com/daulet/tokenizers) project -- of which [sunhailin-Leo/tokenizers](https://github.com/sunhailin-Leo/tokenizers/) is a fork.

## Questions

**Why fork and not collaborate with an already existing tokenizers project ?**

I plan to revamp how the library is organized, its "ergonomics"
to be more aligned with [GoMLX](https://github.com/gomlx/gomlx/) APIs, and add documentation. 
I will also expand the functionality to match (as much as I'm able to do) HuggingFace's library.
All this will completely break the API of the original repositories,
and I felt too much to ask from the original authors.
