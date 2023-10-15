# Tokenizers for Go
Tokenizers for Language Models - Go API for [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)

## Highlights

> [!IMPORTANT]  
> TODO: nothing implemented yet.

* Allow customization to various LLMs, exposing most of the functionality of the HuggingFace Tokenizers library.
* Provide a `from_pretrained` API, that downloads parameters to various known models -- levaraging HuggingFace Hub

## Installation

This library is a wrapper around the Rust implementation by HuggingFace, and hence requires a the Rust library to be compiled and available in the system (for CGO to link it).

> [!IMPORTANT]  
> TODO

## Thank You

* [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
* @sunhailin-Leo, this project is a fork of [sunhailin-Leo/tokenizers](https://github.com/sunhailin-Leo/tokenizers/)
* Original [daulet/tokenizers](https://github.com/daulet/tokenizers) project -- of which [sunhailin-Leo/tokenizers](https://github.com/sunhailin-Leo/tokenizers/) is a fork

## Questions

### Why fork and not collaborate with already existing tokenizers project ?

I plan to do very large changes, the library will likely become bigger (but with more functionality), with breaking changes to the API to be more aligned with [GoMLX](https://github.com/gomlx/gomlx/) APIs. To much to ask from the original authors.



