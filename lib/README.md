## Pre-Compiled Rust Libraries

There is one directory per platform supported with the pre-compiled
`libgomlx_tokenizers.a` library.

It includes the [Huggingface Tokenizers (Rust)](https://github.com/huggingface/tokenizers/tree/main/tokenizers) library
and the small Rust wrapper (that exposes the functionality with C signature).

They are built automatically using the [mage](magefile.org)(a simpler and fancier Makefile, in Go), see file `../magefile.go`.
