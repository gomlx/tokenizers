# Makefile used to build the underlying Rust library for various platforms.
# To list and install cross-compilation support for Rust check out:
#
#   $ rustup target list
#   $ rustup target add <target_platform>
#
# Notice the target naming changes from Go: Rust
build:
	@cd rs && cargo build --release
	@cp rs/target/release/libgomlx_tokenizers.a .
	# @go build .

build-example:
	@docker build -f ./example/Dockerfile . -t tokenizers-example

release-linux-%:
	docker build --platform linux/$* -f docker/build-rs.dockerfile . -t tokenizers.linux-$*
	mkdir -p artifacts/linux-$*
	docker run -v $(PWD)/release/linux-$*:/mnt --entrypoint cp tokenizers.linux-$* /workspace/libgomlx_tokenizers.a /mnt/libgomlx_tokenizers.a
	cd artifacts/linux-$* && \
		tar -czf libgomlx_tokenizers.linux-$*.tar.gz libgomlx_tokenizers.a
	mkdir -p artifacts/all
	cp artifacts/linux-$*/libgomlx_tokenizers.linux-$*.tar.gz artifacts/all/libgomlx_tokenizers.linux-$*.tar.gz

release-darwin-%:
	cd rs && cargo build --release --target $*-apple-darwin
	mkdir -p artifacts/darwin-$*
	cp rs/target/$*-apple-darwin/release/libgomlx_tokenizers.a artifacts/darwin-$*/libgomlx_tokenizers.a
	cd artifacts/darwin-$* && \
		tar -czf libgomlx_tokenizers.darwin-$*.tar.gz libgomlx_tokenizers.a
	mkdir -p artifacts/all
	cp artifacts/darwin-$*/libgomlx_tokenizers.darwin-$*.tar.gz artifacts/all/libgomlx_tokenizers.darwin-$*.tar.gz


release: release-darwin-aarch64 release-darwin-x86_64 release-linux-arm64 release-linux-x86_64
	cp artifacts/all/gomlx_tokenizers.darwin-aarch64.tar.gz artifacts/all/gomlx_tokenizers.darwin-arm64.tar.gz
	cp artifacts/all/gomlx_tokenizers.linux-arm64.tar.gz artifacts/all/gomlx_tokenizers.linux-aarch64.tar.gz
	cp artifacts/all/gomlx_tokenizers.linux-x86_64.tar.gz artifacts/all/gomlx_tokenizers.linux-amd64.tar.gz

test: build
	@go test -v ./... -count=1

clean:
	rm -rf libgomlx_tokenizers.a rs/target
