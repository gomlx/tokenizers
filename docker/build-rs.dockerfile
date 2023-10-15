# syntax=docker/dockerfile:1.3

FROM rust:1.73 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./rs .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=${TARGETPLATFORM} \
    --mount=type=cache,target=/root/target,id=${TARGETPLATFORM} \
    cargo build --release

FROM golang:1.21.2 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
COPY --from=builder-rust /workspace/target/release/libgomlx_tokenizers.a .
#COPY ./release .

# Once API is finished, build Go libraries here: notice we cannot run tests, since
# we are compiling cross-platform.
#RUN --mount=type=cache,target=/root/.cache/go-build \
#    --mount=type=cache,target=/var/cache/go,id=${TARGETPLATFORM} \
#    CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
