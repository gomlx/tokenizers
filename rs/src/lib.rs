mod configure;
mod encode;

use std::ptr::null_mut;
use tokenizers::tokenizer::Tokenizer;

/// PointerOrError returns either a `void *` pointer or an error. 
/// It can be used by functions interfacing with Rust from other languages (using the C binding).
///
/// Either `value` or `error` will be defined. The `value` underlying type is defined by the function
/// returning a `PointerOrError`.
///
/// Ownership of `value` should be documented by the function returning it.
/// Ownership of `error` is transferred back to the caller.
#[repr(C)]
pub struct PointerOrError {
    value: *mut libc::c_void,
    error: *mut libc::c_char,
}

/// This function returns a Tokenizer reference to Golang (casted as a C `void*` in the `value` field) or
/// an error.
///
/// The parameter `bytes` should be the json contents for a `tokenizer.json` file, with its definitions (symbols, 
/// truncation parameters, etc.)
///
/// # Safety
///
/// The caller has ownership of `bytes` and of the returned `Tokenizer`.
#[no_mangle]
pub unsafe extern "C" fn from_bytes(bytes: *const u8, len: u32) -> PointerOrError {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    match Tokenizer::from_bytes(bytes_slice) {
        Ok(t) => return PointerOrError{
            value: Box::into_raw(Box::new(t)).cast(),
            error: null_mut(),
        },
        Err(err) => return PointerOrError{
            value: null_mut(),
            error: std::ffi::CString::new(err.to_string()).unwrap().into_raw(),
        }
    }
}

/// tokenizer.Decode method.
/// The returned string needs to be deallocated with `free_string`.
#[no_mangle]
pub unsafe extern "C" fn decode(
    tokenizer_ptr: *mut libc::c_void,
    ids: *const u32,
    len: u32,
    skip_special_tokens: bool,
) -> *mut libc::c_char {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = tokenizer_ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
    }
    let ids_slice = unsafe { std::slice::from_raw_parts(ids, len as usize) };
    let string = tokenizer
        .decode(ids_slice, skip_special_tokens)
        .expect("failed to decode input");
    let c_string = std::ffi::CString::new(string).unwrap();
    c_string.into_raw()
}

/// Returns the vocab size.
#[no_mangle]
pub unsafe extern "C" fn vocab_size(ptr: *mut libc::c_void) -> u32 {
    let tokenizer: &Tokenizer;
    unsafe {
        tokenizer = ptr
            .cast::<Tokenizer>()
            .as_ref()
            .expect("failed to cast tokenizer");
    }
    tokenizer.get_vocab_size(true) as u32
}

/// Frees a Tokenizer allocated by Rust and returned to Golang.
#[no_mangle]
pub unsafe extern "C" fn free_tokenizer(ptr: *mut libc::c_void) {
    if ptr.is_null() {
        return;
    }
    ptr.cast::<Tokenizer>();
}

/// Frees a `*C.char` allocated by Rust and return to Golang.
#[no_mangle]
pub unsafe extern "C" fn free_string(ptr: *mut libc::c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(ptr));
    }
}
