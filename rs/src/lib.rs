mod configure;
mod encode;
mod decode;

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
