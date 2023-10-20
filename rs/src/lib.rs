mod configure;
mod encode;

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr::null_mut;

use tokenizers::tokenizer::Tokenizer;

/// This function returns a Tokenizer reference to Golang, casted as a C `void*` after reading
/// tokenizer.json to bytes.
///
/// # Safety
///
/// The caller has ownership of `bytes` and of the returned `Tokenizer`.
#[no_mangle]
pub unsafe extern "C" fn from_bytes(bytes: *const u8, len: u32) -> *mut libc::c_void {
    let bytes_slice = unsafe { std::slice::from_raw_parts(bytes, len as usize) };
    let tokenizer = Tokenizer::from_bytes(bytes_slice).expect("failed to create tokenizer");

    let ptr = Box::into_raw(Box::new(tokenizer));
    ptr.cast()
}


/// # Safety
///
/// This function is return Tokenizer object to Golang from tokenizer.json
#[no_mangle]
pub unsafe extern "C" fn from_file(config: *const libc::c_char) -> *mut libc::c_void {
    let config_cstr = unsafe { CStr::from_ptr(config) };
    let config = config_cstr.to_str().unwrap();
    let config = PathBuf::from(config);
    match Tokenizer::from_file(config) {
        Ok(tokenizer) => {
            let ptr = Box::into_raw(Box::new(tokenizer));
            ptr.cast()
        }
        Err(_) => null_mut(),
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
