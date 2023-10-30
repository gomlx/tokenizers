use tokenizers::tokenizer::Tokenizer;

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
