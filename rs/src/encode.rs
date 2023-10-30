use crate::free_string;
use std::ffi::CStr;
use std::ptr::null_mut;
use tokenizers::Encoding;
use tokenizers::tokenizer::Tokenizer;
use std::error::Error;

/// EncodeParams specifies what information to return from the
/// encoded sentences.
/// It controls which fields in Buffer are set.
#[repr(C)]
pub struct EncodeParams {
    add_special_tokens: bool,
    return_tokens: bool,
    return_type_ids: bool,
    return_special_tokens_mask: bool,
    return_attention_mask: bool,
    return_offsets: bool,
    with_offsets_char_mode: bool,
}

/// EncodeResult represents the result of encoding one (`encode` function)
/// or more (`encode_batch` function) sentences.
///
/// It will contain either an error as a C string, or a number of Buffer
/// results, one per sentence encoded -- only one if using `encode` function.
///
/// Once it is no longer used, free the data with `free_encode_results`.
#[repr(C)]
pub struct EncodeResults {
    len: u32,
    encoded: *mut Buffer,
    error: *mut libc::c_char,
}

/// Buffer represents the result of an encoded sentence.
/// Each of the fields are only filled if they were requested in the corresponding
/// EncodeParams setting.
#[repr(C)]
pub struct Buffer {
    ids: *mut u32,
    type_ids: *mut u32,
    special_tokens_mask: *mut u32,
    attention_mask: *mut u32,
    tokens: *mut *mut libc::c_char,
    offsets: *mut Offset,
    len: u32,
}

/// Offset of the toke in the sentence.
/// The Go library limits this to u32 -- we don't expect sentences larger than ~4GB.
#[repr(C)]
pub struct Offset {
    start: u32,
    end: u32,
}

fn encode_process(encoding: Encoding, options: &EncodeParams) -> Result<Buffer, Box<dyn Error>> {
    // ids, tokens
    let mut vec_ids = encoding.get_ids().to_vec();
    vec_ids.shrink_to_fit();
    let ids = vec_ids.as_mut_ptr();
    let len = vec_ids.len();
    std::mem::forget(vec_ids);

    let tokens: *mut *mut libc::c_char;
    if options.return_tokens {
        let tokens_string = encoding.get_tokens();
        let mut vec_tokens: Vec<*mut libc::c_char> = Vec::with_capacity(tokens_string.len());
        for token in tokens_string {
            vec_tokens.push(std::ffi::CString::new(token.as_bytes())?.into_raw());
        }
        vec_tokens.shrink_to_fit();
        tokens = vec_tokens.as_mut_ptr();
        std::mem::forget(vec_tokens);
    } else {
        tokens = null_mut();
    }

    // type_ids
    let type_ids: *mut u32;
    if options.return_type_ids {
        let mut vec_type_ids = encoding.get_type_ids().to_vec();
        vec_type_ids.shrink_to_fit();
        type_ids = vec_type_ids.as_mut_ptr();
        std::mem::forget(vec_type_ids);
    } else {
        type_ids = null_mut();
    }

    // special_tokens_mask
    let mut special_tokens_mask: *mut u32 = null_mut();
    if options.return_special_tokens_mask {
        let mut vec_special_tokens_mask = encoding.get_special_tokens_mask().to_vec();
        vec_special_tokens_mask.shrink_to_fit();
        special_tokens_mask = vec_special_tokens_mask.as_mut_ptr();
        std::mem::forget(vec_special_tokens_mask);
    }

    // attention mask
    let mut attention_mask: *mut u32 = null_mut();
    if options.return_attention_mask {
        let mut vec_attention_mask = encoding.get_attention_mask().to_vec();
        vec_attention_mask.shrink_to_fit();
        attention_mask = vec_attention_mask.as_mut_ptr();
        std::mem::forget(vec_attention_mask);
    }

    // offsets
    let mut offsets: *mut Offset = null_mut();
    if options.return_offsets {
        let mut vec_offsets = encoding
            .get_offsets()
            .iter()
            .map(|s| Offset {
                start: s.0 as u32,
                end: s.1 as u32,
            })
            .collect::<Vec<_>>();
        vec_offsets.shrink_to_fit();
        offsets = vec_offsets.as_mut_ptr();
        std::mem::forget(vec_offsets);
    }

    Ok(Buffer {
        ids,
        type_ids,
        special_tokens_mask,
        attention_mask,
        tokens,
        offsets,
        len: (len as u32),
    })
}

// result_to_encode_results converts errors in a Result<EncodedResult, Error> to
// a new `EncodeResults` struct, with the error converted to C-string.
fn result_to_encode_results(r: Result<EncodeResults, Box<dyn Error>>) -> EncodeResults {
    match r {
        Ok(encode_results) => {
            encode_results
        }
        Err(err) => {
            EncodeResults{
                len: 0,
                encoded: std::ptr::null_mut(),
                error: std::ffi::CString::new(err.to_string())
                    .unwrap().into_raw(),
            }
        }
    }
}

// Create an error from the given message.
fn err<S: AsRef<str>>(message: S) -> Box<dyn Error> {
    Box::new(std::io::Error::new(std::io::ErrorKind::Other, message.as_ref()))
}

// convert_to_tokenizer_ref given a C `void *`.
pub fn convert_to_tokenizer_ref<'a>(tokenizer_ptr: *mut libc::c_void) -> Result<&'a Tokenizer, Box<dyn Error>> {
    unsafe {
        if tokenizer_ptr.cast::<Tokenizer>().as_mut().is_none() {
            return Err(err("tokenizer passed is null"));
        }
        match tokenizer_ptr
            .cast::<Tokenizer>()
            .as_mut() {
            Some(t) => Ok(t),
            None => Err(err("invalid reference to Tokenizer"))
        }
    }
}

fn encode_impl(tokenizer_ptr: *mut libc::c_void,
                   message: *const libc::c_char,
                   options: EncodeParams,
               ) -> Result<EncodeResults, Box<dyn Error>> {
    let tokenizer: &Tokenizer = convert_to_tokenizer_ref(tokenizer_ptr)?;
    let message_cstr = unsafe { CStr::from_ptr(message) };
    let message = message_cstr.to_str().unwrap();

    let encoding_res = if options.with_offsets_char_mode {
        tokenizer.encode_char_offsets(message, options.add_special_tokens)
    } else {
        tokenizer.encode(message, options.add_special_tokens)
    };
    let encoding: Encoding;
    match encoding_res {
        Ok(e) => encoding = e,
        Err(error) => return Err(err(format!("encoding failed: {}", error.to_string()))),
    }

    // Encode it.
    let buffer = encode_process(encoding, &options)?;

    // Package one Buffer into EncodeResults.
    let mut vec_buf: Vec<Buffer> = Vec::with_capacity(1);
    vec_buf.push(buffer);
    let vec_ptr = vec_buf.as_mut_ptr();
    std::mem::forget(vec_buf);
    Ok(EncodeResults{
        len: 1,
        encoded: vec_ptr,
        error: null_mut(),
    })
}


/// Encodes string using given tokenizer and EncodeParams.
#[no_mangle]
pub unsafe extern "C" fn encode(
    tokenizer_ptr: *mut libc::c_void,
    message: *const libc::c_char,
    options: EncodeParams,
) -> EncodeResults {
    result_to_encode_results(
        encode_impl(tokenizer_ptr, message, options))
}

/// Encode a batch of strings using given tokenizer and EncodeParams.
/// The
#[no_mangle]
pub unsafe extern "C" fn encode_batch(
    tokenizer_ptr: *mut libc::c_void,
    num_messages: u32,
    messages: *const *const libc::c_char,
    options: EncodeParams,
) -> EncodeResults {
    result_to_encode_results(
        encode_batch_impl(tokenizer_ptr, num_messages, messages, options))
}

fn encode_batch_impl(
    tokenizer_ptr: *mut libc::c_void,
    num_messages: u32,
    messages: *const *const libc::c_char,
    options: EncodeParams,
) -> Result<EncodeResults, Box<dyn Error>> {
    let tokenizer: &Tokenizer = convert_to_tokenizer_ref(tokenizer_ptr)?;
    let mut encode_messages: Vec<String> = Vec::with_capacity(num_messages as usize);
    unsafe {
        for index in 0..num_messages {
            let cstr_ptr = *messages.offset(index as isize);
            let rust_string = CStr::from_ptr(cstr_ptr).to_string_lossy().into_owned();
            encode_messages.push(rust_string);
        }
    }
    let encoding_res = if options.with_offsets_char_mode {
        tokenizer
            .encode_batch_char_offsets(encode_messages, options.add_special_tokens)
    } else {
        tokenizer
            .encode_batch(encode_messages, options.add_special_tokens)
    };
    let encoding: Vec<Encoding>;
    match encoding_res {
        Ok(e) => encoding = e,
        Err(error) => return Err(err(format!("encoding failed: {}", error.to_string()))),
    }

    // batch process
    let mut vec_buffers: Vec<Buffer> = Vec::with_capacity(num_messages as usize);
    for enc in encoding {
        vec_buffers.push(encode_process(enc, &options)?);
    }
    vec_buffers.shrink_to_fit();
    let encode_results = EncodeResults{
        len: vec_buffers.len() as u32,
        encoded: vec_buffers.as_mut_ptr(),
        error: null_mut(),
    };
    std::mem::forget(vec_buffers);
    Ok(encode_results)
}

/// This function is release a Buffer struct from Rust returned to Golang by `encode`.
// It is not exported to C/Go because one should use EncodeResults instead.
fn free_buffer(buf: Buffer) {
    if !buf.ids.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.ids, buf.len as usize, buf.len as usize);
        }
    }
    if !buf.type_ids.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.type_ids, buf.len as usize, buf.len as usize);
        }
    }
    if !buf.special_tokens_mask.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.special_tokens_mask, buf.len as usize, buf.len as usize);
        }
    }
    if !buf.attention_mask.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.attention_mask, buf.len as usize, buf.len as usize);
        }
    }
    if !buf.tokens.is_null() {
        unsafe {
            let strings = Vec::from_raw_parts(buf.tokens, buf.len as usize, buf.len as usize);
            for s in strings {
                drop(std::ffi::CString::from_raw(s));
            }
        }
    }
    if !buf.offsets.is_null() {
        unsafe {
            Vec::from_raw_parts(buf.offsets, buf.len as usize, buf.len as usize).clear();
        }
    }
}

/// This function is release Vec<Buffer> from Rust returned to Golang by `encode_batch`.
#[no_mangle]
pub unsafe extern "C" fn free_encode_results(results: EncodeResults) {
    if !results.error.is_null() {
        free_string(results.error);
    }
    if results.len > 0 {
        unsafe {
            let vec_buffers = Vec::from_raw_parts(results.encoded, results.len as usize, results.len as usize);
            for buf in vec_buffers {
                free_buffer(buf);
            }
        }
    }
}
