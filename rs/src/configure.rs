use std::ffi::CStr;
use tokenizers::tokenizer::Tokenizer;
use crate::encode::convert_to_tokenizer_ref;


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

/// TruncationParameters represents the truncation parameters
/// that can be set with "with_truncation".
#[repr(C)]
pub struct TruncationParams {
    direction: u8, // 0 -> Left (*); 1 -> Right
    strategy: u8,  // 0 -> LongestFirst (*), 1 -> OnlyFirst, 2 -> OnlySecond,
    max_length: u32,  // Default 512
    stride: u32,  // Default 0
}

/// set_truncation modifies the tokenizer with the given truncation parameters.
/// It returns null if ok, or a string with an error message (owned by caller) if something went wrong.
/// The returned string needs to be freed with `free_string`.
#[no_mangle]
pub unsafe extern "C" fn set_truncation(
    tokenizer_ptr: *mut libc::c_void,
    params: *const TruncationParams,
) -> *mut libc::c_char {
    let tokenizer: &mut Tokenizer;
    unsafe {
        let o = tokenizer_ptr.cast::<Tokenizer>().as_mut();
        if o.is_none() {
            return std::ffi::CString::new("failed to cast tokenizer").unwrap().into_raw();
        } else {
            tokenizer = o.unwrap();
        }
    }
    unsafe {
        let result = if params.is_null() {
                tokenizer.with_truncation(None)
            } else {
                tokenizer.with_truncation(
                    Some(tokenizers::tokenizer::TruncationParams {
                        max_length: (*params).max_length as usize,
                        direction: match (*params).direction {
                            0 => tokenizers::tokenizer::TruncationDirection::Left,
                            1 => tokenizers::tokenizer::TruncationDirection::Right,
                            _ => panic!("invalid truncation direction"),
                        },
                        stride: (*params).stride as usize,
                        strategy: match (*params).strategy {
                            0 => tokenizers::tokenizer::TruncationStrategy::LongestFirst,
                            1 => tokenizers::tokenizer::TruncationStrategy::OnlyFirst,
                            2 => tokenizers::tokenizer::TruncationStrategy::OnlySecond,
                            _ => panic!("invalid truncation strategy"),
                        },
                    }))
            };
        if result.is_err() {
            let err = format!("failed tokenizer.with_truncation: {}", result.unwrap_err());
            return std::ffi::CString::new(err).unwrap().into_raw();
        }
    }

    // No errors.
    std::ptr::null_mut()
}

/// get_truncation gets the current Tokenizer's truncation parameters.
///
/// If there are truncation parameters configured in the Tokenizer, the values are read into the `params` passed,
/// and it returns true.
///
/// If there are no truncation values configured, it returns false.
#[no_mangle]
pub unsafe extern "C" fn get_truncation(
    tokenizer_ptr: *mut libc::c_void, params: *mut TruncationParams) -> bool {
    let tokenizer: &Tokenizer = convert_to_tokenizer_ref(tokenizer_ptr).expect("Invalid Tokenizer object!?");
    match tokenizer.get_truncation() {
        Some(p) => {
            (*params).max_length = p.max_length as u32;
            (*params).stride = p.stride as u32;
            (*params).direction = match p.direction {
                tokenizers::tokenizer::TruncationDirection::Left => 0,
                tokenizers::tokenizer::TruncationDirection::Right => 1,
            };
            (*params).strategy = match p.strategy {
                tokenizers::tokenizer::TruncationStrategy::LongestFirst => 0,
                tokenizers::tokenizer::TruncationStrategy::OnlyFirst => 1,
                tokenizers::tokenizer::TruncationStrategy::OnlySecond => 2,
            };
            return true;
        }
        None => return false,
    }
}

/// PaddingParams represents the padding parameters: it maps to the values in
/// tokenizers::tokenizer::PaddingParams.
#[repr(C)]
pub struct PaddingParams {
    strategy: u32,  // 0 -> BatchLongest, >0 -> Fixed(value)
    direction: u8,  // 0 -> Left, !=0 -> Right
    pad_to_multiple_of: u32,  // Disabled if 0.
    pad_id: u32,
    pad_type_id: u32,
    pad_token: *const libc::c_char,
}

/// set_padding modifies the tokenizer with the given padding parameters.
/// It doesn't return anything.
#[no_mangle]
pub unsafe extern "C" fn set_padding(
    tokenizer_ptr: *mut libc::c_void,
    params: *const PaddingParams) {
    let tokenizer: &mut Tokenizer;
    unsafe {
        let o = tokenizer_ptr.cast::<Tokenizer>().as_mut();
        if o.is_none() {
            return
        }
        tokenizer = o.unwrap()
    }
    if params.is_null() {
        tokenizer.with_padding(None);
        return;
    }

    // Convert char* to string.
    let mut pad_token: String = String::new();
    if !(*params).pad_token.is_null() {
        let pad_token_cstr = unsafe { CStr::from_ptr((*params).pad_token) };
        pad_token = pad_token_cstr.to_str().unwrap().to_string();
    }

    // Set up padding.
    _ = tokenizer.with_padding(
        Some(tokenizers::tokenizer::PaddingParams {
            strategy: match (*params).strategy {
                0 => tokenizers::tokenizer::PaddingStrategy::BatchLongest,
                _ => tokenizers::tokenizer::PaddingStrategy::Fixed((*params).strategy as usize),
            },
            direction: match (*params).direction {
                0 => tokenizers::tokenizer::PaddingDirection::Left,
                _ => tokenizers::tokenizer::PaddingDirection::Right,
            },
            pad_to_multiple_of: if (*params).pad_to_multiple_of == 0 {
                    None
                } else {
                    Some((*params).pad_to_multiple_of as usize)
                },
            pad_id: (*params).pad_id,
            pad_type_id: (*params).pad_type_id,
            pad_token: pad_token,
        }));
}


/// get_padding gets the current Tokenizer's padding parameters.
///
/// If there are padding parameters configured in the Tokenizer, the values are read into the `params` passed,
/// and it returns true. The `params.pad_token` ownership is transferred to the caller, who must free it
/// after use (see `free_string()`).
///
/// If there are no truncation values configured, it returns false.
#[no_mangle]
pub unsafe extern "C" fn get_padding(
    tokenizer_ptr: *mut libc::c_void, params: *mut PaddingParams) -> bool {
    let tokenizer: &Tokenizer = convert_to_tokenizer_ref(tokenizer_ptr).expect("Invalid Tokenizer object!?");
    match tokenizer.get_padding() {
        Some(p) => {
            (*params).pad_id = p.pad_id as u32;
            (*params).pad_type_id = p.pad_type_id as u32;
            (*params).pad_to_multiple_of = match p.pad_to_multiple_of {
                Some(v) => v as u32,
                None => 0,
            };
            (*params).direction = match p.direction {
                tokenizers::tokenizer::PaddingDirection::Left => 0,
                tokenizers::tokenizer::PaddingDirection::Right => 1,
            };
            (*params).strategy = match p.strategy {
                tokenizers::tokenizer::PaddingStrategy::BatchLongest => 0,
                tokenizers::tokenizer::PaddingStrategy::Fixed(value) => value as u32,
            };
            (*params).pad_token = std::ffi::CString::new(p.pad_token.as_bytes()).unwrap().into_raw();
            return true;
        }
        None => return false,
    }
}

