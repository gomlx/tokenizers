use std::ffi::CStr;
use tokenizers::tokenizer::Tokenizer;

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
