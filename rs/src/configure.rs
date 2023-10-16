use tokenizers::tokenizer::Tokenizer;

/// TruncationParameters represents the truncation parameters
/// that can be set with "with_truncation".
#[repr(C)]
pub struct TruncationParameters {
    direction: u8, // 0 -> Left (*); 1 -> Right
    strategy: u8,  // 0 -> LongestFirst (*), 1 -> OnlyFirst, 2 -> OnlySecond,
    max_length: u32,  // Default 512
    stride: u32,  // Default 0
}

/// with_truncation modifies the given tokenizer with the given truncation parameters.
/// It returns a string with an error message (owned by caller) if something went wrong.
#[no_mangle]
pub unsafe extern "C" fn with_truncation(
    tokenizer_ptr: *mut libc::c_void,
    params: *const TruncationParameters,
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

