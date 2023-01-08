use super::dacecore::*;
use std::error::Error;
use std::ffi::CStr;
use std::fmt::Display;

/// Struct representing a DACE Core Exception.
#[derive(Debug)]
pub struct DACEException {
    pub err: u32,
    pub msg: String,
    pub fun: String,
}

impl DACEException {
    /// Generate a DACEException from the current DACE Core error status.
    fn generate() -> DACEException {
        unsafe {
            let err = daceGetError();
            let msg = CStr::from_ptr(daceGetErrorMSG())
                .to_str()
                .unwrap_or("Unable to decode error message")
                .to_string();
            let fun = CStr::from_ptr(daceGetErrorFunName())
                .to_str()
                .unwrap_or("Unable to decode error function name")
                .to_string();
            daceClearError();
            DACEException { err, msg, fun }
        }
    }
}

impl Display for DACEException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Exception #{} \"{}\" in DACE Core function {}",
            self.err, self.msg, self.fun
        )
    }
}

impl Error for DACEException {}

/// Checks if a DACE Core exception occurred.
///
/// The returned Result is:
///
/// * `Ok(())` if no DACE Core exception occurred,
/// * `Err(DACEException)` if a DACE Core exception occurred,
///   where the `DACEException` contains informtion about the error.
#[inline(always)]
pub fn check_exception() -> Result<(), DACEException> {
    if unsafe { daceGetError() } == 0 {
        Ok(())
    } else {
        Err(DACEException::generate())
    }
}

/// Check if a DACE Core exception occurred and panic if this is the case.
///
/// # Panics
///
/// Panics if a DACE Core exception is found.
#[inline(always)]
pub fn check_exception_panic() {
    check_exception().unwrap()
}
