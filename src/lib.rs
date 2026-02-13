mod bloom;
mod checksum;
pub mod codec;
mod mix;
#[cfg(feature = "python")]
mod python;
pub mod tokenlist;
pub mod wordlist;

pub use codec::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn id_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
