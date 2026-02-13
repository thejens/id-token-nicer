use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codec;
use crate::mix;

fn to_vocab(vocab_size: usize) -> PyResult<codec::Vocab> {
    match vocab_size {
        2048 => Ok(codec::Vocab::V2048),
        4096 => Ok(codec::Vocab::V4096),
        8192 => Ok(codec::Vocab::V8192),
        16384 => Ok(codec::Vocab::V16384),
        32768 => Ok(codec::Vocab::V32768),
        n => Err(PyValueError::new_err(format!(
            "vocab_size must be 2048, 4096, 8192, 16384, or 32768, got {n}"
        ))),
    }
}

fn to_style(style: &str) -> PyResult<codec::Style> {
    match style {
        "memorable" => Ok(codec::Style::Memorable),
        "token" => Ok(codec::Style::Token),
        s => Err(PyValueError::new_err(format!(
            "style must be 'memorable' or 'token', got '{s}'"
        ))),
    }
}

fn decode_err(e: codec::DecodeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A codec instance bound to a specific vocabulary size and style.
#[pyclass]
struct Codec {
    vocab: codec::Vocab,
    style: codec::Style,
}

#[pymethods]
impl Codec {
    #[new]
    #[pyo3(signature = (vocab_size=2048, style="memorable"))]
    fn new(vocab_size: usize, style: &str) -> PyResult<Self> {
        Ok(Self {
            vocab: to_vocab(vocab_size)?,
            style: to_style(style)?,
        })
    }

    /// Vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    /// Bits per word for this vocabulary.
    #[getter]
    fn bits_per_word(&self) -> u32 {
        self.vocab.bits_per_word()
    }

    /// Word list style ("memorable" or "token").
    #[getter]
    fn style(&self) -> &'static str {
        match self.style {
            codec::Style::Memorable => "memorable",
            codec::Style::Token => "token",
        }
    }

    /// Encode an integer to words (variable length, no zero-padding).
    fn int_to_words(&self, value: u128) -> String {
        codec::encode_int(value, self.vocab, self.style)
    }

    /// Decode words back to an integer (variable length).
    fn words_to_int(&self, phrase: &str) -> PyResult<u128> {
        codec::decode_int(phrase, self.vocab, self.style).map_err(decode_err)
    }

    /// Encode a UUID string to words.
    fn uuid_to_words(&self, uuid_str: &str) -> PyResult<String> {
        let uuid = uuid::Uuid::parse_str(uuid_str)
            .map_err(|e| PyValueError::new_err(format!("invalid UUID: {e}")))?;
        Ok(codec::encode_uuid(uuid.as_bytes(), self.vocab, self.style))
    }

    /// Decode words back to a UUID string.
    fn words_to_uuid(&self, phrase: &str) -> PyResult<String> {
        let bytes = codec::decode_uuid(phrase, self.vocab, self.style).map_err(decode_err)?;
        Ok(uuid::Uuid::from_bytes(bytes).to_string())
    }

    /// Number of words for UUID encoding with this vocabulary.
    fn uuid_word_count(&self) -> usize {
        codec::uuid_word_count(self.vocab)
    }

    /// Number of words for a given payload byte count.
    fn word_count(&self, payload_bytes: usize) -> usize {
        codec::word_count(payload_bytes, self.vocab)
    }

    fn __repr__(&self) -> String {
        let style = match self.style {
            codec::Style::Memorable => "memorable",
            codec::Style::Token => "token",
        };
        format!("Codec(vocab_size={}, style={:?})", self.vocab.size(), style)
    }
}

// --- Module-level convenience functions ---

#[pyfunction]
#[pyo3(signature = (uuid_str, vocab_size=None, style=None))]
fn encode_uuid(uuid_str: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    let uuid = uuid::Uuid::parse_str(uuid_str)
        .map_err(|e| PyValueError::new_err(format!("invalid UUID: {e}")))?;
    Ok(codec::encode_uuid(uuid.as_bytes(), vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, vocab_size=None, style=None))]
fn decode_uuid(phrase: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    let bytes = codec::decode_uuid(phrase, vocab, style).map_err(decode_err)?;
    Ok(uuid::Uuid::from_bytes(bytes).to_string())
}

#[pyfunction]
#[pyo3(signature = (value, vocab_size=None, style=None))]
fn encode_int(value: u128, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    Ok(codec::encode_int(value, vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, vocab_size=None, style=None))]
fn decode_int(phrase: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<u128> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    codec::decode_int(phrase, vocab, style).map_err(decode_err)
}

#[pyfunction]
#[pyo3(signature = (value, vocab_size=None, style=None))]
fn encode_u128(value: u128, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    Ok(codec::encode_u128(value, vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, vocab_size=None, style=None))]
fn decode_u128(phrase: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<u128> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    codec::decode_u128(phrase, vocab, style).map_err(decode_err)
}

#[pyfunction]
#[pyo3(signature = (value, vocab_size=None, style=None))]
fn encode_u64(value: u64, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    Ok(codec::encode_u64(value, vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, vocab_size=None, style=None))]
fn decode_u64(phrase: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<u64> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    codec::decode_u64(phrase, vocab, style).map_err(decode_err)
}

#[pyfunction]
#[pyo3(signature = (value, vocab_size=None, style=None))]
fn encode_u32(value: u32, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    Ok(codec::encode_u32(value, vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, vocab_size=None, style=None))]
fn decode_u32(phrase: &str, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<u32> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    codec::decode_u32(phrase, vocab, style).map_err(decode_err)
}

#[pyfunction]
#[pyo3(signature = (data, vocab_size=None, style=None))]
fn encode_bytes(data: &[u8], vocab_size: Option<usize>, style: Option<&str>) -> PyResult<String> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    Ok(codec::encode_bytes(data, vocab, style))
}

#[pyfunction]
#[pyo3(signature = (phrase, expected_len, vocab_size=None, style=None))]
fn decode_bytes(phrase: &str, expected_len: usize, vocab_size: Option<usize>, style: Option<&str>) -> PyResult<Vec<u8>> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    let style = to_style(style.unwrap_or("memorable"))?;
    codec::decode_bytes(phrase, expected_len, vocab, style).map_err(decode_err)
}

#[pyfunction]
#[pyo3(signature = (payload_bytes, vocab_size=None))]
fn word_count(payload_bytes: usize, vocab_size: Option<usize>) -> PyResult<usize> {
    let vocab = to_vocab(vocab_size.unwrap_or(2048))?;
    Ok(codec::word_count(payload_bytes, vocab))
}

#[pyfunction]
fn mix_value(value: u128, bits: u32) -> u128 {
    mix::mix_value(value, bits)
}

#[pyfunction]
fn unmix_value(value: u128, bits: u32) -> u128 {
    mix::unmix_value(value, bits)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Codec>()?;
    m.add_function(wrap_pyfunction!(encode_uuid, m)?)?;
    m.add_function(wrap_pyfunction!(decode_uuid, m)?)?;
    m.add_function(wrap_pyfunction!(encode_int, m)?)?;
    m.add_function(wrap_pyfunction!(decode_int, m)?)?;
    m.add_function(wrap_pyfunction!(encode_u128, m)?)?;
    m.add_function(wrap_pyfunction!(decode_u128, m)?)?;
    m.add_function(wrap_pyfunction!(encode_u64, m)?)?;
    m.add_function(wrap_pyfunction!(decode_u64, m)?)?;
    m.add_function(wrap_pyfunction!(encode_u32, m)?)?;
    m.add_function(wrap_pyfunction!(decode_u32, m)?)?;
    m.add_function(wrap_pyfunction!(encode_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(decode_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(word_count, m)?)?;
    m.add_function(wrap_pyfunction!(mix_value, m)?)?;
    m.add_function(wrap_pyfunction!(unmix_value, m)?)?;
    Ok(())
}
