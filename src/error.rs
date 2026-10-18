//! Error types.

use std::fmt;
use std::io;

/// Everything that can go wrong when building, querying or loading an index.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A vector (or a flat batch of vectors) does not have the index dimension.
    DimensionMismatch {
        /// Dimension the index was created with (or a multiple of it for batches).
        expected: usize,
        /// Length that was supplied.
        actual: usize,
    },
    /// The requested dimension is zero or larger than [`MAX_DIM`](crate::MAX_DIM).
    InvalidDimension(usize),
    /// A construction parameter is out of range.
    InvalidParameter {
        /// Name of the offending parameter.
        name: &'static str,
        /// Human-readable description of the accepted range.
        reason: &'static str,
    },
    /// A vector contains NaN or infinity, or its squared norm exceeds
    /// `f32::MAX / 8`, the bound that keeps every distance between accepted
    /// vectors finite.
    NonFiniteVector,
    /// A zero vector was supplied to a cosine index, where it has no direction.
    ZeroVector,
    /// The key is already present (and not deleted) in the index.
    DuplicateKey(u64),
    /// The index would exceed [`MAX_NODES`](crate::MAX_NODES) nodes, tombstones included.
    CapacityExceeded,
    /// A quantized search was requested on an index without quantization.
    QuantizationDisabled,
    /// An I/O error while reading or writing a serialized index.
    Io(io::Error),
    /// The serialized bytes are not a valid index.
    Format(FormatError),
    /// A serialized index describes an index that would need more memory than
    /// the loader was allowed to allocate (see
    /// [`HnswIndex::from_bytes_with_limit`](crate::HnswIndex::from_bytes_with_limit)).
    MemoryLimitExceeded {
        /// Heap bytes the loaded index would hold.
        required: usize,
        /// The limit in effect.
        limit: usize,
    },
}

/// Reasons a serialized index is rejected by [`HnswIndex::from_bytes`](crate::HnswIndex::from_bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FormatError {
    /// The input does not start with the `HNSWIDX\0` magic bytes.
    BadMagic,
    /// The format version is not one this build can read.
    UnsupportedVersion(u32),
    /// The input ends before the structure it describes is complete.
    Truncated,
    /// The CRC-32 trailer does not match the contents.
    ChecksumMismatch {
        /// Checksum stored in the trailer.
        stored: u32,
        /// Checksum computed over the received bytes.
        computed: u32,
    },
    /// Bytes remain after the last section.
    TrailingBytes,
    /// A field holds a value that violates an invariant of the index.
    Invalid(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::InvalidDimension(dim) => write!(
                f,
                "invalid dimension {dim}: must be between 1 and {}",
                crate::MAX_DIM
            ),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::NonFiniteVector => f.write_str("vector contains NaN or infinite values"),
            Self::ZeroVector => f.write_str("zero vector has no direction under cosine distance"),
            Self::DuplicateKey(key) => write!(f, "key {key} is already present"),
            Self::CapacityExceeded => write!(
                f,
                "index capacity of {} nodes exceeded; compact to reclaim tombstones",
                crate::MAX_NODES
            ),
            Self::QuantizationDisabled => {
                f.write_str("quantized search requested but quantization is not enabled")
            }
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Format(err) => write!(f, "invalid index file: {err}"),
            Self::MemoryLimitExceeded { required, limit } => write!(
                f,
                "loading the index needs {required} bytes of memory, more than the limit of {limit}"
            ),
        }
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => f.write_str("missing HNSWIDX magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported format version {v}"),
            Self::Truncated => f.write_str("unexpected end of input"),
            Self::ChecksumMismatch { stored, computed } => write!(
                f,
                "checksum mismatch: stored {stored:#010x}, computed {computed:#010x}"
            ),
            Self::TrailingBytes => f.write_str("unexpected bytes after the last section"),
            Self::Invalid(what) => write!(f, "invalid field: {what}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Format(err) => Some(err),
            _ => None,
        }
    }
}

impl std::error::Error for FormatError {}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<FormatError> for Error {
    fn from(err: FormatError) -> Self {
        Self::Format(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_informative() {
        let err = Error::DimensionMismatch {
            expected: 4,
            actual: 3,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 4, got 3");
        let err = Error::from(FormatError::ChecksumMismatch {
            stored: 1,
            computed: 2,
        });
        assert_eq!(
            err.to_string(),
            "invalid index file: checksum mismatch: stored 0x00000001, computed 0x00000002"
        );
        let err = Error::MemoryLimitExceeded {
            required: 300,
            limit: 200,
        };
        assert_eq!(
            err.to_string(),
            "loading the index needs 300 bytes of memory, more than the limit of 200"
        );
    }

    #[test]
    fn source_chains_to_inner_errors() {
        use std::error::Error as _;
        let err = Error::from(io::Error::other("disk on fire"));
        assert!(err.source().is_some());
        assert!(Error::ZeroVector.source().is_none());
    }
}
