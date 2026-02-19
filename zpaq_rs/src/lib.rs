//! Safe Rust bindings for libzpaq (compression, decompression, crypto utilities).
//!
//! The bindings are backed by a small C++ shim compiled in `build.rs`.
//!
//! Notes on performance:
//! - Rust LTO applies within Rust crates, and C++ LTO applies within the C++ objects.
//! - Cross-language inlining (Rust <-> C++) is toolchain-dependent and not guaranteed.
//! - Prefer `compress_size`/`compress_size_stream` when you only need sizes.
//!
//! Use `compress_to_vec`
//! / `decompress_to_vec` for simple use cases, or `compress_stream` /
//! `decompress_stream` to work with any `Read`/`Write` streams.

mod sys;

use std::collections::VecDeque;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::slice;

pub type Result<T> = std::result::Result<T, ZpaqError>;

#[derive(Debug)]
pub enum ZpaqError {
    Ffi(String),
    NulInString,
}

impl std::fmt::Display for ZpaqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZpaqError::Ffi(s) => write!(f, "libzpaq: {s}"),
            ZpaqError::NulInString => write!(f, "string contained NUL byte"),
        }
    }
}

impl std::error::Error for ZpaqError {}

fn last_error_string() -> Option<String> {
    unsafe {
        let len = sys::zpaq_last_error_len();
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        let copied = sys::zpaq_last_error_copy(buf.as_mut_ptr() as *mut c_char, len);
        buf.truncate(copied);
        String::from_utf8(buf).ok()
    }
}

fn err_from_last() -> ZpaqError {
    last_error_string()
        .map(ZpaqError::Ffi)
        .unwrap_or_else(|| ZpaqError::Ffi("unknown error".to_string()))
}

fn clear_last_error() {
    unsafe { sys::zpaq_clear_last_error() };
}

/// A `Write` implementation that discards data while counting bytes.
///
/// Useful when you only care about compressed sizes (e.g. NCD/NED style metrics).
#[derive(Debug, Default, Clone, Copy)]
pub struct CountingWriter {
    bytes: u64,
}

impl CountingWriter {
    pub fn bytes_written(&self) -> u64 {
        self.bytes
    }
}

impl Write for CountingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(buf.len() as u64)
            .ok_or_else(|| std::io::Error::other("byte counter overflow"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

// ---------------- Callback plumbing ----------------

struct ReadCtx<R: Read + Send> {
    reader: R,
}

struct WriteCtx<W: Write + Send> {
    writer: W,
}

unsafe extern "C" fn read_cb<R: Read + Send>(
    ctx: *mut std::os::raw::c_void,
    buf: *mut c_char,
    n: c_int,
) -> c_int {
    unsafe {
        let ctx = &mut *(ctx as *mut ReadCtx<R>);
        let slice = slice::from_raw_parts_mut(buf as *mut u8, n as usize);
        match ctx.reader.read(slice) {
            Ok(bytes) => bytes as c_int,
            Err(e) => {
                set_callback_error(&e.to_string());
                sys::RUST_CALLBACK_ERROR
            }
        }
    }
}

unsafe extern "C" fn write_cb<W: Write + Send>(
    ctx: *mut std::os::raw::c_void,
    buf: *const c_char,
    n: c_int,
) -> c_int {
    unsafe {
        let ctx = &mut *(ctx as *mut WriteCtx<W>);
        let slice = slice::from_raw_parts(buf as *const u8, n as usize);
        match ctx.writer.write_all(slice) {
            Ok(()) => 0,
            Err(e) => {
                set_callback_error(&e.to_string());
                sys::RUST_CALLBACK_ERROR
            }
        }
    }
}

// ---------------- Streaming compressor ----------------

#[derive(Default)]
struct StreamReader {
    buf: VecDeque<u8>,
}

impl StreamReader {
    fn push(&mut self, b: u8) {
        self.buf.push_back(b);
    }
}

impl Read for StreamReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let mut n = 0usize;
        while n < out.len() {
            match self.buf.pop_front() {
                Some(b) => {
                    out[n] = b;
                    n += 1;
                }
                None => break,
            }
        }
        Ok(n)
    }
}

/// Streaming ZPAQ compressor that exposes incremental encoded bit counts.
///
/// Note: Streaming mode supports numeric levels 1..=3 and explicit method
/// strings (x/s/i/0...) that do not use block preprocessing.
pub struct StreamingCompressor {
    compressor: *mut sys::Compressor,
    reader: *mut sys::RustReader,
    writer: *mut sys::RustWriter,
    reader_ctx: *mut ReadCtx<StreamReader>,
    writer_ctx: *mut WriteCtx<CountingWriter>,
}

unsafe impl Send for StreamingCompressor {}

impl StreamingCompressor {
    pub fn new(method: &str) -> Result<Self> {
        let method_trim = method.trim();
        if method_trim.is_empty() {
            return Err(ZpaqError::Ffi("method string is empty".into()));
        }
        let numeric = method_trim.parse::<i32>().ok();
        let level = numeric.filter(|v| (1..=3).contains(v));
        if numeric.is_some() && level.is_none() {
            return Err(ZpaqError::Ffi(
                "streaming numeric levels support 1..3 only; use x/s/i/0 with no preprocessing"
                    .into(),
            ));
        }

        let compressor = unsafe { sys::zpaq_compressor_new() };
        if compressor.is_null() {
            return Err(ZpaqError::Ffi("zpaq_compressor_new failed".into()));
        }

        let reader_ctx = Box::into_raw(Box::new(ReadCtx {
            reader: StreamReader::default(),
        }));
        let writer_ctx = Box::into_raw(Box::new(WriteCtx {
            writer: CountingWriter::default(),
        }));

        let reader =
            unsafe { sys::zpaq_reader_new(reader_ctx.cast(), None, Some(read_cb::<StreamReader>)) };
        if reader.is_null() {
            unsafe {
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(ZpaqError::Ffi("zpaq_reader_new failed".into()));
        }

        let writer = unsafe {
            sys::zpaq_writer_new(
                writer_ctx.cast(),
                Some(put_cb::<CountingWriter>),
                Some(write_cb::<CountingWriter>),
            )
        };
        if writer.is_null() {
            unsafe {
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(ZpaqError::Ffi("zpaq_writer_new failed".into()));
        }

        let rc_out = unsafe { sys::zpaq_compressor_set_output(compressor, writer) };
        if rc_out != 0 {
            unsafe {
                sys::zpaq_writer_free(writer);
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(err_from_last());
        }

        let rc_in = unsafe { sys::zpaq_compressor_set_input(compressor, reader) };
        if rc_in != 0 {
            unsafe {
                sys::zpaq_writer_free(writer);
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(err_from_last());
        }

        let rc_tag = unsafe { sys::zpaq_compressor_write_tag(compressor) };
        if rc_tag != 0 {
            unsafe {
                sys::zpaq_writer_free(writer);
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(err_from_last());
        }

        let rc_block = if let Some(level) = level {
            unsafe { sys::zpaq_compressor_start_block_level(compressor, level) }
        } else {
            let method_c = CString::new(method_trim).map_err(|_| ZpaqError::NulInString)?;
            unsafe { sys::zpaq_compressor_start_block_method(compressor, method_c.as_ptr()) }
        };
        if rc_block != 0 {
            unsafe {
                sys::zpaq_writer_free(writer);
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(err_from_last());
        }

        let rc_seg =
            unsafe { sys::zpaq_compressor_start_segment(compressor, ptr::null(), ptr::null()) };
        if rc_seg != 0 {
            unsafe {
                sys::zpaq_writer_free(writer);
                sys::zpaq_reader_free(reader);
                sys::zpaq_compressor_free(compressor);
                drop(Box::from_raw(reader_ctx));
                drop(Box::from_raw(writer_ctx));
            }
            return Err(err_from_last());
        }

        Ok(Self {
            compressor,
            reader,
            writer,
            reader_ctx,
            writer_ctx,
        })
    }

    pub fn push(&mut self, b: u8) -> Result<()> {
        unsafe {
            let ctx = &mut *self.reader_ctx;
            ctx.reader.push(b);
        }
        let rc = unsafe { sys::zpaq_compressor_compress(self.compressor, 1) };
        if rc < 0 {
            return Err(err_from_last());
        }
        Ok(())
    }

    pub fn bits(&self) -> f64 {
        unsafe { sys::zpaq_compressor_get_bits(self.compressor) }
    }
}

impl Drop for StreamingCompressor {
    fn drop(&mut self) {
        unsafe {
            sys::zpaq_writer_free(self.writer);
            sys::zpaq_reader_free(self.reader);
            sys::zpaq_compressor_free(self.compressor);
            drop(Box::from_raw(self.reader_ctx));
            drop(Box::from_raw(self.writer_ctx));
        }
    }
}

unsafe extern "C" fn put_cb<W: Write + Send>(ctx: *mut std::os::raw::c_void, c: c_int) -> c_int {
    unsafe {
        let ctx = &mut *(ctx as *mut WriteCtx<W>);
        let byte = [c as u8];
        match ctx.writer.write_all(&byte) {
            Ok(()) => 0,
            Err(e) => {
                set_callback_error(&e.to_string());
                sys::RUST_CALLBACK_ERROR
            }
        }
    }
}

fn set_callback_error(msg: &str) {
    if let Ok(cstr) = CString::new(msg) {
        unsafe { sys::zpaq_set_last_error(cstr.as_ptr()) };
    }
}

struct FfiReader<R: Read + Send> {
    raw: *mut sys::RustReader,
    ctx: *mut ReadCtx<R>,
}

impl<R: Read + Send> FfiReader<R> {
    fn new(reader: R) -> Result<Self> {
        let ctx = Box::into_raw(Box::new(ReadCtx { reader }));
        let raw = unsafe { sys::zpaq_reader_new(ctx as *mut _, None, Some(read_cb::<R>)) };
        if raw.is_null() {
            unsafe {
                drop(Box::from_raw(ctx));
            }
            return Err(err_from_last());
        }
        Ok(Self { raw, ctx })
    }
}

impl<R: Read + Send> Drop for FfiReader<R> {
    fn drop(&mut self) {
        unsafe {
            sys::zpaq_reader_free(self.raw);
            drop(Box::from_raw(self.ctx));
        }
    }
}

struct FfiWriter<W: Write + Send> {
    raw: *mut sys::RustWriter,
    ctx: *mut WriteCtx<W>,
}

impl<W: Write + Send> FfiWriter<W> {
    fn new(writer: W) -> Result<Self> {
        let ctx = Box::into_raw(Box::new(WriteCtx { writer }));
        let raw =
            unsafe { sys::zpaq_writer_new(ctx as *mut _, Some(put_cb::<W>), Some(write_cb::<W>)) };
        if raw.is_null() {
            unsafe {
                drop(Box::from_raw(ctx));
            }
            return Err(err_from_last());
        }
        Ok(Self { raw, ctx })
    }
}

impl<W: Write + Send> Drop for FfiWriter<W> {
    fn drop(&mut self) {
        unsafe {
            sys::zpaq_writer_free(self.raw);
            drop(Box::from_raw(self.ctx));
        }
    }
}

// ---------------- Public API ----------------

/// Compress all bytes from `input` into a Vec using the given method string (e.g. "1", "14", "x4.0ci1").
pub fn compress_to_vec(input: &[u8], method: &str) -> Result<Vec<u8>> {
    let cursor = std::io::Cursor::new(input);
    let mut out = Vec::new();
    compress_stream(cursor, &mut out, method, None, None)?;
    Ok(out)
}

/// Compute the compressed size (in bytes) without materializing the compressed output.
///
/// This is usually the fastest way to get $C(x)$ for information-theoretic distances.
pub fn compress_size(input: &[u8], method: &str) -> Result<u64> {
    compress_size_stream(std::io::Cursor::new(input), method, None, None)
}

/// Compute the compressed size (in bytes) using multiple threads.
///
/// This splits the input into ZPAQ blocks (based on the method's block size) and
/// compresses blocks in parallel using `libzpaq::compressBlock`.
pub fn compress_size_parallel(input: &[u8], method: &str, threads: usize) -> Result<u64> {
    compress_size_stream_parallel(std::io::Cursor::new(input), method, None, None, threads)
}

/// Compute the compressed size (in bytes) without materializing the compressed output.
///
/// This uses a C++ counting writer to avoid Rust-side per-write overhead.
pub fn compress_size_stream<R: Read + Send>(
    reader: R,
    method: &str,
    filename: Option<&str>,
    comment: Option<&str>,
) -> Result<u64> {
    clear_last_error();
    let method_c = CString::new(method).map_err(|_| ZpaqError::NulInString)?;
    let filename_c = match filename {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };
    let comment_c = match comment {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };
    let reader = FfiReader::new(reader)?;
    let mut out_size: u64 = 0;
    let rc = unsafe {
        sys::zpaq_compress_size(
            reader.raw,
            method_c.as_ptr(),
            filename_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            comment_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            1,
            &mut out_size as *mut u64,
        )
    };
    if rc == 0 {
        Ok(out_size)
    } else {
        Err(err_from_last())
    }
}

/// Compute the compressed size (in bytes) using multiple threads.
///
/// If `threads <= 1`, this falls back to the single-threaded path.
pub fn compress_size_stream_parallel<R: Read + Send>(
    reader: R,
    method: &str,
    filename: Option<&str>,
    comment: Option<&str>,
    threads: usize,
) -> Result<u64> {
    clear_last_error();
    let method_c = CString::new(method).map_err(|_| ZpaqError::NulInString)?;
    let filename_c = match filename {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };
    let comment_c = match comment {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };
    let reader = FfiReader::new(reader)?;
    let mut out_size: u64 = 0;
    let rc = unsafe {
        sys::zpaq_compress_size_parallel(
            reader.raw,
            method_c.as_ptr(),
            filename_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            comment_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            1,
            threads as i32,
            &mut out_size as *mut u64,
        )
    };
    if rc == 0 {
        Ok(out_size)
    } else {
        Err(err_from_last())
    }
}

/// Compute the archive size (in bytes) that `zpaq add` would write for a single file.
///
/// This runs the real `zpaq.cpp` JIDAC pipeline in-process (including fragment dedup),
/// with `archive=""` so output is discarded but the size accounting is identical.
///
/// This is the right metric to compare against:
/// `zpaq add my.arc <file> ...` → `du -b my.arc`.
pub fn zpaq_add_archive_size_file(path: &str, method: &str, threads: usize) -> Result<u64> {
    clear_last_error();
    let path_c = CString::new(path).map_err(|_| ZpaqError::NulInString)?;
    let method_c = CString::new(method).map_err(|_| ZpaqError::NulInString)?;
    let mut out_size: u64 = 0;
    let rc = unsafe {
        sys::zpaq_jidac_add_archive_size_file(
            path_c.as_ptr(),
            method_c.as_ptr(),
            threads as c_int,
            &mut out_size as *mut u64,
        )
    };
    if rc == 0 {
        Ok(out_size)
    } else {
        Err(err_from_last())
    }
}

/// Decompress a full ZPAQ stream into memory.
pub fn decompress_to_vec(input: &[u8]) -> Result<Vec<u8>> {
    let cursor = std::io::Cursor::new(input);
    let mut out = Vec::new();
    decompress_stream(cursor, &mut out)?;
    Ok(out)
}

/// Compute the decompressed size (in bytes) without materializing output.
pub fn decompress_size(input: &[u8]) -> Result<u64> {
    decompress_size_stream(std::io::Cursor::new(input))
}

/// Compute the decompressed size (in bytes) without materializing output.
pub fn decompress_size_stream<R: Read + Send>(reader: R) -> Result<u64> {
    clear_last_error();
    let reader = FfiReader::new(reader)?;
    let mut out_size: u64 = 0;
    let rc = unsafe { sys::zpaq_decompress_size(reader.raw, &mut out_size as *mut u64) };
    if rc == 0 {
        Ok(out_size)
    } else {
        Err(err_from_last())
    }
}

/// Compress from any `Read` to any `Write`.
pub fn compress_stream<R: Read + Send, W: Write + Send>(
    reader: R,
    writer: W,
    method: &str,
    filename: Option<&str>,
    comment: Option<&str>,
) -> Result<()> {
    clear_last_error();
    let method_c = CString::new(method).map_err(|_| ZpaqError::NulInString)?;
    let filename_c = match filename {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };
    let comment_c = match comment {
        Some(s) => Some(CString::new(s).map_err(|_| ZpaqError::NulInString)?),
        None => None,
    };

    let reader = FfiReader::new(reader)?;
    let writer = FfiWriter::new(writer)?;

    let rc = unsafe {
        sys::zpaq_compress(
            reader.raw,
            writer.raw,
            method_c.as_ptr(),
            filename_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            comment_c
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null()),
            1,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(err_from_last())
    }
}

/// Decompress from any `Read` to any `Write`.
pub fn decompress_stream<R: Read + Send, W: Write + Send>(reader: R, writer: W) -> Result<()> {
    clear_last_error();
    let reader = FfiReader::new(reader)?;
    let writer = FfiWriter::new(writer)?;
    let rc = unsafe { sys::zpaq_decompress(reader.raw, writer.raw) };
    if rc == 0 {
        Ok(())
    } else {
        Err(err_from_last())
    }
}

/// Derive a 32-byte stretched key using scrypt (libzpaq defaults: N=16384, r=8, p=1).
pub fn stretch_key(key32: [u8; 32], salt32: [u8; 32]) -> Result<[u8; 32]> {
    clear_last_error();
    let mut out = [0u8; 32];
    let rc = unsafe { sys::zpaq_stretch_key(out.as_mut_ptr(), key32.as_ptr(), salt32.as_ptr()) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(err_from_last())
    }
}

/// Fill a buffer with cryptographically strong random bytes.
pub fn random_bytes(len: usize) -> Result<Vec<u8>> {
    clear_last_error();
    let mut buf = vec![0u8; len];
    let rc = unsafe { sys::zpaq_random(buf.as_mut_ptr(), len as c_int) };
    if rc == 0 {
        Ok(buf)
    } else {
        Err(err_from_last())
    }
}

/// Minimal SHA-1 helper using the libzpaq implementation.
pub fn sha1(bytes: &[u8]) -> Result<[u8; 20]> {
    clear_last_error();
    let s = unsafe { sys::zpaq_sha1_new() };
    if s.is_null() {
        return Err(err_from_last());
    }
    unsafe {
        sys::zpaq_sha1_write(s, bytes.as_ptr() as *const c_char, bytes.len() as i64);
    }
    let mut out = [0u8; 20];
    let rc = unsafe { sys::zpaq_sha1_result(s, out.as_mut_ptr()) };
    unsafe { sys::zpaq_sha1_free(s) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(err_from_last())
    }
}

/// Minimal SHA-256 helper using the libzpaq implementation.
pub fn sha256(bytes: &[u8]) -> Result<[u8; 32]> {
    clear_last_error();
    let s = unsafe { sys::zpaq_sha256_new() };
    if s.is_null() {
        return Err(err_from_last());
    }
    unsafe {
        for &b in bytes {
            sys::zpaq_sha256_put(s, b as c_int);
        }
    }
    let mut out = [0u8; 32];
    let rc = unsafe { sys::zpaq_sha256_result(s, out.as_mut_ptr()) };
    unsafe { sys::zpaq_sha256_free(s) };
    if rc == 0 {
        Ok(out)
    } else {
        Err(err_from_last())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_payloads() -> Vec<Vec<u8>> {
        vec![
            b"".to_vec(),
            b"hello zpaq".to_vec(),
            (0..1024).map(|i| (i * 31 % 251) as u8).collect(),
            (0..20_000)
                .map(|i| (i * 1315423911u64 as usize % 256) as u8)
                .collect(),
        ]
    }

    #[test]
    fn roundtrip_small() {
        let data = b"hello zpaq";
        let c = compress_to_vec(data, "1").expect("compress");
        let d = decompress_to_vec(&c).expect("decompress");
        assert_eq!(d, data);
    }

    #[test]
    fn roundtrip_random() {
        let data: Vec<u8> = (0..10_000).map(|i| (i * 31 % 251) as u8).collect();
        let c = compress_to_vec(&data, "2").expect("compress");
        let d = decompress_to_vec(&c).expect("decompress");
        assert_eq!(d, data);
    }

    #[test]
    fn compress_size_matches_vec_len() {
        for method in ["1", "2", "3", "4", "5", "x4.3ci1"] {
            for data in test_payloads() {
                let c = compress_to_vec(&data, method).expect("compress");
                let sz = compress_size(&data, method).expect("compress_size");
                assert_eq!(sz as usize, c.len(), "method={method}");
            }
        }
    }

    #[test]
    fn decompress_size_matches_output_len() {
        for method in ["1", "2", "x4.3ci1"] {
            for data in test_payloads() {
                let c = compress_to_vec(&data, method).expect("compress");
                let out = decompress_to_vec(&c).expect("decompress");
                let sz = decompress_size(&c).expect("decompress_size");
                assert_eq!(sz as usize, out.len(), "method={method}");
            }
        }
    }

    #[test]
    fn sha_vectors() {
        // "abc" test vectors
        let s1 = sha1(b"abc").expect("sha1");
        assert_eq!(hex::encode(s1), "a9993e364706816aba3e25717850c26c9cd0d89d");
        let s256 = sha256(b"abc").expect("sha256");
        assert_eq!(
            hex::encode(s256),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn random_bytes_len() {
        let b = random_bytes(256).expect("random_bytes");
        assert_eq!(b.len(), 256);
    }

    #[test]
    fn nul_in_method_errors() {
        let err = compress_stream(
            std::io::Cursor::new(b"hello"),
            Vec::<u8>::new(),
            "a\0b",
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, ZpaqError::NulInString));
    }

    #[test]
    fn callback_error_propagates() {
        struct FailingReader;

        impl std::io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
        }

        let err = compress_size_stream(FailingReader, "1", None, None).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("callback failed"));
    }
}
