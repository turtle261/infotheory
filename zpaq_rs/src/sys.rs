//! Raw FFI bindings to the C++ shim around libzpaq.
//!
//! These are unsafe and should be used through the safe wrappers in `lib.rs`.

#![allow(dead_code)]

use std::os::raw::{c_char, c_double, c_int, c_uchar, c_uint, c_ulonglong, c_void};

#[repr(C)]
pub struct RustReader {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RustWriter {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Compressor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Decompresser {
    _private: [u8; 0],
}

#[repr(C)]
pub struct StringBuffer {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SHA1 {
    _private: [u8; 0],
}

#[repr(C)]
pub struct SHA256 {
    _private: [u8; 0],
}

#[repr(C)]
pub struct AES_CTR {
    _private: [u8; 0],
}

pub const RUST_CALLBACK_ERROR: c_int = -2;

pub type GetFn = Option<unsafe extern "C" fn(ctx: *mut c_void) -> c_int>;
pub type ReadFn =
    Option<unsafe extern "C" fn(ctx: *mut c_void, buf: *mut c_char, n: c_int) -> c_int>;
pub type PutFn = Option<unsafe extern "C" fn(ctx: *mut c_void, c: c_int) -> c_int>;
pub type WriteFn =
    Option<unsafe extern "C" fn(ctx: *mut c_void, buf: *const c_char, n: c_int) -> c_int>;

#[link(name = "zpaq_rs_ffi", kind = "static")]
unsafe extern "C" {
    // Error channel
    pub fn zpaq_clear_last_error();
    pub fn zpaq_last_error_ptr() -> *const c_char;
    pub fn zpaq_last_error_len() -> usize;
    pub fn zpaq_last_error_copy(buf: *mut c_char, buf_len: usize) -> usize;
    pub fn zpaq_set_last_error(msg: *const c_char);

    // Reader/Writer
    pub fn zpaq_reader_new(ctx: *mut c_void, get_cb: GetFn, read_cb: ReadFn) -> *mut RustReader;
    pub fn zpaq_reader_free(r: *mut RustReader);
    pub fn zpaq_writer_new(ctx: *mut c_void, put_cb: PutFn, write_cb: WriteFn) -> *mut RustWriter;
    pub fn zpaq_writer_free(w: *mut RustWriter);

    // Convenience top-level
    pub fn zpaq_compress(
        input: *mut RustReader,
        output: *mut RustWriter,
        method: *const c_char,
        filename: *const c_char,
        comment: *const c_char,
        dosha1: c_int,
    ) -> c_int;
    pub fn zpaq_decompress(input: *mut RustReader, output: *mut RustWriter) -> c_int;

    // Size-only convenience (avoids copying compressed/decompressed bytes into Rust)
    pub fn zpaq_compress_size(
        input: *mut RustReader,
        method: *const c_char,
        filename: *const c_char,
        comment: *const c_char,
        dosha1: c_int,
        out_size: *mut u64,
    ) -> c_int;
    pub fn zpaq_compress_size_parallel(
        in_: *mut RustReader,
        method: *const ::std::os::raw::c_char,
        filename: *const ::std::os::raw::c_char,
        comment: *const ::std::os::raw::c_char,
        dosha1: ::std::os::raw::c_int,
        threads: ::std::os::raw::c_int,
        out_size: *mut u64,
    ) -> ::std::os::raw::c_int;
    pub fn zpaq_decompress_size(input: *mut RustReader, out_size: *mut u64) -> c_int;

    // JIDAC (zpaq.cpp) convenience
    pub fn zpaq_jidac_add_archive_size_file(
        path: *const c_char,
        method: *const c_char,
        threads: c_int,
        out_archive_size_bytes: *mut u64,
    ) -> c_int;

    // StringBuffer
    pub fn zpaq_string_buffer_new(initial: usize) -> *mut StringBuffer;
    pub fn zpaq_string_buffer_free(sb: *mut StringBuffer);
    pub fn zpaq_string_buffer_size(sb: *const StringBuffer) -> usize;
    pub fn zpaq_string_buffer_remaining(sb: *const StringBuffer) -> usize;
    pub fn zpaq_string_buffer_data(sb: *mut StringBuffer) -> *const c_uchar;
    pub fn zpaq_string_buffer_reset(sb: *mut StringBuffer);
    pub fn zpaq_string_buffer_resize(sb: *mut StringBuffer, n: usize);

    // Compressor
    pub fn zpaq_compressor_new() -> *mut Compressor;
    pub fn zpaq_compressor_free(c: *mut Compressor);
    pub fn zpaq_compressor_set_output(c: *mut Compressor, out: *mut RustWriter) -> c_int;
    pub fn zpaq_compressor_set_input(c: *mut Compressor, input: *mut RustReader) -> c_int;
    pub fn zpaq_compressor_write_tag(c: *mut Compressor) -> c_int;
    pub fn zpaq_compressor_start_block_level(c: *mut Compressor, level: c_int) -> c_int;
    pub fn zpaq_compressor_start_block_method(c: *mut Compressor, method: *const c_char) -> c_int;
    pub fn zpaq_compressor_start_block_hcomp(c: *mut Compressor, hcomp: *const c_char) -> c_int;
    pub fn zpaq_compressor_set_verify(c: *mut Compressor, verify: c_int) -> c_int;
    pub fn zpaq_compressor_start_segment(
        c: *mut Compressor,
        filename: *const c_char,
        comment: *const c_char,
    ) -> c_int;
    pub fn zpaq_compressor_post_process(
        c: *mut Compressor,
        pcomp: *const c_char,
        len: c_int,
    ) -> c_int;
    pub fn zpaq_compressor_compress(c: *mut Compressor, n: c_int) -> c_int;
    pub fn zpaq_compressor_end_segment(c: *mut Compressor, sha1_or_null: *const c_uchar) -> c_int;
    pub fn zpaq_compressor_end_segment_checksum(
        c: *mut Compressor,
        size_out: *mut i64,
        dosha1: c_int,
        out_hash20: *mut c_uchar,
    ) -> c_int;
    pub fn zpaq_compressor_get_size(c: *mut Compressor) -> i64;
    pub fn zpaq_compressor_get_bits(c: *mut Compressor) -> c_double;
    pub fn zpaq_compressor_get_checksum(c: *mut Compressor, out_hash20: *mut c_uchar) -> c_int;
    pub fn zpaq_compressor_end_block(c: *mut Compressor) -> c_int;

    // Decompresser
    pub fn zpaq_decompresser_new() -> *mut Decompresser;
    pub fn zpaq_decompresser_free(d: *mut Decompresser);
    pub fn zpaq_decompresser_set_input(d: *mut Decompresser, input: *mut RustReader) -> c_int;
    pub fn zpaq_decompresser_find_block(d: *mut Decompresser, mem_out: *mut c_double) -> c_int;
    pub fn zpaq_decompresser_find_filename(
        d: *mut Decompresser,
        filename_out: *mut RustWriter,
    ) -> c_int;
    pub fn zpaq_decompresser_read_comment(
        d: *mut Decompresser,
        comment_out: *mut RustWriter,
    ) -> c_int;
    pub fn zpaq_decompresser_set_output(d: *mut Decompresser, out: *mut RustWriter) -> c_int;
    pub fn zpaq_decompresser_decompress(d: *mut Decompresser, n: c_int) -> c_int;
    pub fn zpaq_decompresser_read_segment_end(d: *mut Decompresser, out_21: *mut c_uchar) -> c_int;
    pub fn zpaq_decompresser_buffered(d: *mut Decompresser) -> c_int;

    // SHA1 / SHA256
    pub fn zpaq_sha1_new() -> *mut SHA1;
    pub fn zpaq_sha1_free(s: *mut SHA1);
    pub fn zpaq_sha1_put(s: *mut SHA1, c: c_int);
    pub fn zpaq_sha1_write(s: *mut SHA1, buf: *const c_char, n: i64);
    pub fn zpaq_sha1_usize(s: *const SHA1) -> c_ulonglong;
    pub fn zpaq_sha1_size(s: *const SHA1) -> c_double;
    pub fn zpaq_sha1_result(s: *mut SHA1, out_hash20: *mut c_uchar) -> c_int;

    pub fn zpaq_sha256_new() -> *mut SHA256;
    pub fn zpaq_sha256_free(s: *mut SHA256);
    pub fn zpaq_sha256_put(s: *mut SHA256, c: c_int);
    pub fn zpaq_sha256_usize(s: *const SHA256) -> c_ulonglong;
    pub fn zpaq_sha256_size(s: *const SHA256) -> c_double;
    pub fn zpaq_sha256_result(s: *mut SHA256, out_hash32: *mut c_uchar) -> c_int;

    // AES CTR and utilities
    pub fn zpaq_aes_ctr_new(key: *const c_char, keylen: c_int, iv: *const c_char) -> *mut AES_CTR;
    pub fn zpaq_aes_ctr_free(a: *mut AES_CTR);
    pub fn zpaq_aes_ctr_encrypt_slice(
        a: *mut AES_CTR,
        buf: *mut c_char,
        n: c_int,
        offset: c_ulonglong,
    ) -> c_int;
    pub fn zpaq_aes_ctr_encrypt_block(
        a: *mut AES_CTR,
        s0: c_uint,
        s1: c_uint,
        s2: c_uint,
        s3: c_uint,
        out_ct16: *mut c_uchar,
    ) -> c_int;

    pub fn zpaq_stretch_key(
        out32: *mut c_uchar,
        key32: *const c_uchar,
        salt32: *const c_uchar,
    ) -> c_int;
    pub fn zpaq_random(buf: *mut c_uchar, n: c_int) -> c_int;
    pub fn zpaq_to_u16(p: *const c_char) -> u16;
}
