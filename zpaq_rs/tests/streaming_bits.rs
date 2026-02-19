use zpaq_rs::{StreamingCompressor, compress_size};

#[test]
#[ignore = "streaming zpaq encoder is experimental; enable when stable"]
fn streaming_bits_matches_compress_size_minus_header() {
    let data = b"abababababababababababababababababababababababababababababababab";
    let method = "2";

    let mut stream = StreamingCompressor::new(method).expect("streaming compressor");
    for &b in data {
        stream.push(b).expect("push byte");
    }
    let bits = stream.bits();

    let header = compress_size(&[], method).unwrap_or(0) as f64 * 8.0;
    let size_bits = compress_size(data, method).unwrap_or(0) as f64 * 8.0;
    let expected = (size_bits - header).max(0.0);

    let diff = (bits - expected).abs();
    assert!(
        diff < 256.0,
        "stream bits mismatch: bits={bits:.3} expected={expected:.3} diff={diff:.3}"
    );
}
