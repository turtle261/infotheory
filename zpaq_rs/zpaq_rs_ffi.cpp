#include "zpaq/libzpaq.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cstdio>
#include <cstdlib>

// Platform-specific headers for file descriptor operations
#ifdef _WIN32
    #include <io.h>      // for _dup, _dup2, _close, _fileno
    #include <windows.h> // for LPWSTR
    #define dup _dup
    #define dup2 _dup2
    #define close _close
    #define fileno _fileno
    #define DEV_NULL "NUL"
    
    // On Windows, wmain signature uses LPWSTR*
    // This is renamed to zpaq_cli_main via preprocessor
    int zpaq_cli_main(int argc, LPWSTR* argv);
    
    // Helper to convert UTF-8 to wide string
    static std::wstring utf8_to_wide(const char* utf8) {
        if (!utf8) return std::wstring();
        int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, nullptr, 0);
        if (len == 0) return std::wstring();
        std::wstring result(len - 1, 0);
        MultiByteToWideChar(CP_UTF8, 0, utf8, -1, &result[0], len);
        return result;
    }
    
    // Wrapper to call zpaq_cli_main with char** arguments
    static int zpaq_cli_main_wrapper(int argc, const char** argv) {
        std::vector<std::wstring> wargs;
        std::vector<LPWSTR> wargv;
        wargs.reserve(argc);
        wargv.reserve(argc);
        
        for (int i = 0; i < argc; ++i) {
            wargs.push_back(utf8_to_wide(argv[i]));
            wargv.push_back(&wargs.back()[0]);
        }
        
        return zpaq_cli_main(argc, wargv.data());
    }
    
    #define zpaq_cli_main zpaq_cli_main_wrapper
#else
    #include <unistd.h>  // for dup, dup2, close
    #define DEV_NULL "/dev/null"
    
    // On UNIX, main signature uses const char**
    // This is renamed to zpaq_cli_main via preprocessor
    int zpaq_cli_main(int argc, const char** argv);
#endif

// Reverse-search helper:
// - Use system memrchr where it is known to exist (GNU/Linux + BSDs).
// - Use a local fallback on platforms where memrchr is typically unavailable
//   (notably macOS and Windows). IF PORTING: Here may be a point to change!
static inline const void* zpaq_memrchr(const void* s, int c, size_t n) {
#if defined(__APPLE__) || defined(_WIN32)
    if (!s || n == 0) return nullptr;
    const unsigned char* begin = static_cast<const unsigned char*>(s);
    const unsigned char* p = begin + n;
    const unsigned char ch = static_cast<unsigned char>(c);
    while (p != begin) {
        --p;
        if (*p == ch) return p;
    }
    return nullptr;
#else
    return ::memrchr(s, c, n);
#endif
}

namespace {

class LibZpaqError final : public std::exception {
  std::string msg_;

public:
  explicit LibZpaqError(std::string msg) : msg_(std::move(msg)) {}
  const char* what() const noexcept override { return msg_.c_str(); }
};

thread_local std::string g_last_error;

inline void clear_last_error() { g_last_error.clear(); }

inline void set_last_error(const char* msg) {
  g_last_error.assign(msg ? msg : "(null)");
}

// Sentinel for Rust callback failure (panic or explicit error)
constexpr int kRustCallbackError = -2;

struct CountingWriter final : public libzpaq::Writer {
  uint64_t n = 0;
  void put(int) override { ++n; }
  void write(const char* buf, int len) override {
    if (buf && len > 0) n += static_cast<uint64_t>(len);
  }
};

} // namespace

namespace {

constexpr int kPutBufferSize = 1 << 15;

} // namespace

extern "C" {

// ---------------- Error channel ----------------

void zpaq_clear_last_error() { clear_last_error(); }

const char* zpaq_last_error_ptr() { return g_last_error.empty() ? nullptr : g_last_error.c_str(); }

size_t zpaq_last_error_len() { return g_last_error.size(); }

void zpaq_set_last_error(const char* msg) { set_last_error(msg); }

// Copy error message into buf (not NUL-terminated unless space allows).
// Returns number of bytes copied.
size_t zpaq_last_error_copy(char* buf, size_t buf_len) {
  if (!buf || buf_len == 0) return 0;
  const size_t n = g_last_error.size();
  const size_t copy_n = n < buf_len ? n : buf_len;
  if (copy_n) std::memcpy(buf, g_last_error.data(), copy_n);
  return copy_n;
}

// ---------------- Reader/Writer callback shims ----------------

typedef int (*zpaq_get_fn)(void* ctx);
typedef int (*zpaq_read_fn)(void* ctx, char* buf, int n);

typedef int (*zpaq_put_fn)(void* ctx, int c);
typedef int (*zpaq_write_fn)(void* ctx, const char* buf, int n);

struct zpaq_reader {
  void* ctx;
  zpaq_get_fn get_cb;
  zpaq_read_fn read_cb;

  explicit zpaq_reader(void* c, zpaq_get_fn g, zpaq_read_fn r)
      : ctx(c), get_cb(g), read_cb(r) {}
};

struct zpaq_writer {
  void* ctx;
  zpaq_put_fn put_cb;
  zpaq_write_fn write_cb;

  explicit zpaq_writer(void* c, zpaq_put_fn p, zpaq_write_fn w)
      : ctx(c), put_cb(p), write_cb(w) {}
};

class RustReader final : public libzpaq::Reader {
  zpaq_reader inner_;

public:
  RustReader(void* ctx, zpaq_get_fn get_cb, zpaq_read_fn read_cb)
      : inner_(ctx, get_cb, read_cb) {}

  int get() override {
    if (inner_.get_cb) {
      const int v = inner_.get_cb(inner_.ctx);
      if (v == kRustCallbackError) libzpaq::error("Rust reader callback failed");
      return v;
    }
    if (!inner_.read_cb) return -1;
    char b = 0;
    const int n = inner_.read_cb(inner_.ctx, &b, 1);
    if (n == kRustCallbackError) libzpaq::error("Rust reader callback failed");
    if (n <= 0) return -1;
    return static_cast<unsigned char>(b);
  }

  int read(char* buf, int n) override {
    if (!buf || n <= 0) return 0;
    if (inner_.read_cb) {
      const int got = inner_.read_cb(inner_.ctx, buf, n);
      if (got == kRustCallbackError) libzpaq::error("Rust reader callback failed");
      return got;
    }
    // fallback to default implementation calling get()
    return libzpaq::Reader::read(buf, n);
  }
};

class RustWriter final : public libzpaq::Writer {
  zpaq_writer inner_;
  char buf_[kPutBufferSize];
  int used_ = 0;

  void flush_buf() {
    if (used_ <= 0) return;
    if (inner_.write_cb) {
      const int rc = inner_.write_cb(inner_.ctx, buf_, used_);
      if (rc == kRustCallbackError) libzpaq::error("Rust writer callback failed");
    } else if (inner_.put_cb) {
      for (int i = 0; i < used_; ++i) {
        const int rc = inner_.put_cb(inner_.ctx, static_cast<unsigned char>(buf_[i]));
        if (rc == kRustCallbackError) libzpaq::error("Rust writer callback failed");
      }
    }
    used_ = 0;
  }

public:
  RustWriter(void* ctx, zpaq_put_fn put_cb, zpaq_write_fn write_cb)
      : inner_(ctx, put_cb, write_cb) {}

  ~RustWriter() override { flush_buf(); }

  void put(int c) override {
    if (!inner_.write_cb && !inner_.put_cb) return;
    buf_[used_++] = static_cast<char>(c);
    if (used_ == kPutBufferSize) flush_buf();
  }

  void write(const char* buf, int n) override {
    if (!buf || n <= 0) return;
    flush_buf();
    if (inner_.write_cb) {
      const int rc = inner_.write_cb(inner_.ctx, buf, n);
      if (rc == kRustCallbackError) libzpaq::error("Rust writer callback failed");
      return;
    }
    // fallback to per-byte put
    for (int i = 0; i < n; ++i) put(static_cast<unsigned char>(buf[i]));
  }
};

// Opaque handles exposed to Rust.
RustReader* zpaq_reader_new(void* ctx, zpaq_get_fn get_cb, zpaq_read_fn read_cb) {
  clear_last_error();
  try {
    return new RustReader(ctx, get_cb, read_cb);
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_reader_free(RustReader* r) {
  delete r;
}

RustWriter* zpaq_writer_new(void* ctx, zpaq_put_fn put_cb, zpaq_write_fn write_cb) {
  clear_last_error();
  try {
    return new RustWriter(ctx, put_cb, write_cb);
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_writer_free(RustWriter* w) {
  delete w;
}

// ---------------- Top-level convenience API ----------------

int zpaq_compress(RustReader* in, RustWriter* out, const char* method, const char* filename,
                 const char* comment, int dosha1) {
  clear_last_error();
  try {
    libzpaq::compress(in, out, method, filename, comment, dosha1 != 0);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompress(RustReader* in, RustWriter* out) {
  clear_last_error();
  try {
    libzpaq::decompress(in, out);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

static int method_block_size(const char* method) {
  int bs = 4;
  if (method && method[0] && method[1] >= '0' && method[1] <= '9') {
    bs = method[1] - '0';
    if (method[2] >= '0' && method[2] <= '9') bs = bs * 10 + (method[2] - '0');
    if (bs > 11) bs = 11;
  }
  // Same formula as libzpaq::compress
  const int block = (0x100000 << bs) - 4096;
  return block > 0 ? block : (1 << 20);
}

int zpaq_compress_size(RustReader* in, const char* method, const char* filename, const char* comment, int dosha1,
                      uint64_t* out_size) {
  clear_last_error();
  try {
    CountingWriter out;
    libzpaq::compress(in, &out, method, filename, comment, dosha1 != 0);
    if (out_size) *out_size = out.n;
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompress_size(RustReader* in, uint64_t* out_size) {
  clear_last_error();
  try {
    CountingWriter out;
    libzpaq::decompress(in, &out);
    if (out_size) *out_size = out.n;
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compress_size_parallel(RustReader* in, const char* method, const char* filename, const char* comment, int dosha1,
                               int threads, uint64_t* out_size) {
  clear_last_error();
  try {
    if (!in) return -1;
    if (threads <= 1) {
      CountingWriter out;
      libzpaq::compress(in, &out, method, filename, comment, dosha1 != 0);
      if (out_size) *out_size = out.n;
      return 0;
    }

    const int bs = method_block_size(method);
    struct Block {
      size_t idx;
      std::string data;
    };

    std::mutex mu;
    std::condition_variable cv;
    std::deque<Block> q;
    bool done = false;
    bool failed = false;
    std::string fail_msg;
    std::vector<uint64_t> sizes;

    auto worker = [&]() {
      for (;;) {
        Block blk;
        {
          std::unique_lock<std::mutex> lock(mu);
          cv.wait(lock, [&] { return failed || done || !q.empty(); });
          if (failed) return;
          if (q.empty()) {
            if (done) return;
            continue;
          }
          blk = std::move(q.front());
          q.pop_front();
        }

        try {
          libzpaq::StringBuffer sb(bs);
          sb.write(nullptr, blk.data.size());
          if (blk.data.size()) std::memcpy(sb.data(), blk.data.data(), blk.data.size());
          sb.resize(blk.data.size());

          CountingWriter out;
          const char* fn = (blk.idx == 0) ? filename : nullptr;
          const char* cm = (blk.idx == 0) ? comment : nullptr;
          libzpaq::compressBlock(&sb, &out, method, fn, cm, dosha1 != 0);

          {
            std::lock_guard<std::mutex> lock(mu);
            if (blk.idx >= sizes.size()) sizes.resize(blk.idx + 1, 0);
            sizes[blk.idx] = out.n;
          }
        } catch (const std::exception& e) {
          std::lock_guard<std::mutex> lock(mu);
          if (!failed) {
            failed = true;
            fail_msg = e.what();
          }
          cv.notify_all();
          return;
        }
      }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(threads));
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);

    size_t idx = 0;
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mu);
        if (failed) break;
      }

      std::string buf;
      buf.resize(static_cast<size_t>(bs));
      const int n = in->read(&buf[0], bs);
      if (n <= 0) break;
      buf.resize(static_cast<size_t>(n));

      {
        std::lock_guard<std::mutex> lock(mu);
        q.push_back(Block{idx++, std::move(buf)});
      }
      cv.notify_one();
    }

    {
      std::lock_guard<std::mutex> lock(mu);
      done = true;
    }
    cv.notify_all();
    for (auto& t : pool) t.join();

    {
      std::lock_guard<std::mutex> lock(mu);
      if (failed) {
        set_last_error(fail_msg.c_str());
        return -1;
      }
    }

    uint64_t total = 0;
    for (uint64_t s : sizes) total += s;
    if (out_size) *out_size = total;
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

static bool parse_last_archive_mb(const char* s, size_t n, double* out_mb) {
  if (!s || n == 0 || !out_mb) return false;
  // Look for the last occurrence of "= <num> MB" in the captured stderr.
  // The summary line in zpaq.cpp is:
  //   "... = %1.6f MB\n"
  const char* end = s + n;
  const char* p = end;
  while (p > s) {
    // Find previous '='
    const char* eq = (const char*)zpaq_memrchr(s, '=', (size_t)(p - s));
    if (!eq) break;
    const char* q = eq + 1;
    while (q < end && (*q == ' ' || *q == '\t')) ++q;
    char* num_end = nullptr;
    const double val = std::strtod(q, &num_end);
    if (num_end && num_end > q) {
      const char* r = num_end;
      while (r < end && (*r == ' ' || *r == '\t')) ++r;
      if (r + 2 <= end && r[0] == 'M' && r[1] == 'B') {
        *out_mb = val;
        return true;
      }
    }
    p = eq;
  }
  return false;
}

int zpaq_jidac_add_archive_size_file(const char* path, const char* method, int threads, uint64_t* out_archive_size_bytes) {
  clear_last_error();
  try {
    if (!path || !*path || !method || !*method || !out_archive_size_bytes) return -1;

    static std::mutex g_mu;
    std::lock_guard<std::mutex> lock(g_mu);

    fflush(stdout);
    fflush(stderr);

    int old_stderr = dup(fileno(stderr));
    int old_stdout = dup(fileno(stdout));
    if (old_stderr < 0 || old_stdout < 0) {
      if (old_stderr >= 0) close(old_stderr);
      if (old_stdout >= 0) close(old_stdout);
      set_last_error("dup() failed");
      return -1;
    }

    // Capture stderr; discard stdout.
    // Note: open_memstream() has no file descriptor, so we use tmpfile().
    FILE* err_stream = tmpfile();
    FILE* out_stream = fopen(DEV_NULL, "w");
    if (!err_stream || !out_stream) {
      if (err_stream) fclose(err_stream);
      if (out_stream) fclose(out_stream);
      close(old_stderr);
      close(old_stdout);
      set_last_error("failed to redirect stdio");
      return -1;
    }
    if (dup2(fileno(err_stream), fileno(stderr)) < 0 || dup2(fileno(out_stream), fileno(stdout)) < 0) {
      fclose(err_stream);
      fclose(out_stream);
      close(old_stderr);
      close(old_stdout);
      set_last_error("dup2() failed");
      return -1;
    }

    // Build argv like: zpaq add "" <path> -method <method> -threads <N>
    std::string threads_s = std::to_string(threads);
    const char* argv[9];
    int argc = 0;
    argv[argc++] = "zpaq";
    argv[argc++] = "add";
    argv[argc++] = "";
    argv[argc++] = path;
    argv[argc++] = "-method";
    argv[argc++] = method;
    argv[argc++] = "-threads";
    argv[argc++] = threads_s.c_str();
    argv[argc++] = nullptr;

    int rc = 0;
    try {
      rc = zpaq_cli_main(argc - 1, argv);
    } catch (const std::exception& e) {
      rc = 2;
      set_last_error(e.what());
    }

    fflush(stdout);
    fflush(stderr);

    // Read captured stderr
    std::string captured;
    fseek(err_stream, 0, SEEK_END);
    long n = ftell(err_stream);
    if (n > 0) {
      captured.resize((size_t)n);
      rewind(err_stream);
      const size_t got = fread(&captured[0], 1, captured.size(), err_stream);
      captured.resize(got);
    }
    fclose(err_stream);
    fclose(out_stream);

    // Restore
    dup2(old_stderr, fileno(stderr));
    dup2(old_stdout, fileno(stdout));
    close(old_stderr);
    close(old_stdout);

    double archive_mb = 0.0;
    const bool ok = parse_last_archive_mb(captured.data(), captured.size(), &archive_mb);
    if (!ok) {
      set_last_error("failed to parse zpaq summary output");
      return -1;
    }

    const double bytes_d = archive_mb * 1000000.0;
    const uint64_t bytes = bytes_d <= 0.0 ? 0ULL : (uint64_t)(bytes_d + 0.5);
    *out_archive_size_bytes = bytes;
    return rc == 0 ? 0 : -1;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

// ---------------- StringBuffer ----------------

libzpaq::StringBuffer* zpaq_string_buffer_new(size_t initial) {
  clear_last_error();
  try {
    return new libzpaq::StringBuffer(initial);
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_string_buffer_free(libzpaq::StringBuffer* sb) { delete sb; }

size_t zpaq_string_buffer_size(const libzpaq::StringBuffer* sb) {
  return sb ? sb->size() : 0;
}

size_t zpaq_string_buffer_remaining(const libzpaq::StringBuffer* sb) {
  return sb ? sb->remaining() : 0;
}

const unsigned char* zpaq_string_buffer_data(libzpaq::StringBuffer* sb) {
  if (!sb) return nullptr;
  return sb->size() ? sb->data() : nullptr;
}

void zpaq_string_buffer_reset(libzpaq::StringBuffer* sb) {
  if (sb) sb->reset();
}

void zpaq_string_buffer_resize(libzpaq::StringBuffer* sb, size_t n) {
  if (sb) sb->resize(n);
}

// ---------------- Compressor ----------------

libzpaq::Compressor* zpaq_compressor_new() {
  clear_last_error();
  try {
    return new libzpaq::Compressor();
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_compressor_free(libzpaq::Compressor* c) { delete c; }

int zpaq_compressor_set_output(libzpaq::Compressor* c, RustWriter* out) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->setOutput(out);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_set_input(libzpaq::Compressor* c, RustReader* in) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->setInput(in);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_write_tag(libzpaq::Compressor* c) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->writeTag();
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_start_block_level(libzpaq::Compressor* c, int level) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->startBlock(level);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_start_block_method(libzpaq::Compressor* c, const char* method) {
  clear_last_error();
  try {
    if (!c || !method || !*method) return -1;
    const char type = method[0];
    if (!(type == 'x' || type == 's' || type == 'i' || type == '0')) {
      set_last_error("method must start with one of: x, s, i, 0 (or use numeric 1..3)");
      return -1;
    }
    int args[9] = {0};
    std::string config = libzpaq::makeConfig(method, args);
    if (args[1] != 0) {
      std::string msg = "method uses block preprocessing (args[1]=" + std::to_string(args[1]) +
                        "); not streamable";
      set_last_error(msg.c_str());
      return -1;
    }
    libzpaq::StringBuffer pcomp_cmd;
    c->startBlock(config.c_str(), args, &pcomp_cmd);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_start_block_hcomp(libzpaq::Compressor* c, const char* hcomp_bytecode) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->startBlock(hcomp_bytecode);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_set_verify(libzpaq::Compressor* c, int verify) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->setVerify(verify != 0);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_start_segment(libzpaq::Compressor* c, const char* filename, const char* comment) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->startSegment(filename, comment);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_post_process(libzpaq::Compressor* c, const char* pcomp_bytecode, int len) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->postProcess(pcomp_bytecode, len);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_compress(libzpaq::Compressor* c, int n) {
  clear_last_error();
  try {
    if (!c) return -1;
    const bool more = c->compress(n);
    return more ? 1 : 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_end_segment(libzpaq::Compressor* c, const unsigned char* sha1_20_or_null) {
  clear_last_error();
  try {
    if (!c) return -1;
    const char* ptr = sha1_20_or_null ? reinterpret_cast<const char*>(sha1_20_or_null) : nullptr;
    c->endSegment(ptr);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_end_segment_checksum(libzpaq::Compressor* c, int64_t* size_out, int dosha1,
                                        unsigned char out_hash20[20]) {
  clear_last_error();
  try {
    if (!c) return -1;
    char* r = c->endSegmentChecksum(size_out, dosha1 != 0);
    if (!r) return 0;
    if (out_hash20) std::memcpy(out_hash20, r, 20);
    return 1;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int64_t zpaq_compressor_get_size(libzpaq::Compressor* c) {
  return c ? c->getSize() : 0;
}

double zpaq_compressor_get_bits(libzpaq::Compressor* c) {
  return c ? c->getEncodedBits() : 0.0;
}

int zpaq_compressor_get_checksum(libzpaq::Compressor* c, unsigned char out_hash20[20]) {
  clear_last_error();
  try {
    if (!c || !out_hash20) return -1;
    const char* p = c->getChecksum();
    if (!p) return 0;
    std::memcpy(out_hash20, p, 20);
    return 1;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_compressor_end_block(libzpaq::Compressor* c) {
  clear_last_error();
  try {
    if (!c) return -1;
    c->endBlock();
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

// ---------------- Decompresser ----------------

libzpaq::Decompresser* zpaq_decompresser_new() {
  clear_last_error();
  try {
    return new libzpaq::Decompresser();
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_decompresser_free(libzpaq::Decompresser* d) { delete d; }

int zpaq_decompresser_set_input(libzpaq::Decompresser* d, RustReader* in) {
  clear_last_error();
  try {
    if (!d) return -1;
    d->setInput(in);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_find_block(libzpaq::Decompresser* d, double* mem_out) {
  clear_last_error();
  try {
    if (!d) return -1;
    return d->findBlock(mem_out) ? 1 : 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_find_filename(libzpaq::Decompresser* d, RustWriter* filename_out) {
  clear_last_error();
  try {
    if (!d) return -1;
    return d->findFilename(filename_out) ? 1 : 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_read_comment(libzpaq::Decompresser* d, RustWriter* comment_out) {
  clear_last_error();
  try {
    if (!d) return -1;
    d->readComment(comment_out);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_set_output(libzpaq::Decompresser* d, RustWriter* out) {
  clear_last_error();
  try {
    if (!d) return -1;
    d->setOutput(out);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_decompress(libzpaq::Decompresser* d, int n) {
  clear_last_error();
  try {
    if (!d) return -1;
    const bool more = d->decompress(n);
    return more ? 1 : 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_read_segment_end(libzpaq::Decompresser* d, unsigned char out_21[21]) {
  clear_last_error();
  try {
    if (!d) return -1;
    char tmp[21];
    d->readSegmentEnd(tmp);
    if (out_21) std::memcpy(out_21, tmp, 21);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_decompresser_buffered(libzpaq::Decompresser* d) {
  return d ? d->buffered() : 0;
}

// ---------------- SHA1 / SHA256 ----------------

libzpaq::SHA1* zpaq_sha1_new() {
  clear_last_error();
  try {
    return new libzpaq::SHA1();
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_sha1_free(libzpaq::SHA1* s) { delete s; }

void zpaq_sha1_put(libzpaq::SHA1* s, int c) {
  if (s) s->put(c);
}

void zpaq_sha1_write(libzpaq::SHA1* s, const char* buf, int64_t n) {
  if (s) s->write(buf, n);
}

uint64_t zpaq_sha1_usize(const libzpaq::SHA1* s) { return s ? s->usize() : 0; }

double zpaq_sha1_size(const libzpaq::SHA1* s) { return s ? s->size() : 0.0; }

int zpaq_sha1_result(libzpaq::SHA1* s, unsigned char out_hash20[20]) {
  clear_last_error();
  try {
    if (!s || !out_hash20) return -1;
    const char* p = s->result();
    std::memcpy(out_hash20, p, 20);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

libzpaq::SHA256* zpaq_sha256_new() {
  clear_last_error();
  try {
    return new libzpaq::SHA256();
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_sha256_free(libzpaq::SHA256* s) { delete s; }

void zpaq_sha256_put(libzpaq::SHA256* s, int c) {
  if (s) s->put(c);
}

uint64_t zpaq_sha256_usize(const libzpaq::SHA256* s) { return s ? s->usize() : 0; }

double zpaq_sha256_size(const libzpaq::SHA256* s) { return s ? s->size() : 0.0; }

int zpaq_sha256_result(libzpaq::SHA256* s, unsigned char out_hash32[32]) {
  clear_last_error();
  try {
    if (!s || !out_hash32) return -1;
    const char* p = s->result();
    std::memcpy(out_hash32, p, 32);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

// ---------------- AES_CTR / scrypt / random ----------------

libzpaq::AES_CTR* zpaq_aes_ctr_new(const char* key, int keylen, const char* iv) {
  clear_last_error();
  try {
    return new libzpaq::AES_CTR(key, keylen, iv);
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  }
}

void zpaq_aes_ctr_free(libzpaq::AES_CTR* a) { delete a; }

int zpaq_aes_ctr_encrypt_slice(libzpaq::AES_CTR* a, char* buf, int n, uint64_t offset) {
  clear_last_error();
  try {
    if (!a) return -1;
    a->encrypt(buf, n, offset);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_aes_ctr_encrypt_block(libzpaq::AES_CTR* a, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
                              unsigned char out_ct16[16]) {
  clear_last_error();
  try {
    if (!a || !out_ct16) return -1;
    a->encrypt(s0, s1, s2, s3, out_ct16);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_stretch_key(unsigned char out32[32], const unsigned char key32[32], const unsigned char salt32[32]) {
  clear_last_error();
  try {
    if (!out32 || !key32 || !salt32) return -1;
    libzpaq::stretchKey(reinterpret_cast<char*>(out32), reinterpret_cast<const char*>(key32),
                        reinterpret_cast<const char*>(salt32));
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

int zpaq_random(unsigned char* buf, int n) {
  clear_last_error();
  try {
    if (!buf || n < 0) return -1;
    libzpaq::random(reinterpret_cast<char*>(buf), n);
    return 0;
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  }
}

uint16_t zpaq_to_u16(const char* p) {
  return static_cast<uint16_t>(libzpaq::toU16(p));
}

} // extern "C"
