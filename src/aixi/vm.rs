//! VM-backed AIXI environment (libvirt/virsh).
//!
//! This environment is intentionally task-agnostic. All experiment logic
//! (actions, reward policy, observation policy, and info-theoretic pruning)
//! is configured via `VmEnvironmentConfig`.

use crate::aixi::common::{Action, PerceptVal, RandomGenerator, Reward};
use crate::aixi::environment::Environment;
use crate::{RateBackend, cross_entropy_rate_backend, entropy_rate_backend, marginal_entropy_bytes};
use rosaplus::RosaPlus;
use rwkvzip::Compressor;
use rwkvzip::coders::softmax_pdf_inplace;
use ssh2::Session;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};
use virt::connect::Connect;
use virt::domain::Domain;
use virt::domain_snapshot::DomainSnapshot;
use virt::sys;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PayloadEncoding {
    Utf8,
    Hex,
}

impl PayloadEncoding {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "utf8" | "text" => Some(Self::Utf8),
            "hex" => Some(Self::Hex),
            _ => None,
        }
    }

    pub fn decode(self, s: &str) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Utf8 => Ok(s.as_bytes().to_vec()),
            Self::Hex => hex_decode(s),
        }
    }

    pub fn encode(self, bytes: &[u8]) -> String {
        match self {
            Self::Utf8 => String::from_utf8_lossy(bytes).to_string(),
            Self::Hex => hex_encode(bytes),
        }
    }
}

fn hex_decode(s: &str) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut buf = 0u8;
    let mut high = true;
    for c in s.bytes() {
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(anyhow::anyhow!("invalid hex byte: {}", c as char)),
        };
        if high {
            buf = v << 4;
            high = false;
        } else {
            buf |= v;
            out.push(buf);
            high = true;
        }
    }
    if !high {
        return Err(anyhow::anyhow!("hex string has odd length"));
    }
    Ok(out)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(hex_digit(b >> 4));
        s.push(hex_digit(b & 0x0F));
    }
    s
}

fn hex_digit(v: u8) -> char {
    match v {
        0..=9 => (b'0' + v) as char,
        _ => (b'a' + (v - 10)) as char,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VmTransport {
    Serial,
    Ssh,
}

impl VmTransport {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "serial" => Some(Self::Serial),
            "ssh" => Some(Self::Ssh),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VmSshCommand {
    pub command: String,
    pub args: Vec<String>,
    pub stdin_payload: bool,
    pub env: Vec<(String, String)>,
    pub workdir: Option<String>,
    pub run_as: Option<String>,
}

#[derive(Clone, Debug)]
pub enum VmSshProvisionStep {
    Upload {
        local: String,
        remote: String,
        mode: Option<u32>,
    },
    UploadText {
        remote: String,
        text: String,
        mode: Option<u32>,
    },
    Run {
        command: VmSshCommand,
    },
    RunScript {
        local: String,
        remote: String,
        mode: Option<u32>,
    },
}

#[derive(Clone, Debug)]
pub struct VmSshConfig {
    pub host: String,
    pub port: u16,
    pub user: String,
    pub password: Option<String>,
    pub private_key: Option<String>,
    pub public_key: Option<String>,
    pub passphrase: Option<String>,
    pub connect_timeout_ms: u64,
    pub retry_interval_ms: u64,
    pub action_command: VmSshCommand,
    pub action_commands: Option<Vec<VmSshCommand>>,
    pub provision_steps: Vec<VmSshProvisionStep>,
    pub provision_always: bool,
    pub ready_command: Option<VmSshCommand>,
}

#[derive(Clone, Debug)]
pub struct VmConsoleConfig {
    pub socket_path: String,
    pub timeout_ms: u64,
}

#[derive(Clone, Debug)]
pub struct VmProtocolConfig {
    pub action_prefix: String,
    pub action_suffix: String,
    pub obs_prefix: String,
    pub rew_prefix: String,
    pub done_prefix: String,
    pub data_prefix: String,
    pub wire_encoding: PayloadEncoding,
}

impl Default for VmProtocolConfig {
    fn default() -> Self {
        Self {
            action_prefix: "ACT ".to_string(),
            action_suffix: "\n".to_string(),
            obs_prefix: "OBS ".to_string(),
            rew_prefix: "REW ".to_string(),
            done_prefix: "DONE ".to_string(),
            data_prefix: "DATA ".to_string(),
            wire_encoding: PayloadEncoding::Hex,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VmActionSpec {
    pub name: Option<String>,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug)]
pub enum FuzzMutator {
    FlipBit,
    FlipByte,
    InsertByte,
    DeleteByte,
    SpliceSeed,
    ResetSeed,
    Havoc,
}

#[derive(Clone, Debug)]
pub struct VmFuzzConfig {
    pub seeds: Vec<Vec<u8>>,
    pub mutators: Vec<FuzzMutator>,
    pub min_len: usize,
    pub max_len: usize,
    pub dictionary: Vec<Vec<u8>>,
    pub rng_seed: u64,
}

#[derive(Clone, Debug)]
pub enum VmActionSource {
    Literal(Vec<VmActionSpec>),
    Fuzz(VmFuzzConfig),
}

#[derive(Clone, Copy, Debug)]
pub enum VmObservationPolicy {
    FromGuest,
    OutputHash,
    RawOutput,
}

#[derive(Clone, Copy, Debug)]
pub enum VmObservationStreamMode {
    PadTruncate,
    Pad,
    Truncate,
}

#[derive(Clone, Copy, Debug)]
pub enum VmTraceFraming {
    Len32Le,
    Line,
}

#[derive(Clone, Debug)]
pub struct VmTraceConfig {
    pub socket_path: Option<String>,
    pub timeout_ms: u64,
    pub max_bytes: usize,
    pub framing: VmTraceFraming,
    pub encoding: PayloadEncoding,
    pub line_prefix: Option<String>,
    pub reset_on_episode: bool,
    pub ssh_command: Option<VmSshCommand>,
}

#[derive(Clone, Debug)]
pub enum VmRewardPolicy {
    FromGuest,
    Pattern {
        pattern: String,
        base_reward: i64,
        bonus_reward: i64,
    },
    EntropyReduction {
        baseline_bytes: Vec<u8>,
        max_order: i64,
        scale: f64,
    },
    TraceEntropy {
        max_order: i64,
        scale: f64,
        normalize: bool,
    },
}

#[derive(Clone, Debug)]
pub struct VmActionFilter {
    pub min_entropy: Option<f64>,
    pub max_entropy: Option<f64>,
    pub min_intrinsic_dependence: Option<f64>,
    pub min_novelty: Option<f64>,
    pub novelty_prior: Option<Vec<u8>>,
    pub max_order: i64,
    pub reject_reward: Option<i64>,
}

#[derive(Clone, Copy, Debug)]
pub enum ResourceApplyMode {
    Live,
    Config,
    Both,
}

#[derive(Clone, Debug)]
pub struct VmResourceLimits {
    pub vcpus: Option<u32>,
    pub memory_mib: Option<u64>,
    pub apply_mode: ResourceApplyMode,
}

#[derive(Clone, Debug)]
pub struct VmHook {
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct VmHooks {
    /// Host-side hooks for experiment-specific setup (e.g. overlay management).
    pub pre_revert: Vec<VmHook>,
    pub post_revert: Vec<VmHook>,
}

#[derive(Clone)]
pub struct VmEnvironmentConfig {
    pub libvirt_uri: Option<String>,
    pub domain: String,
    pub snapshot: String,
    pub transport: VmTransport,
    pub console: Option<VmConsoleConfig>,
    pub ssh: Option<VmSshConfig>,
    pub protocol: VmProtocolConfig,
    pub stats_backend: RateBackend,
    pub trace: Option<VmTraceConfig>,
    pub auto_snapshot: bool,
    pub episode_steps: usize,
    pub step_cost: i64,
    pub debug_mode: bool,
    pub boot_ready: Option<String>,
    pub boot_timeout_ms: u64,
    pub step_timeout_ms: u64,
    pub max_response_lines: usize,
    pub max_output_bytes: usize,
    pub observation_bits: usize,
    pub reward_bits: usize,
    pub observation_policy: VmObservationPolicy,
    /// Observation symbols per action (used to normalize raw output streams).
    pub observation_stream_len: usize,
    /// Stream normalization policy for raw output observations.
    pub observation_stream_mode: VmObservationStreamMode,
    /// Padding byte used when stream normalization pads.
    pub observation_stream_pad_byte: u8,
    pub reward_policy: VmRewardPolicy,
    pub action_source: VmActionSource,
    pub action_filter: Option<VmActionFilter>,
    pub resource_limits: Option<VmResourceLimits>,
    pub hooks: VmHooks,
}

struct VmConsole {
    reader: BufReader<UnixStream>,
    writer: UnixStream,
}

impl VmConsole {
    fn connect(path: &str, timeout_ms: u64) -> anyhow::Result<Self> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut stream = None;

        while Instant::now() < deadline {
            match UnixStream::connect(path) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::ConnectionRefused || e.raw_os_error() == Some(11) => {
                    // EAGAIN (11) or Refused: QEMU might still be initializing the device.
                    std::thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return Err(e.into()),
            }
        }

        let stream = stream.ok_or_else(|| anyhow::anyhow!("timeout connecting to VM console socket: {}", path))?;
        stream.set_read_timeout(Some(Duration::from_millis(timeout_ms)))?;
        let writer = stream.try_clone()?;
        Ok(Self {
            reader: BufReader::new(stream),
            writer,
        })
    }

    fn send_line(&mut self, line: &str) -> anyhow::Result<()> {
        self.writer.write_all(line.as_bytes())?;
        self.writer.flush()?;
        Ok(())
    }

    fn read_line(&mut self, buf: &mut String) -> std::io::Result<usize> {
        self.reader.read_line(buf)
    }
}

fn is_timeout_err(err: &std::io::Error) -> bool {
    matches!(
        err.kind(),
        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
    )
}

fn shell_escape(arg: &str) -> String {
    if arg.is_empty() {
        return "''".to_string();
    }
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('\'');
    for ch in arg.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

fn render_ssh_command(cmd: &VmSshCommand) -> String {
    let mut parts = Vec::new();
    parts.push(shell_escape(&cmd.command));
    for arg in &cmd.args {
        parts.push(shell_escape(arg));
    }
    let mut full = parts.join(" ");

    if !cmd.env.is_empty() {
        let mut env_parts = Vec::with_capacity(cmd.env.len());
        for (k, v) in &cmd.env {
            env_parts.push(format!("{}={}", shell_escape(k), shell_escape(v)));
        }
        full = format!("{} {}", env_parts.join(" "), full);
    }

    if let Some(ref dir) = cmd.workdir {
        full = format!("cd {} && {}", shell_escape(dir), full);
    }

    if let Some(ref user) = cmd.run_as {
        full = format!("sudo -n -u {} -- {}", shell_escape(user), full);
    }

    full
}

struct SshResult {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    exit_status: Option<i32>,
}

struct SshClient {
    session: Session,
    step_timeout_ms: u64,
    max_output_bytes: usize,
}

impl SshClient {
    fn connect(
        config: VmSshConfig,
        connect_timeout_ms: u64,
        step_timeout_ms: u64,
        max_output_bytes: usize,
    ) -> anyhow::Result<Self> {
        let addrs = (config.host.as_str(), config.port).to_socket_addrs()?;
        let mut last_err = None;
        let mut tcp_opt = None;
        for addr in addrs {
            match TcpStream::connect_timeout(&addr, Duration::from_millis(connect_timeout_ms)) {
                Ok(tcp) => {
                    tcp_opt = Some(tcp);
                    break;
                }
                Err(e) => last_err = Some(e),
            }
        }
        let tcp = tcp_opt.ok_or_else(|| {
            let err = last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "no socket address resolved".to_string());
            anyhow::anyhow!("ssh connect failed: {}", err)
        })?;
        tcp.set_read_timeout(Some(Duration::from_millis(step_timeout_ms)))?;
        tcp.set_write_timeout(Some(Duration::from_millis(step_timeout_ms)))?;
        let mut session = Session::new()?;
        session.set_tcp_stream(tcp);
        session.set_timeout(step_timeout_ms as u32);
        session.set_blocking(true);
        session.handshake()?;

        if let Some(ref key) = config.private_key {
            let pubkey = config.public_key.as_deref().map(Path::new);
            session.userauth_pubkey_file(
                &config.user,
                pubkey,
                Path::new(key),
                config.passphrase.as_deref(),
            )?;
        } else if let Some(ref pw) = config.password {
            session.userauth_password(&config.user, pw)?;
        } else {
            return Err(anyhow::anyhow!("ssh password or private_key is required"));
        }

        if !session.authenticated() {
            return Err(anyhow::anyhow!("ssh authentication failed"));
        }

        Ok(Self {
            session,
            step_timeout_ms,
            max_output_bytes,
        })
    }

    fn exec(&mut self, cmd: &VmSshCommand, stdin: Option<&[u8]>) -> anyhow::Result<SshResult> {
        let mut channel = self.session.channel_session()?;
        let cmdline = render_ssh_command(cmd);
        channel.exec(&cmdline)?;
        if let Some(input) = stdin {
            let _ = channel.write_all(input);
        }
        let _ = channel.send_eof();

        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut stderr_stream = channel.stderr();
        let mut buf = [0u8; 4096];
        let start = Instant::now();

        loop {
            let mut progress = false;
            match channel.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    progress = true;
                    if stdout.len() < self.max_output_bytes {
                        let take = (self.max_output_bytes - stdout.len()).min(n);
                        stdout.extend_from_slice(&buf[..take]);
                    }
                }
                Err(e) if is_timeout_err(&e) => {}
                Err(e) => return Err(e.into()),
            }
            match stderr_stream.read(&mut buf) {
                Ok(0) => {}
                Ok(n) => {
                    progress = true;
                    if stderr.len() < self.max_output_bytes {
                        let take = (self.max_output_bytes - stderr.len()).min(n);
                        stderr.extend_from_slice(&buf[..take]);
                    }
                }
                Err(e) if is_timeout_err(&e) => {}
                Err(e) => return Err(e.into()),
            }

            if channel.eof() {
                break;
            }
            if start.elapsed() > Duration::from_millis(self.step_timeout_ms) {
                return Err(anyhow::anyhow!("ssh command timeout"));
            }
            if !progress {
                std::thread::sleep(Duration::from_millis(5));
            }
        }

        channel.wait_close()?;
        let exit_status = channel.exit_status().ok();
        Ok(SshResult {
            stdout,
            stderr,
            exit_status,
        })
    }

    fn upload_file(&self, local: &str, remote: &str, mode: Option<u32>) -> anyhow::Result<()> {
        let sftp = self.session.sftp()?;
        let mut remote_file = sftp.create(Path::new(remote))?;
        let mut file = File::open(local)?;
        std::io::copy(&mut file, &mut remote_file)?;
        if let Some(mode) = mode {
            let stat = ssh2::FileStat {
                size: None,
                uid: None,
                gid: None,
                perm: Some(mode),
                atime: None,
                mtime: None,
            };
            sftp.setstat(Path::new(remote), stat)?;
        }
        Ok(())
    }

    fn upload_bytes(&self, data: &[u8], remote: &str, mode: Option<u32>) -> anyhow::Result<()> {
        let sftp = self.session.sftp()?;
        let mut remote_file = sftp.create(Path::new(remote))?;
        remote_file.write_all(data)?;
        if let Some(mode) = mode {
            let stat = ssh2::FileStat {
                size: None,
                uid: None,
                gid: None,
                perm: Some(mode),
                atime: None,
                mtime: None,
            };
            sftp.setstat(Path::new(remote), stat)?;
        }
        Ok(())
    }
}

struct VmTraceStream {
    reader: BufReader<UnixStream>,
    framing: VmTraceFraming,
    encoding: PayloadEncoding,
    max_bytes: usize,
    line_prefix: Option<String>,
    header_buf: Vec<u8>,
    pending: Option<PendingFrame>,
}

struct PendingFrame {
    remaining: usize,
    keep_target: usize,
    keep_buf: Vec<u8>,
}

impl PendingFrame {
    fn new(len: usize, max_bytes: usize) -> Self {
        let keep_target = len.min(max_bytes);
        Self {
            remaining: len,
            keep_target,
            keep_buf: Vec::with_capacity(keep_target),
        }
    }
}

impl VmTraceStream {
    fn connect(cfg: &VmTraceConfig) -> anyhow::Result<Self> {
        let path = cfg
            .socket_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("trace socket_path missing"))?;
        let stream = UnixStream::connect(path)?;
        stream.set_read_timeout(Some(Duration::from_millis(cfg.timeout_ms)))?;
        Ok(Self {
            reader: BufReader::new(stream),
            framing: cfg.framing,
            encoding: cfg.encoding,
            max_bytes: cfg.max_bytes,
            line_prefix: cfg.line_prefix.clone(),
            header_buf: Vec::new(),
            pending: None,
        })
    }

    fn read_frame(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        match self.framing {
            VmTraceFraming::Line => self.read_line_frame(),
            VmTraceFraming::Len32Le => self.read_len32_frame(),
        }
    }

    fn read_line_frame(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => return Err(anyhow::anyhow!("trace stream closed")),
            Ok(_) => {}
            Err(e) if is_timeout_err(&e) => {
                if line.is_empty() {
                    return Ok(None);
                }
                return Err(anyhow::anyhow!("trace line timed out"));
            }
            Err(e) => return Err(e.into()),
        }
        let trimmed = line.trim_end_matches(&['\r', '\n'][..]);
        let payload = if let Some(prefix) = &self.line_prefix {
            trimmed.strip_prefix(prefix).unwrap_or(trimmed)
        } else {
            trimmed
        };
        let data = self.encoding.decode(payload)?;
        let mut data = data;
        if data.len() > self.max_bytes {
            data.truncate(self.max_bytes);
        }
        Ok(Some(data))
    }

    fn read_len32_frame(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        loop {
            if self.pending.is_none() {
                if !self.read_len_header()? {
                    return Ok(None);
                }
                let mut buf = [0u8; 4];
                buf.copy_from_slice(&self.header_buf[..4]);
                self.header_buf.clear();
                let len = u32::from_le_bytes(buf) as usize;
                if len == 0 {
                    return Ok(Some(Vec::new()));
                }
                self.pending = Some(PendingFrame::new(len, self.max_bytes));
            }

            if let Some(data) = self.read_pending_frame()? {
                return Ok(Some(data));
            }
            return Ok(None);
        }
    }

    fn read_len_header(&mut self) -> anyhow::Result<bool> {
        while self.header_buf.len() < 4 {
            let mut buf = [0u8; 4];
            let needed = 4 - self.header_buf.len();
            let n = match self.reader.read(&mut buf[..needed]) {
                Ok(0) => return Err(anyhow::anyhow!("trace stream closed")),
                Ok(n) => n,
                Err(e) if is_timeout_err(&e) => {
                    if self.header_buf.is_empty() {
                        return Ok(false);
                    }
                    return Err(anyhow::anyhow!("trace header timed out"));
                }
                Err(e) => return Err(e.into()),
            };
            self.header_buf.extend_from_slice(&buf[..n]);
        }
        Ok(true)
    }

    fn read_pending_frame(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("trace pending frame missing"))?;
        let mut scratch = [0u8; 8192];
        while pending.remaining > 0 {
            let chunk = pending.remaining.min(scratch.len());
            let n = match self.reader.read(&mut scratch[..chunk]) {
                Ok(0) => return Err(anyhow::anyhow!("trace stream closed")),
                Ok(n) => n,
                Err(e) if is_timeout_err(&e) => {
                    return Err(anyhow::anyhow!("trace frame timed out"));
                }
                Err(e) => return Err(e.into()),
            };

            let keep_remaining = pending.keep_target.saturating_sub(pending.keep_buf.len());
            let keep = keep_remaining.min(n);
            if keep > 0 {
                pending.keep_buf.extend_from_slice(&scratch[..keep]);
            }
            pending.remaining = pending.remaining.saturating_sub(n);
        }

        let data = std::mem::take(&mut pending.keep_buf);
        self.pending = None;
        Ok(Some(data))
    }
}

struct LibvirtDriver {
    conn: Connect,
    debug: bool,
}

impl LibvirtDriver {
    fn new(uri: Option<String>, debug: bool) -> anyhow::Result<Self> {
        let conn = Connect::open(uri.as_deref())?;
        Ok(Self { conn, debug })
    }

    fn domain(&self, name: &str) -> anyhow::Result<Domain> {
        Ok(Domain::lookup_by_name(&self.conn, name)?)
    }

    fn ensure_running(&self, domain: &str) -> anyhow::Result<()> {
        let dom = self.domain(domain)?;
        if dom.is_active()? {
            return Ok(());
        }
        if self.debug {
            eprintln!("[VM Env] libvirt start {}", domain);
        }
        dom.create()?;
        Ok(())
    }

    fn snapshot_exists(&self, domain: &str, snapshot: &str) -> anyhow::Result<bool> {
        let dom = self.domain(domain)?;
        let snaps = dom.list_all_snapshots(0)?;
        for snap in snaps {
            if snap.get_name()? == snapshot {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn snapshot_revert(&self, domain: &str, snapshot: &str) -> anyhow::Result<()> {
        let dom = self.domain(domain)?;
        if self.debug {
            eprintln!("[VM Env] libvirt snapshot-revert {} {}", domain, snapshot);
        }
        let snap = DomainSnapshot::lookup_by_name(&dom, snapshot, 0)?;
        let flags = sys::VIR_DOMAIN_SNAPSHOT_REVERT_RUNNING | sys::VIR_DOMAIN_SNAPSHOT_REVERT_FORCE;
        snap.revert(flags)?;
        Ok(())
    }

    fn snapshot_create(&self, domain: &str, snapshot: &str) -> anyhow::Result<()> {
        let dom = self.domain(domain)?;
        if self.debug {
            eprintln!("[VM Env] libvirt snapshot-create {} {}", domain, snapshot);
        }
        let xml = format!(
            "<domainsnapshot><name>{}</name><description>infotheory auto snapshot</description><memory snapshot='internal'/></domainsnapshot>",
            snapshot
        );
        let flags = sys::VIR_DOMAIN_SNAPSHOT_CREATE_ATOMIC;
        DomainSnapshot::create_xml(&dom, &xml, flags)?;
        Ok(())
    }

    fn set_vcpus(&self, domain: &str, vcpus: u32, mode: ResourceApplyMode) -> anyhow::Result<()> {
        let dom = self.domain(domain)?;
        let flags = match mode {
            ResourceApplyMode::Live => sys::VIR_DOMAIN_AFFECT_LIVE,
            ResourceApplyMode::Config => sys::VIR_DOMAIN_AFFECT_CONFIG,
            ResourceApplyMode::Both => sys::VIR_DOMAIN_AFFECT_LIVE | sys::VIR_DOMAIN_AFFECT_CONFIG,
        };
        dom.set_vcpus_flags(vcpus, flags)?;
        Ok(())
    }

    fn set_memory_mib(
        &self,
        domain: &str,
        memory_mib: u64,
        mode: ResourceApplyMode,
    ) -> anyhow::Result<()> {
        let dom = self.domain(domain)?;
        let flags = match mode {
            ResourceApplyMode::Live => sys::VIR_DOMAIN_AFFECT_LIVE,
            ResourceApplyMode::Config => sys::VIR_DOMAIN_AFFECT_CONFIG,
            ResourceApplyMode::Both => sys::VIR_DOMAIN_AFFECT_LIVE | sys::VIR_DOMAIN_AFFECT_CONFIG,
        };
        let memory_kib = memory_mib.saturating_mul(1024);
        dom.set_memory_flags(memory_kib, flags)?;
        Ok(())
    }
}

struct FuzzState {
    current: Vec<u8>,
    rng: RandomGenerator,
}

enum TraceModel {
    Rosa {
        model: RosaPlus,
        max_order: i64,
    },
    Ctw {
        tree: crate::ctw::ContextTree,
    },
    FacCtw {
        tree: crate::ctw::FacContextTree,
        bits_per_symbol: usize,
    },
    Rwkv7 {
        compressor: Compressor,
        primed: bool,
    },
}

impl TraceModel {
    fn new(backend: &RateBackend, max_order: i64) -> Self {
        match backend {
            RateBackend::RosaPlus => {
                let mut model = RosaPlus::new(max_order, false, 0, 42);
                model.build_lm_full_bytes_no_finalize_endpos();
                TraceModel::Rosa { model, max_order }
            }
            RateBackend::Rwkv7 { model } => {
                let compressor = Compressor::new_from_model(model.clone());
                TraceModel::Rwkv7 {
                    compressor,
                    primed: false,
                }
            }
            RateBackend::Ctw { depth } => TraceModel::Ctw {
                tree: crate::ctw::ContextTree::new(*depth),
            },
            RateBackend::FacCtw {
                base_depth,
                num_percept_bits: _,
                encoding_bits,
            } => {
                let bits_per_symbol = (*encoding_bits).min(8).max(1);
                TraceModel::FacCtw {
                    tree: crate::ctw::FacContextTree::new(*base_depth, bits_per_symbol),
                    bits_per_symbol,
                }
            }
        }
    }

    fn reset(&mut self) {
        match self {
            TraceModel::Rosa { model, max_order } => {
                let mut fresh = RosaPlus::new(*max_order, false, 0, 42);
                fresh.build_lm_full_bytes_no_finalize_endpos();
                *model = fresh;
            }
            TraceModel::Ctw { tree } => tree.clear(),
            TraceModel::FacCtw { tree, .. } => tree.clear(),
            TraceModel::Rwkv7 { compressor, primed } => {
                compressor.state.reset();
                *primed = false;
            }
        }
    }

    fn update_and_score(&mut self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        match self {
            TraceModel::Rosa { model, .. } => {
                let mut bits = 0.0;
                let mut tx = model.begin_tx();
                for &b in data {
                    let p = model.prob_for_last(b as u32).max(1e-12);
                    bits -= p.log2();
                    model.train_example_tx(&mut tx, &[b]);
                }
                bits
            }
            TraceModel::Ctw { tree } => {
                let log_before = tree.get_log_block_probability();
                for &b in data {
                    for i in (0..8).rev() {
                        tree.update(((b >> i) & 1) == 1);
                    }
                }
                let log_after = tree.get_log_block_probability();
                let log_delta = log_after - log_before;
                -log_delta / std::f64::consts::LN_2
            }
            TraceModel::FacCtw {
                tree,
                bits_per_symbol,
            } => {
                let log_before = tree.get_log_block_probability();
                for &b in data {
                    for i in 0..*bits_per_symbol {
                        tree.update(((b >> i) & 1) == 1, i);
                    }
                }
                let log_after = tree.get_log_block_probability();
                let log_delta = log_after - log_before;
                -log_delta / std::f64::consts::LN_2
            }
            TraceModel::Rwkv7 { compressor, primed } => {
                if !*primed {
                    let vocab_size = compressor.vocab_size();
                    let logits =
                        compressor
                            .model
                            .forward(&mut compressor.scratch, 0, &mut compressor.state);
                    softmax_pdf_inplace(
                        logits,
                        vocab_size,
                        &mut compressor.pdf_buffer,
                    );
                    *primed = true;
                }
                let mut bits = 0.0;
                let vocab_size = compressor.vocab_size();
                for &b in data {
                    let p = compressor.pdf_buffer[b as usize].max(1e-12);
                    bits -= p.log2();
                    let logits = compressor
                        .model
                        .forward(&mut compressor.scratch, b as u32, &mut compressor.state);
                    softmax_pdf_inplace(logits, vocab_size, &mut compressor.pdf_buffer);
                }
                bits
            }
        }
    }
}

pub struct VmEnvironment {
    config: VmEnvironmentConfig,
    driver: LibvirtDriver,
    console: Option<VmConsole>,
    ssh: Option<SshClient>,
    trace: Option<VmTraceStream>,
    trace_model: Option<TraceModel>,
    obs: PerceptVal,
    rew: Reward,
    obs_stream: Vec<PerceptVal>,
    step_in_episode: usize,
    needs_reset: bool,
    fuzz_state: Option<FuzzState>,
    baseline_entropy: Option<f64>,
    snapshot_ready: bool,
}

impl VmEnvironment {
    pub fn new(config: VmEnvironmentConfig) -> anyhow::Result<Self> {
        if config.domain.is_empty() {
            return Err(anyhow::anyhow!("vm_config.domain must be set"));
        }
        if config.snapshot.is_empty() {
            return Err(anyhow::anyhow!("vm_config.snapshot must be set"));
        }
        if config.episode_steps == 0 {
            return Err(anyhow::anyhow!("vm_config.episode_steps must be > 0"));
        }
        match config.transport {
            VmTransport::Serial => {
                if config.console.is_none() {
                    return Err(anyhow::anyhow!("vm_config.console must be set for serial transport"));
                }
            }
            VmTransport::Ssh => {
                if config.ssh.is_none() {
                    return Err(anyhow::anyhow!("vm_config.ssh must be set for ssh transport"));
                }
            }
        }
        if matches!(config.observation_policy, VmObservationPolicy::RawOutput)
            && config.observation_stream_len == 0
        {
            return Err(anyhow::anyhow!(
                "vm_observation.stream_len must be > 0 for raw output streams"
            ));
        }
        if matches!(config.observation_policy, VmObservationPolicy::RawOutput)
            && config.observation_bits < 8
        {
            return Err(anyhow::anyhow!(
                "observation_bits must be >= 8 for raw output streams"
            ));
        }
        if matches!(config.reward_policy, VmRewardPolicy::TraceEntropy { .. })
            && config.trace.is_none()
        {
            return Err(anyhow::anyhow!(
                "vm_trace must be configured for vm_reward.mode=trace-entropy"
            ));
        }
        if let Some(trace) = &config.trace {
            if trace.max_bytes == 0 {
                return Err(anyhow::anyhow!("vm_trace.max_bytes must be > 0"));
            }
            if trace.ssh_command.is_some() && config.transport != VmTransport::Ssh {
                return Err(anyhow::anyhow!(
                    "vm_trace.ssh_command requires ssh transport"
                ));
            }
            if trace.ssh_command.is_none() {
                if trace.socket_path.is_none() {
                    return Err(anyhow::anyhow!("vm_trace.socket_path must be set"));
                }
                if trace.timeout_ms == 0 {
                    return Err(anyhow::anyhow!("vm_trace.timeout_ms must be > 0"));
                }
            }
        }

        let driver = LibvirtDriver::new(config.libvirt_uri.clone(), config.debug_mode)?;
        let baseline_entropy = match &config.reward_policy {
            VmRewardPolicy::EntropyReduction {
                baseline_bytes,
                max_order,
                ..
            } => {
                let h = if *max_order == 0 {
                    marginal_entropy_bytes(baseline_bytes)
                } else {
                    entropy_rate_backend(
                        baseline_bytes,
                        *max_order,
                        &config.stats_backend,
                    )
                };
                Some(h)
            }
            _ => None,
        };

        let trace_model = match &config.reward_policy {
            VmRewardPolicy::TraceEntropy { max_order, .. } => {
                Some(TraceModel::new(&config.stats_backend, *max_order))
            }
            _ => None,
        };

        let fuzz_state = match &config.action_source {
            VmActionSource::Fuzz(fuzz) => {
                if fuzz.seeds.is_empty() {
                    return Err(anyhow::anyhow!("vm_actions.fuzz requires at least one seed"));
                }
                if fuzz.mutators.is_empty() {
                    return Err(anyhow::anyhow!(
                        "vm_actions.fuzz requires at least one mutator"
                    ));
                }
                let seed = fuzz.seeds[0].clone();
                Some(FuzzState {
                    current: seed,
                    rng: RandomGenerator::new().fork_with(fuzz.rng_seed),
                })
            }
            VmActionSource::Literal(actions) => {
                if actions.is_empty() {
                    return Err(anyhow::anyhow!(
                        "vm_actions.actions requires at least one action"
                    ));
                }
                None
            }
        };

        let mut env = Self {
            config,
            driver,
            console: None,
            ssh: None,
            trace: None,
            trace_model,
            obs: 0,
            rew: 0,
            obs_stream: Vec::new(),
            step_in_episode: 0,
            needs_reset: true,
            fuzz_state,
            baseline_entropy,
            snapshot_ready: false,
        };
        env.reset_episode()?;
        Ok(env)
    }

    fn reset_episode(&mut self) -> anyhow::Result<()> {
        if self.config.debug_mode {
            eprintln!(
                "[VM Env] Resetting episode: domain={}, snapshot={}",
                self.config.domain, self.config.snapshot
            );
        }
        if !self.snapshot_ready {
            self.ensure_snapshot_ready()?;
        }
        for hook in &self.config.hooks.pre_revert {
            let _ = Command::new(&hook.command).args(&hook.args).status();
        }
        if self.driver.snapshot_exists(&self.config.domain, &self.config.snapshot)? {
            self.driver
                .snapshot_revert(&self.config.domain, &self.config.snapshot)?;
        }
        self.driver.ensure_running(&self.config.domain)?;
        if let Some(limits) = &self.config.resource_limits {
            if let Some(vcpus) = limits.vcpus {
                let _ = self.driver.set_vcpus(&self.config.domain, vcpus, limits.apply_mode);
            }
            if let Some(memory_mib) = limits.memory_mib {
                let _ =
                    self.driver
                        .set_memory_mib(&self.config.domain, memory_mib, limits.apply_mode);
            }
        }
        for hook in &self.config.hooks.post_revert {
            let _ = Command::new(&hook.command).args(&hook.args).status();
        }

        self.console = None;
        self.trace = None;
        self.ssh = None;
        match self.config.transport {
            VmTransport::Serial => {
                let console_cfg = self
                    .config
                    .console
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("console missing"))?;
                let console =
                    VmConsole::connect(&console_cfg.socket_path, console_cfg.timeout_ms)?;
                self.console = Some(console);
                if self.trace_model.is_some() {
                    if let Some(trace_cfg) = &self.config.trace {
                        if trace_cfg.reset_on_episode {
                            if let Some(model) = &mut self.trace_model {
                                model.reset();
                            }
                        }
                        if trace_cfg.ssh_command.is_none() {
                            match VmTraceStream::connect(trace_cfg) {
                                Ok(trace_stream) => self.trace = Some(trace_stream),
                                Err(e) => {
                                    if self.config.debug_mode {
                                        eprintln!("[VM Env] Trace connect failed: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(ready) = self.config.boot_ready.clone() {
                    self.wait_for_ready(&ready)?;
                }
            }
            VmTransport::Ssh => {
                if self.trace_model.is_some() {
                    if let Some(trace_cfg) = &self.config.trace {
                        if trace_cfg.reset_on_episode {
                            if let Some(model) = &mut self.trace_model {
                                model.reset();
                            }
                        }
                    }
                }
                self.ensure_ssh_ready()?;
            }
        }

        self.step_in_episode = 0;
        self.needs_reset = false;
        Ok(())
    }

    fn ensure_snapshot_ready(&mut self) -> anyhow::Result<()> {
        if self.snapshot_ready {
            return Ok(());
        }
        if self.driver.snapshot_exists(&self.config.domain, &self.config.snapshot)? {
            self.snapshot_ready = true;
            return Ok(());
        }
        if !self.config.auto_snapshot {
            return Err(anyhow::anyhow!(
                "snapshot '{}' is missing and auto_snapshot=false",
                self.config.snapshot
            ));
        }
        self.driver.ensure_running(&self.config.domain)?;
        if self.config.transport == VmTransport::Ssh {
            let steps = self
                .config
                .ssh
                .as_ref()
                .map(|cfg| cfg.provision_steps.clone())
                .unwrap_or_default();
            if !steps.is_empty() {
                self.run_provision_steps(&steps)?;
            } else if self
                .config
                .ssh
                .as_ref()
                .and_then(|cfg| cfg.ready_command.as_ref())
                .is_some()
            {
                self.ensure_ssh_ready()?;
            }
        }
        self.driver
            .snapshot_create(&self.config.domain, &self.config.snapshot)?;
        if !self
            .driver
            .snapshot_exists(&self.config.domain, &self.config.snapshot)?
        {
            return Err(anyhow::anyhow!(
                "snapshot '{}' was not created successfully",
                self.config.snapshot
            ));
        }
        self.snapshot_ready = true;
        Ok(())
    }

    fn ensure_ssh_ready(&mut self) -> anyhow::Result<()> {
        if self.ssh.is_some() {
            return Ok(());
        }
        let cfg = self
            .config
            .ssh
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ssh config missing"))?
            .clone();
        let timeout = Duration::from_millis(self.config.boot_timeout_ms);
        let start = Instant::now();
        loop {
            match SshClient::connect(
                cfg.clone(),
                cfg.connect_timeout_ms,
                self.config.step_timeout_ms,
                self.config.max_output_bytes,
            ) {
                Ok(client) => {
                    self.ssh = Some(client);
                    if let Some(ref ready_cmd) = cfg.ready_command {
                        let res = self.ssh.as_mut().unwrap().exec(ready_cmd, None)?;
                        if res.exit_status.unwrap_or(1) == 0 {
                            return Ok(());
                        }
                        self.ssh = None;
                    } else {
                        return Ok(());
                    }
                }
                Err(e) => {
                    if start.elapsed() > timeout {
                        return Err(anyhow::anyhow!("timeout waiting for ssh ready: {}", e));
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(cfg.retry_interval_ms));
        }
    }

    fn run_provision_steps(&mut self, steps: &[VmSshProvisionStep]) -> anyhow::Result<()> {
        if steps.is_empty() {
            return Ok(());
        }
        self.ensure_ssh_ready()?;
        let ssh = self
            .ssh
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("ssh session missing"))?;
        for step in steps {
            match step {
                VmSshProvisionStep::Upload { local, remote, mode } => {
                    ssh.upload_file(local, remote, *mode)?;
                }
                VmSshProvisionStep::UploadText { remote, text, mode } => {
                    ssh.upload_bytes(text.as_bytes(), remote, *mode)?;
                }
                VmSshProvisionStep::Run { command } => {
                    let res = ssh.exec(command, None)?;
                    if res.exit_status.unwrap_or(1) != 0 {
                        return Err(anyhow::anyhow!(
                            "provision step failed: {}",
                            render_ssh_command(command)
                        ));
                    }
                }
                VmSshProvisionStep::RunScript { local, remote, mode } => {
                    ssh.upload_file(local, remote, *mode)?;
                    let mut cmd = VmSshCommand {
                        command: remote.to_string(),
                        args: Vec::new(),
                        stdin_payload: false,
                        env: Vec::new(),
                        workdir: None,
                        run_as: None,
                    };
                    if remote.ends_with(".sh") {
                        cmd.command = "/bin/sh".to_string();
                        cmd.args.push(remote.to_string());
                    }
                    let res = ssh.exec(&cmd, None)?;
                    if res.exit_status.unwrap_or(1) != 0 {
                        return Err(anyhow::anyhow!(
                            "provision script failed: {}",
                            render_ssh_command(&cmd)
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn wait_for_ready(&mut self, ready: &str) -> anyhow::Result<()> {
        let start = Instant::now();
        let timeout = Duration::from_millis(self.config.boot_timeout_ms);
        let console = self
            .console
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("console missing"))?;
        let mut line = String::new();
        loop {
            if start.elapsed() > timeout {
                return Err(anyhow::anyhow!("timeout waiting for boot ready marker"));
            }
            line.clear();
            match console.read_line(&mut line) {
                Ok(0) => {}
                Ok(_) => {
                    if line.contains(ready) {
                        return Ok(());
                    }
                }
                Err(e) if is_timeout_err(&e) => {}
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn get_action_payload(&mut self, action: Action) -> anyhow::Result<Vec<u8>> {
        match &self.config.action_source {
            VmActionSource::Literal(actions) => {
                let idx = action as usize;
                if idx >= actions.len() {
                    return Err(anyhow::anyhow!("action index out of range"));
                }
                Ok(actions[idx].payload.clone())
            }
            VmActionSource::Fuzz(fuzz) => {
                let state = self
                    .fuzz_state
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("fuzz state missing"))?;
                let idx = action as usize % fuzz.mutators.len();
                let mut input = state.current.clone();
                let mutator = &fuzz.mutators[idx];
                apply_mutator(mutator, &mut input, fuzz, &mut state.rng);
                if input.len() < fuzz.min_len {
                    input.resize(fuzz.min_len, 0);
                }
                if input.len() > fuzz.max_len {
                    input.truncate(fuzz.max_len);
                }
                state.current = input.clone();
                Ok(input)
            }
        }
    }

    fn filter_action(&self, payload: &[u8]) -> Option<i64> {
        let filter = self.config.action_filter.as_ref()?;
        if payload.is_empty() {
            return filter.reject_reward;
        }
        let (entropy, intrinsic, novelty) = self.compute_filter_metrics(payload, filter);

        if let Some(min_entropy) = filter.min_entropy {
            if entropy < min_entropy {
                return filter.reject_reward;
            }
        }
        if let Some(max_entropy) = filter.max_entropy {
            if entropy > max_entropy {
                return filter.reject_reward;
            }
        }
        if let Some(min_intrinsic) = filter.min_intrinsic_dependence {
            if intrinsic < min_intrinsic {
                return filter.reject_reward;
            }
        }
        if let Some(min_novelty) = filter.min_novelty {
            if filter.novelty_prior.is_some() && novelty < min_novelty {
                return filter.reject_reward;
            }
        }
        None
    }

    fn compute_filter_metrics(&self, payload: &[u8], filter: &VmActionFilter) -> (f64, f64, f64) {
        let entropy = if filter.max_order == 0 {
            marginal_entropy_bytes(payload)
        } else {
            entropy_rate_backend(payload, filter.max_order, &self.config.stats_backend)
        };
        let h_marg = marginal_entropy_bytes(payload);
        let h_rate = if filter.max_order == 0 {
            h_marg
        } else {
            entropy_rate_backend(payload, filter.max_order, &self.config.stats_backend)
        };
        let intrinsic = if h_marg < 1e-9 {
            0.0
        } else {
            ((h_marg - h_rate) / h_marg).clamp(0.0, 1.0)
        };
        let novelty = if let Some(ref prior) = filter.novelty_prior {
            cross_entropy_rate_backend(payload, prior, filter.max_order, &self.config.stats_backend)
        } else {
            0.0
        };
        (entropy, intrinsic, novelty)
    }

    fn format_action(&self, action: Action, payload: &[u8]) -> String {
        let encoded = self.config.protocol.wire_encoding.encode(payload);
        if encoded.is_empty() {
            format!("{}{}{}", self.config.protocol.action_prefix, action, self.config.protocol.action_suffix)
        } else {
            format!(
                "{}{} {}{}",
                self.config.protocol.action_prefix,
                action,
                encoded,
                self.config.protocol.action_suffix
            )
        }
    }

    fn read_response(&mut self) -> anyhow::Result<VmResponse> {
        let mut response = VmResponse::default();
        let start = Instant::now();
        let timeout = Duration::from_millis(self.config.step_timeout_ms);
        let console = self
            .console
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("console missing"))?;
        let protocol = self.config.protocol.clone();
        let mut line = String::new();
        let mut lines = 0usize;

        while start.elapsed() <= timeout && lines < self.config.max_response_lines {
            line.clear();
            match console.read_line(&mut line) {
                Ok(0) => {}
                Ok(_) => {
                    lines += 1;
                    if parse_response_line(&protocol, &line, &mut response)? {
                        break;
                    }
                }
                Err(e) if is_timeout_err(&e) => break,
                Err(e) => return Err(e.into()),
            }
        }

        Ok(response)
    }

    fn select_ssh_command(&self, action: Action) -> anyhow::Result<VmSshCommand> {
        let cfg = self
            .config
            .ssh
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ssh config missing"))?;
        if let Some(list) = &cfg.action_commands {
            let idx = action as usize;
            if idx >= list.len() {
                return Err(anyhow::anyhow!("ssh action index out of range"));
            }
            return Ok(list[idx].clone());
        }
        Ok(cfg.action_command.clone())
    }

    fn run_ssh_action(&mut self, action: Action, payload: &[u8]) -> anyhow::Result<VmResponse> {
        self.ensure_ssh_ready()?;
        let cmd = self.select_ssh_command(action)?;
        let stdin = if cmd.stdin_payload { Some(payload) } else { None };
        let result = {
            let ssh = self
                .ssh
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("ssh session missing"))?;
            ssh.exec(&cmd, stdin)?
        };
        Ok(self.response_from_ssh_result(result))
    }

    fn response_from_ssh_result(&self, result: SshResult) -> VmResponse {
        let mut response = VmResponse::default();
        let protocol = self.config.protocol.clone();
        for line in result.stdout.split(|b| *b == b'\n') {
            let line_str = String::from_utf8_lossy(line);
            if let Ok(done) = parse_response_line(&protocol, &line_str, &mut response) {
                if done {
                    break;
                }
            }
        }
        response.output.extend_from_slice(&result.stdout);
        response.output.extend_from_slice(&result.stderr);
        response.exit_status = result.exit_status;
        response
    }

    fn hash_observation(&self, response: &VmResponse) -> PerceptVal {
        let data = response
            .data
            .as_ref()
            .map(|d| d.as_slice())
            .unwrap_or_else(|| response.output.as_slice());
        let h = robust_hash_bytes(data);
        self.mask_observation(h)
    }

    fn mask_observation(&self, value: u64) -> u64 {
        let bits = self.config.observation_bits;
        if bits >= 64 {
            value
        } else if bits == 0 {
            0
        } else {
            value & ((1u64 << bits) - 1)
        }
    }

    fn compute_reward(&mut self, response: &VmResponse) -> Reward {
        let trace_data = if matches!(self.config.reward_policy, VmRewardPolicy::TraceEntropy { .. }) {
            self.read_trace_bytes()
        } else {
            Vec::new()
        };
        let mut reward = match &self.config.reward_policy {
            VmRewardPolicy::FromGuest => response.rew.unwrap_or(0),
            VmRewardPolicy::Pattern {
                pattern,
                base_reward,
                bonus_reward,
            } => {
                let text = String::from_utf8_lossy(&response.output);
                if text.contains(pattern) {
                    base_reward + bonus_reward
                } else {
                    *base_reward
                }
            }
            VmRewardPolicy::EntropyReduction {
                max_order,
                scale,
                ..
            } => {
                let data = response
                    .data
                    .as_ref()
                    .map(|d| d.as_slice())
                    .unwrap_or_else(|| response.output.as_slice());
                let h_obs = if *max_order == 0 {
                    marginal_entropy_bytes(data)
                } else {
                    entropy_rate_backend(data, *max_order, &self.config.stats_backend)
                };
                let h_base = self.baseline_entropy.unwrap_or(0.0);
                let er = (h_base - h_obs) * scale;
                er.round() as i64
            }
            VmRewardPolicy::TraceEntropy {
                scale,
                normalize,
                ..
            } => {
                let data = &trace_data;
                let bits = match self.trace_model.as_mut() {
                    Some(model) => model.update_and_score(data),
                    None => 0.0,
                };
                let bits = if *normalize && !data.is_empty() {
                    bits / data.len() as f64
                } else {
                    bits
                };
                (bits * scale).round() as i64
            }
        };

        reward = reward.saturating_sub(self.config.step_cost);
        let min_reward = self.min_reward();
        let max_reward = self.max_reward();
        reward.clamp(min_reward, max_reward)
    }

    fn read_trace_bytes(&mut self) -> Vec<u8> {
        let trace_cfg = match &self.config.trace {
            Some(cfg) => cfg.clone(),
            None => return Vec::new(),
        };
        let max_bytes = trace_cfg.max_bytes;
        let ssh_cmd = trace_cfg.ssh_command.clone();
        if let Some(cmd) = ssh_cmd.as_ref() {
            if let Err(e) = self.ensure_ssh_ready() {
                if self.config.debug_mode {
                    eprintln!("[VM Env] Trace SSH connect failed: {}", e);
                }
                return Vec::new();
            }
            let ssh = match self.ssh.as_mut() {
                Some(ssh) => ssh,
                None => return Vec::new(),
            };
            match ssh.exec(cmd, None) {
                Ok(result) => {
                    let mut out = result.stdout;
                    if out.len() > max_bytes {
                        out.truncate(max_bytes);
                    }
                    out
                }
                Err(e) => {
                    if self.config.debug_mode {
                        eprintln!("[VM Env] Trace SSH read failed: {}", e);
                    }
                    Vec::new()
                }
            }
        } else {
            if self.trace.is_none() {
                match VmTraceStream::connect(&trace_cfg) {
                    Ok(stream) => self.trace = Some(stream),
                    Err(e) => {
                        if self.config.debug_mode {
                            eprintln!("[VM Env] Trace reconnect failed: {}", e);
                        }
                        return Vec::new();
                    }
                }
            }

            let Some(stream) = self.trace.as_mut() else {
                return Vec::new();
            };
            match stream.read_frame() {
                Ok(Some(data)) => data,
                Ok(None) => Vec::new(),
                Err(e) => {
                    if self.config.debug_mode {
                        eprintln!("[VM Env] Trace read failed: {}", e);
                    }
                    self.trace = None;
                    Vec::new()
                }
            }
        }
    }

    fn action_count(&self) -> usize {
        match &self.config.action_source {
            VmActionSource::Literal(actions) => actions.len(),
            VmActionSource::Fuzz(fuzz) => fuzz.mutators.len(),
        }
    }

    fn build_observation_stream(&self, response: &VmResponse) -> Vec<PerceptVal> {
        let mut observations = match self.config.observation_policy {
            VmObservationPolicy::FromGuest => {
                if let Some(obs) = response.obs {
                    vec![self.mask_observation(obs)]
                } else {
                    vec![self.hash_observation(response)]
                }
            }
            VmObservationPolicy::OutputHash => vec![self.hash_observation(response)],
            VmObservationPolicy::RawOutput => {
                let data = response
                    .data
                    .as_ref()
                    .map(|d| d.as_slice())
                    .unwrap_or_else(|| response.output.as_slice());
                data.iter().map(|b| *b as PerceptVal).collect()
            }
        };

        if observations.is_empty() {
            observations.push(0);
        }

        self.normalize_observation_stream(&mut observations);
        observations
    }

    fn normalize_observation_stream(&self, observations: &mut Vec<PerceptVal>) {
        let mask = if self.config.observation_bits >= 64 {
            u64::MAX
        } else if self.config.observation_bits == 0 {
            0
        } else {
            (1u64 << self.config.observation_bits) - 1
        };
        for obs in observations.iter_mut() {
            *obs &= mask;
        }
        let target = self.config.observation_stream_len;
        if target == 0 {
            return;
        }
        if observations.len() > target {
            match self.config.observation_stream_mode {
                VmObservationStreamMode::Truncate => {
                    observations.truncate(target);
                }
                VmObservationStreamMode::PadTruncate => {
                    observations.truncate(target);
                }
                VmObservationStreamMode::Pad => {}
            }
        } else if observations.len() < target {
            match self.config.observation_stream_mode {
                VmObservationStreamMode::Pad | VmObservationStreamMode::PadTruncate => {
                    let pad = self.config.observation_stream_pad_byte as PerceptVal;
                    observations.resize(target, pad);
                }
                VmObservationStreamMode::Truncate => {}
            }
        }
    }
}

fn parse_response_line(
    protocol: &VmProtocolConfig,
    line: &str,
    response: &mut VmResponse,
) -> anyhow::Result<bool> {
    if let Some(rest) = line.strip_prefix(&protocol.obs_prefix) {
        if let Ok(v) = rest.trim().parse::<u64>() {
            response.obs = Some(v);
        }
        return Ok(response.ready());
    }
    if let Some(rest) = line.strip_prefix(&protocol.rew_prefix) {
        if let Ok(v) = rest.trim().parse::<i64>() {
            response.rew = Some(v);
        }
        return Ok(response.ready());
    }
    if let Some(rest) = line.strip_prefix(&protocol.done_prefix) {
        response.done = rest.trim() != "0";
        return Ok(response.ready());
    }
    if let Some(rest) = line.strip_prefix(&protocol.data_prefix) {
        let data = protocol.wire_encoding.decode(rest.trim())?;
        response.data = Some(data);
        return Ok(response.ready());
    }
    response.output.extend_from_slice(line.as_bytes());
    Ok(response.ready())
}

impl Environment for VmEnvironment {
    fn perform_action(&mut self, action: Action) {
        if self.step_in_episode == 0 && self.needs_reset {
            if let Err(e) = self.reset_episode() {
                if self.config.debug_mode {
                    eprintln!("[VM Env] Reset failed: {}", e);
                }
            }
        }

        let payload = match self.get_action_payload(action) {
            Ok(payload) => payload,
            Err(e) => {
                if self.config.debug_mode {
                    eprintln!("[VM Env] Invalid action: {}", e);
                }
                self.obs = 0;
                self.rew = self.min_reward();
                self.obs_stream.clear();
                self.obs_stream.push(0);
                self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
                if self.step_in_episode == 0 {
                    self.needs_reset = true;
                }
                return;
            }
        };

        if let Some(reject_reward) = self.filter_action(&payload) {
            self.obs = 0;
            self.rew = reject_reward.clamp(self.min_reward(), self.max_reward());
            self.obs_stream.clear();
            self.obs_stream.push(0);
            self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
            if self.step_in_episode == 0 {
                self.needs_reset = true;
            }
            return;
        }

        let response = match self.config.transport {
            VmTransport::Serial => {
                let msg = self.format_action(action, &payload);
                if let Some(console) = self.console.as_mut() {
                    if let Err(e) = console.send_line(&msg) {
                        if self.config.debug_mode {
                            eprintln!("[VM Env] Failed to send action: {}", e);
                        }
                        VmResponse::default()
                    } else {
                        match self.read_response() {
                            Ok(resp) => resp,
                            Err(e) => {
                                if self.config.debug_mode {
                                    eprintln!("[VM Env] Failed to read response: {}", e);
                                }
                                VmResponse::default()
                            }
                        }
                    }
                } else {
                    if self.config.debug_mode {
                        eprintln!("[VM Env] Console not connected");
                    }
                    VmResponse::default()
                }
            }
            VmTransport::Ssh => match self.run_ssh_action(action, &payload) {
                Ok(resp) => resp,
                Err(e) => {
                    if self.config.debug_mode {
                        eprintln!("[VM Env] SSH action failed: {}", e);
                    }
                    self.ssh = None;
                    VmResponse::default()
                }
            },
        };

        self.obs_stream = self.build_observation_stream(&response);
        self.obs = self.obs_stream.first().copied().unwrap_or(0);
        self.rew = self.compute_reward(&response);

        if self.config.debug_mode {
            eprintln!(
                "[VM Env] Action={} Obs={} Rew={} Done={}",
                action, self.obs, self.rew, response.done
            );
        }

        self.step_in_episode = (self.step_in_episode + 1) % self.config.episode_steps;
        if self.step_in_episode == 0 {
            self.needs_reset = true;
        }
    }

    fn get_observation(&self) -> PerceptVal {
        self.obs
    }

    fn drain_observations(&mut self) -> Vec<PerceptVal> {
        if self.obs_stream.is_empty() {
            vec![self.obs]
        } else {
            std::mem::take(&mut self.obs_stream)
        }
    }

    fn get_reward(&self) -> Reward {
        self.rew
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn get_observation_bits(&self) -> usize {
        self.config.observation_bits
    }

    fn get_reward_bits(&self) -> usize {
        self.config.reward_bits
    }

    fn get_action_bits(&self) -> usize {
        let n = self.action_count();
        if n <= 1 {
            return 1;
        }
        (n as f64).log2().ceil() as usize
    }

    fn get_num_actions(&self) -> usize {
        self.action_count()
    }
}

#[derive(Default)]
struct VmResponse {
    obs: Option<PerceptVal>,
    rew: Option<Reward>,
    done: bool,
    output: Vec<u8>,
    data: Option<Vec<u8>>,
    exit_status: Option<i32>,
}

impl VmResponse {
    fn ready(&self) -> bool {
        self.done || (self.rew.is_some() && self.obs.is_some())
    }
}

fn robust_hash_bytes(data: &[u8]) -> u64 {
    let mut h = 0u64;
    for &b in data {
        h = h.rotate_left(7) ^ (b as u64);
    }
    h
}

fn apply_mutator(
    mutator: &FuzzMutator,
    input: &mut Vec<u8>,
    fuzz: &VmFuzzConfig,
    rng: &mut RandomGenerator,
) {
    match mutator {
        FuzzMutator::FlipBit => {
            if input.is_empty() {
                input.push(0);
            }
            let idx = rng.gen_range(input.len());
            let bit = rng.gen_range(8);
            input[idx] ^= 1u8 << bit;
        }
        FuzzMutator::FlipByte => {
            if input.is_empty() {
                input.push(0);
            }
            let idx = rng.gen_range(input.len());
            input[idx] ^= rng.next_u64() as u8;
        }
        FuzzMutator::InsertByte => {
            let idx = if input.is_empty() {
                0
            } else {
                rng.gen_range(input.len() + 1)
            };
            let byte = if !fuzz.dictionary.is_empty() {
                let d = rng.gen_range(fuzz.dictionary.len());
                let entry = &fuzz.dictionary[d];
                if entry.is_empty() {
                    0
                } else {
                    entry[rng.gen_range(entry.len())]
                }
            } else {
                rng.next_u64() as u8
            };
            input.insert(idx, byte);
        }
        FuzzMutator::DeleteByte => {
            if input.len() > 1 {
                let idx = rng.gen_range(input.len());
                input.remove(idx);
            }
        }
        FuzzMutator::SpliceSeed => {
            if fuzz.seeds.is_empty() {
                return;
            }
            let seed = &fuzz.seeds[rng.gen_range(fuzz.seeds.len())];
            if input.is_empty() {
                input.extend_from_slice(seed);
            } else if !seed.is_empty() {
                let cut = rng.gen_range(input.len());
                let seed_cut = rng.gen_range(seed.len());
                let mut out = Vec::new();
                out.extend_from_slice(&input[..cut]);
                out.extend_from_slice(&seed[seed_cut..]);
                *input = out;
            }
        }
        FuzzMutator::ResetSeed => {
            if fuzz.seeds.is_empty() {
                return;
            }
            *input = fuzz.seeds[rng.gen_range(fuzz.seeds.len())].clone();
        }
        FuzzMutator::Havoc => {
            let flips = 1 + rng.gen_range(8);
            for _ in 0..flips {
                if input.is_empty() {
                    input.push(0);
                }
                let idx = rng.gen_range(input.len());
                input[idx] ^= rng.next_u64() as u8;
            }
        }
    }
}
