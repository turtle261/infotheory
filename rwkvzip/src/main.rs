//! rwkvzip CLI - Neural network compression using RWKV7.
//!
//! A high-performance lossless compressor that leverages the RWKV7 language model's
//! predictive capabilities combined with entropy coding for optimal compression.
//!
//! # Usage
//!
//! ```bash
//! # Compress a file (default: arithmetic coding)
//! rwkvzip compress input.txt output.canz --model model.safetensors
//!
//! # Compress with rANS (faster, slightly larger)
//! rwkvzip compress input.txt output.canz --model model.safetensors -c rans
//!
//! # Decompress
//! rwkvzip decompress output.canz restored.txt --model model.safetensors
//!
//! # Self-test (compress + decompress + verify roundtrip)
//! rwkvzip self-test input.txt --model model.safetensors
//!
//! # Calculate cross-entropy without compression
//! rwkvzip entropy input.txt --model model.safetensors
//! ```
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rwkvzip::{compress_with_stats, crc32, CoderType, Compressor};

// =============================================================================
// Command-Line Interface
// =============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP_TEXT: &str = r#"rwkvzip - Neural network compressor using RWKV7

USAGE:
    rwkvzip <COMMAND> [OPTIONS]

COMMANDS:
    compress      Compress a file using RWKV7 predictions + entropy coding
    decompress    Decompress a previously compressed file
    self-test     Compress, decompress, and verify roundtrip integrity
    entropy       Calculate cross-entropy (bits/byte) without compression
    train         Train a byte-level RWKV7 model on enwik8
    help          Show this help message

OPTIONS:
    -h, --help       Show help for a command
    -V, --version    Show version information

Run 'rwkvzip <COMMAND> --help' for more information on a command."#;

const COMPRESS_HELP: &str = r#"Compress a file using RWKV7 predictions + entropy coding

USAGE:
    rwkvzip compress <INPUT> <OUTPUT> --model <PATH> [OPTIONS]

ARGUMENTS:
    <INPUT>     Input file to compress
    <OUTPUT>    Output path for compressed file

OPTIONS:
    -m, --model <PATH>    Path to RWKV7 model weights (.safetensors)
    -c, --coder <TYPE>    Entropy coder: 'ac' (arithmetic, default) or 'rans'
    -h, --help            Show this help message"#;

const DECOMPRESS_HELP: &str = r#"Decompress a previously compressed file

USAGE:
    rwkvzip decompress <INPUT> <OUTPUT> --model <PATH>

ARGUMENTS:
    <INPUT>     Compressed input file
    <OUTPUT>    Output path for decompressed file

OPTIONS:
    -m, --model <PATH>    Path to RWKV7 model weights (.safetensors)
    -h, --help            Show this help message"#;

const SELF_TEST_HELP: &str = r#"Self-test: compress, decompress, and verify roundtrip integrity

USAGE:
    rwkvzip self-test <INPUT> --model <PATH> [OPTIONS]

ARGUMENTS:
    <INPUT>     Input file to test

OPTIONS:
    -m, --model <PATH>     Path to RWKV7 model weights (.safetensors)
    -c, --coder <TYPE>     Entropy coder: 'ac' (arithmetic, default) or 'rans'
    --output <PATH>        Write decompressed output to file (for inspection)
    -h, --help             Show this help message"#;

const ENTROPY_HELP: &str = r#"Calculate cross-entropy (bits/byte) without compression

USAGE:
    rwkvzip entropy <INPUT> --model <PATH>

ARGUMENTS:
    <INPUT>     Input file to analyze

OPTIONS:
    -m, --model <PATH>    Path to RWKV7 model weights (.safetensors)
    -h, --help            Show this help message"#;

const TRAIN_HELP: &str = r#"Train a byte-level RWKV7 model on enwik8

USAGE:
    rwkvzip train [OPTIONS]

OPTIONS:
    --dataset <PATH>      Dataset path (default: files/enwik8)
    --out <PATH>          Output model path (.safetensors) (default: out/rwkv7_byte_small.safetensors)
    --resume <PATH>       Resume training from existing model (.safetensors)
    --steps <N>           Training steps (default: 2000)
    --bsz <N>             Batch size (default: 32)
    --seq <N>             Sequence length (default: 128)
    --accum <N>           Gradient accumulation steps (default: 1, effectively bsz*accum)
    --lr <F>              Learning rate (default: 2e-4)
    --layers <N>          Number of transformer blocks (default: 6)
    --hidden <N>          Hidden size / embedding dim (default: 256)
    --intermediate <N>    FFN hidden size (default: 1024)
    --dw <N>              Decay low-rank dim (w_lora) (default: 32)
    --da <N>              A low-rank dim (default: 32)
    --dv <N>              V low-rank dim (default: 32)
    --dg <N>              G low-rank dim (default: 64)
    --ln-eps <F>          LayerNorm epsilon (default: 1e-5)
    --gn-eps <F>          GroupNorm epsilon (default: 64e-5)
    --seed <N>            RNG seed (default: 42)
    --validate-file <PATH> Validate compression roundtrip on this file (default: files/bench.txt)
    --device <cpu|cuda>   Force device (default: auto)
    --no-validate         Skip compression roundtrip validation
    -h, --help            Show this help message"#;

/// Parsed command-line arguments.
enum Command {
    Compress {
        input: String,
        output: String,
        model: String,
        coder: CoderType,
    },
    Decompress {
        input: String,
        output: String,
        model: String,
    },
    SelfTest {
        input: String,
        model: String,
        coder: CoderType,
        output: Option<String>,
    },
    Entropy {
        input: String,
        model: String,
    },
    Train {
        dataset: Option<String>,
        out: Option<String>,
        resume: Option<String>,
        steps: Option<usize>,
        bsz: Option<i64>,
        seq: Option<i64>,
        accum: Option<usize>,
        lr: Option<f64>,
        layers: Option<usize>,
        hidden: Option<i64>,
        intermediate: Option<i64>,
        dw: Option<i64>,
        da: Option<i64>,
        dv: Option<i64>,
        dg: Option<i64>,
        ln_eps: Option<f64>,
        gn_eps: Option<f64>,
        seed: Option<u64>,
        validate_file: Option<String>,
        device: Option<String>,
        no_validate: bool,
    },
    Help(Option<String>),
    Version,
}

/// Parse entropy coder type from string.
fn parse_coder(s: &str) -> Result<CoderType> {
    match s.to_lowercase().as_str() {
        "ac" | "arithmetic" => Ok(CoderType::AC),
        "rans" | "ans" => Ok(CoderType::RANS),
        _ => bail!("Unknown coder type: '{}'. Use 'ac' or 'rans'.", s),
    }
}

/// Parse command-line arguments into structured command.
fn parse_args() -> Result<Command> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        return Ok(Command::Help(None));
    }

    let cmd = args[1].as_str();
    let rest = &args[2..];

    match cmd {
        "-h" | "--help" | "help" => {
            let subcmd = rest.first().map(|s| s.as_str().to_string());
            Ok(Command::Help(subcmd))
        }
        "-V" | "--version" => Ok(Command::Version),

        "compress" => parse_compress(rest),
        "decompress" => parse_decompress(rest),
        "self-test" => parse_self_test(rest),
        "entropy" => parse_entropy(rest),
        "train" => parse_train(rest),

        other => bail!(
            "Unknown command: '{}'. Run 'rwkvzip help' for usage.",
            other
        ),
    }
}

/// Parse 'train' subcommand arguments.
fn parse_train(args: &[String]) -> Result<Command> {
    let mut dataset: Option<String> = None;
    let mut out: Option<String> = None;
    let mut resume: Option<String> = None;
    let mut steps: Option<usize> = None;
    let mut bsz: Option<i64> = None;
    let mut seq: Option<i64> = None;
    let mut accum: Option<usize> = None;
    let mut lr: Option<f64> = None;
    let mut layers: Option<usize> = None;
    let mut hidden: Option<i64> = None;
    let mut intermediate: Option<i64> = None;
    let mut dw: Option<i64> = None;
    let mut da: Option<i64> = None;
    let mut dv: Option<i64> = None;
    let mut dg: Option<i64> = None;
    let mut ln_eps: Option<f64> = None;
    let mut gn_eps: Option<f64> = None;
    let mut seed: Option<u64> = None;
    let mut validate_file: Option<String> = None;
    let mut device: Option<String> = None;
    let mut no_validate = false;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(Some("train".to_string()))),
            "--dataset" => {
                i += 1;
                dataset = Some(args.get(i).context("Missing value for --dataset")?.clone());
            }
            "--out" => {
                i += 1;
                out = Some(args.get(i).context("Missing value for --out")?.clone());
            }
            "--resume" => {
                i += 1;
                resume = Some(args.get(i).context("Missing value for --resume")?.clone());
            }
            "--steps" => {
                i += 1;
                let v = args.get(i).context("Missing value for --steps")?;
                steps = Some(v.parse::<usize>().context("Invalid --steps")?);
            }
            "--bsz" => {
                i += 1;
                let v = args.get(i).context("Missing value for --bsz")?;
                bsz = Some(v.parse::<i64>().context("Invalid --bsz")?);
            }
            "--seq" => {
                i += 1;
                let v = args.get(i).context("Missing value for --seq")?;
                seq = Some(v.parse::<i64>().context("Invalid --seq")?);
            }
            "--lr" => {
                i += 1;
                let v = args.get(i).context("Missing value for --lr")?;
                lr = Some(v.parse::<f64>().context("Invalid --lr")?);
            }
            "--layers" => {
                i += 1;
                let v = args.get(i).context("Missing value for --layers")?;
                layers = Some(v.parse::<usize>().context("Invalid --layers")?);
            }
            "--hidden" => {
                i += 1;
                let v = args.get(i).context("Missing value for --hidden")?;
                hidden = Some(v.parse::<i64>().context("Invalid --hidden")?);
            }
            "--intermediate" => {
                i += 1;
                let v = args.get(i).context("Missing value for --intermediate")?;
                intermediate = Some(v.parse::<i64>().context("Invalid --intermediate")?);
            }
            "--dw" => {
                i += 1;
                let v = args.get(i).context("Missing value for --dw")?;
                dw = Some(v.parse::<i64>().context("Invalid --dw")?);
            }
            "--da" => {
                i += 1;
                let v = args.get(i).context("Missing value for --da")?;
                da = Some(v.parse::<i64>().context("Invalid --da")?);
            }
            "--dv" => {
                i += 1;
                let v = args.get(i).context("Missing value for --dv")?;
                dv = Some(v.parse::<i64>().context("Invalid --dv")?);
            }
            "--dg" => {
                i += 1;
                let v = args.get(i).context("Missing value for --dg")?;
                dg = Some(v.parse::<i64>().context("Invalid --dg")?);
            }
            "--ln-eps" => {
                i += 1;
                let v = args.get(i).context("Missing value for --ln-eps")?;
                ln_eps = Some(v.parse::<f64>().context("Invalid --ln-eps")?);
            }
            "--gn-eps" => {
                i += 1;
                let v = args.get(i).context("Missing value for --gn-eps")?;
                gn_eps = Some(v.parse::<f64>().context("Invalid --gn-eps")?);
            }
            "--seed" => {
                i += 1;
                let v = args.get(i).context("Missing value for --seed")?;
                seed = Some(v.parse::<u64>().context("Invalid --seed")?);
            }
            "--validate-file" => {
                i += 1;
                validate_file = Some(
                    args.get(i)
                        .context("Missing value for --validate-file")?
                        .clone(),
                );
            }
            "--device" => {
                i += 1;
                device = Some(args.get(i).context("Missing value for --device")?.clone());
            }
            "--no-validate" => {
                no_validate = true;
            }
            "--accum" => {
                i += 1;
                let v = args.get(i).context("Missing value for --accum")?;
                accum = Some(v.parse::<usize>().context("Invalid --accum")?);
            }
            _ => bail!("Unknown option: {}", arg),
        }
        i += 1;
    }

    Ok(Command::Train {
        dataset,
        out,
        resume,
        steps,
        bsz,
        seq,
        accum,
        lr,
        layers,
        hidden,
        intermediate,
        dw,
        da,
        dv,
        dg,
        ln_eps,
        gn_eps,
        seed,
        validate_file,
        device,
        no_validate,
    })
}

/// Parse 'compress' subcommand arguments.
fn parse_compress(args: &[String]) -> Result<Command> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut model: Option<String> = None;
    let mut coder = CoderType::AC;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(Some("compress".to_string()))),
            "-m" | "--model" => {
                i += 1;
                model = Some(args.get(i).context("Missing value for --model")?.clone());
            }
            "-c" | "--coder" => {
                i += 1;
                let coder_str = args.get(i).context("Missing value for --coder")?;
                coder = parse_coder(coder_str)?;
            }
            _ if !arg.starts_with('-') => {
                if input.is_none() {
                    input = Some(arg.clone());
                } else if output.is_none() {
                    output = Some(arg.clone());
                } else {
                    bail!("Unexpected argument: {}", arg);
                }
            }
            _ => bail!("Unknown option: {}", arg),
        }
        i += 1;
    }

    Ok(Command::Compress {
        input: input.context("Missing input file")?,
        output: output.context("Missing output file")?,
        model: model.context("Missing --model path")?,
        coder,
    })
}

/// Parse 'decompress' subcommand arguments.
fn parse_decompress(args: &[String]) -> Result<Command> {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut model: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(Some("decompress".to_string()))),
            "-m" | "--model" => {
                i += 1;
                model = Some(args.get(i).context("Missing value for --model")?.clone());
            }
            _ if !arg.starts_with('-') => {
                if input.is_none() {
                    input = Some(arg.clone());
                } else if output.is_none() {
                    output = Some(arg.clone());
                } else {
                    bail!("Unexpected argument: {}", arg);
                }
            }
            _ => bail!("Unknown option: {}", arg),
        }
        i += 1;
    }

    Ok(Command::Decompress {
        input: input.context("Missing input file")?,
        output: output.context("Missing output file")?,
        model: model.context("Missing --model path")?,
    })
}

/// Parse 'self-test' subcommand arguments.
fn parse_self_test(args: &[String]) -> Result<Command> {
    let mut input: Option<String> = None;
    let mut model: Option<String> = None;
    let mut coder = CoderType::AC;
    let mut output: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(Some("self-test".to_string()))),
            "-m" | "--model" => {
                i += 1;
                model = Some(args.get(i).context("Missing value for --model")?.clone());
            }
            "-c" | "--coder" => {
                i += 1;
                let coder_str = args.get(i).context("Missing value for --coder")?;
                coder = parse_coder(coder_str)?;
            }
            "--output" => {
                i += 1;
                output = Some(args.get(i).context("Missing value for --output")?.clone());
            }
            _ if !arg.starts_with('-') => {
                if input.is_none() {
                    input = Some(arg.clone());
                } else {
                    bail!("Unexpected argument: {}", arg);
                }
            }
            _ => bail!("Unknown option: {}", arg),
        }
        i += 1;
    }

    Ok(Command::SelfTest {
        input: input.context("Missing input file")?,
        model: model.context("Missing --model path")?,
        coder,
        output,
    })
}

/// Parse 'entropy' subcommand arguments.
fn parse_entropy(args: &[String]) -> Result<Command> {
    let mut input: Option<String> = None;
    let mut model: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-h" | "--help" => return Ok(Command::Help(Some("entropy".to_string()))),
            "-m" | "--model" => {
                i += 1;
                model = Some(args.get(i).context("Missing value for --model")?.clone());
            }
            _ if !arg.starts_with('-') => {
                if input.is_none() {
                    input = Some(arg.clone());
                } else {
                    bail!("Unexpected argument: {}", arg);
                }
            }
            _ => bail!("Unknown option: {}", arg),
        }
        i += 1;
    }

    Ok(Command::Entropy {
        input: input.context("Missing input file")?,
        model: model.context("Missing --model path")?,
    })
}

/// Display help message for a command.
fn show_help(subcmd: Option<String>) {
    match subcmd.as_deref() {
        Some("compress") => println!("{}", COMPRESS_HELP),
        Some("decompress") => println!("{}", DECOMPRESS_HELP),
        Some("self-test") => println!("{}", SELF_TEST_HELP),
        Some("entropy") => println!("{}", ENTROPY_HELP),
        Some("train") => println!("{}", TRAIN_HELP),
        _ => println!("{}", HELP_TEXT),
    }
}

// =============================================================================
// Command Implementations
// =============================================================================

fn main() -> Result<()> {
    let cmd = parse_args()?;

    match cmd {
        Command::Help(subcmd) => {
            show_help(subcmd);
            Ok(())
        }
        Command::Version => {
            println!("rwkvzip {}", VERSION);
            Ok(())
        }
        Command::Compress {
            input,
            output,
            model,
            coder,
        } => cmd_compress(&input, &output, &model, coder),
        Command::Decompress {
            input,
            output,
            model,
        } => cmd_decompress(&input, &output, &model),
        Command::SelfTest {
            input,
            model,
            coder,
            output,
        } => cmd_self_test(&input, &model, coder, output.as_deref()),
        Command::Entropy { input, model } => cmd_entropy(&input, &model),
        Command::Train {
            dataset,
            out,
            resume,
            steps,
            bsz,
            seq,
            accum,
            lr,
            layers,
            hidden,
            intermediate,
            dw,
            da,
            dv,
            dg,
            ln_eps,
            gn_eps,
            seed,
            validate_file,
            device,
            no_validate,
        } => cmd_train(
            dataset,
            out,
            resume,
            steps,
            bsz,
            seq,
            accum,
            lr,
            layers,
            hidden,
            intermediate,
            dw,
            da,
            dv,
            dg,
            ln_eps,
            gn_eps,
            seed,
            validate_file,
            device,
            no_validate,
        ),
    }
}

fn cmd_train(
    dataset: Option<String>,
    out: Option<String>,
    resume: Option<String>,
    steps: Option<usize>,
    bsz: Option<i64>,
    seq: Option<i64>,
    accum: Option<usize>,
    lr: Option<f64>,
    layers: Option<usize>,
    hidden: Option<i64>,
    intermediate: Option<i64>,
    dw: Option<i64>,
    da: Option<i64>,
    dv: Option<i64>,
    dg: Option<i64>,
    ln_eps: Option<f64>,
    gn_eps: Option<f64>,
    seed: Option<u64>,
    validate_file: Option<String>,
    device: Option<String>,
    no_validate: bool,
) -> Result<()> {
    #[cfg(feature = "training")]
    {
        use rwkvzip::rwkv7::training::{train_enwik8, TrainConfig};
        let mut cfg = TrainConfig::default();
        if let Some(p) = dataset {
            cfg.dataset_path = p.into();
        }
        if let Some(p) = out {
            cfg.output_model_path = p.into();
        }
        if let Some(v) = steps {
            cfg.steps = v;
        }
        if let Some(v) = bsz {
            cfg.batch_size = v;
        }
        if let Some(v) = seq {
            cfg.seq_len = v;
        }
        if let Some(v) = accum {
            cfg.grad_accum_steps = v;
        }
        if let Some(v) = lr {
            cfg.lr = v;
        }
        if let Some(v) = layers {
            cfg.model_cfg.num_layers = v;
        }
        if let Some(v) = hidden {
            cfg.model_cfg.hidden_size = v;
            cfg.model_cfg.num_heads = v / cfg.model_cfg.head_dim;
        }
        if let Some(v) = intermediate {
            cfg.model_cfg.intermediate_size = v;
        }
        if let Some(v) = dw {
            cfg.model_cfg.decay_low_rank = v;
        }
        if let Some(v) = da {
            cfg.model_cfg.a_low_rank = v;
        }
        if let Some(v) = dv {
            cfg.model_cfg.v_low_rank = v;
        }
        if let Some(v) = dg {
            cfg.model_cfg.g_low_rank = v;
        }
        if let Some(v) = ln_eps {
            cfg.model_cfg.layer_norm_eps = v;
        }
        if let Some(v) = gn_eps {
            cfg.model_cfg.group_norm_eps = v;
        }
        if let Some(v) = seed {
            cfg.seed = v;
        }
        cfg.device = device;
        if let Some(v) = validate_file {
            cfg.validate_roundtrip_path = Some(v.into());
        } else if no_validate {
            cfg.validate_roundtrip_path = None;
        }

        // Handle resume functionality
        if let Some(resume_path) = resume {
            println!("Resuming training from existing model: {}", resume_path);

            // Check if resume file exists
            if !std::path::Path::new(&resume_path).exists() {
                bail!("Resume model file not found: {}", resume_path);
            }

            // Set the resume path in the config for the training function to use
            cfg.resume_model_path = Some(resume_path.into());
        }

        println!("Training RWKV7 byte-level model...");
        let report = train_enwik8(cfg)?;
        println!(
            "Done. loss={:.4} | {:.0} tok/s | saved {}",
            report.final_loss,
            report.tokens_per_sec,
            report.output_model_path.display()
        );
        return Ok(());
    }

    #[cfg(not(feature = "training"))]
    {
        let _ = (
            dataset,
            out,
            resume,
            steps,
            bsz,
            seq,
            lr,
            layers,
            hidden,
            intermediate,
            dw,
            da,
            dv,
            dg,
            ln_eps,
            gn_eps,
            seed,
            validate_file,
            device,
            no_validate,
        );
        bail!("This binary was built without the 'training' feature.");
    }
}

/// Compress a file.
fn cmd_compress(input: &str, output: &str, model: &str, coder: CoderType) -> Result<()> {
    println!("Loading model from {}...", model);
    let mut compressor = Compressor::new(model)?;

    println!("Reading input from {}...", input);
    let data = fs::read(input).context("Failed to read input file")?;

    println!("Compressing {} bytes with {}...", data.len(), coder);
    let (compressed, stats) = compress_with_stats(&mut compressor, &data, coder)?;

    println!("Writing output to {}...", output);
    let mut file = BufWriter::new(File::create(output)?);
    file.write_all(&compressed)?;
    file.flush()?;

    println!("\nCompression complete!");
    println!("{}", stats);

    Ok(())
}

/// Decompress a file.
fn cmd_decompress(input: &str, output: &str, model: &str) -> Result<()> {
    println!("Loading model from {}...", model);
    let mut compressor = Compressor::new(model)?;

    println!("Reading compressed data from {}...", input);
    let compressed = fs::read(input).context("Failed to read compressed file")?;

    println!("Decompressing {} bytes...", compressed.len());
    let start = Instant::now();
    let decompressed = compressor.decompress(&compressed)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Writing output to {}...", output);
    let mut file = BufWriter::new(File::create(output)?);
    file.write_all(&decompressed)?;
    file.flush()?;

    println!("\nDecompression complete!");
    println!(
        "{} bytes -> {} bytes | time={:.2}s | {:.0} B/s",
        compressed.len(),
        decompressed.len(),
        elapsed,
        decompressed.len() as f64 / elapsed
    );

    Ok(())
}

/// Run self-test: compress, decompress, and verify roundtrip.
fn cmd_self_test(
    input: &str,
    model: &str,
    coder: CoderType,
    output_path: Option<&str>,
) -> Result<()> {
    println!("=== rwkvzip Self-Test ===\n");

    println!("Loading model from {}...", model);
    let mut compressor = Compressor::new(model)?;

    println!("Reading input from {}...", input);
    let original = fs::read(input).context("Failed to read input file")?;
    let original_crc = crc32(&original);

    println!(
        "Original: {} bytes, CRC32=0x{:08X}\n",
        original.len(),
        original_crc
    );

    // Compress
    println!("Compressing with {}...", coder);
    let compress_start = Instant::now();
    let (compressed, compress_stats) = compress_with_stats(&mut compressor, &original, coder)?;
    let compress_time = compress_start.elapsed().as_secs_f64();

    println!("Compressed: {} bytes", compressed.len());
    println!("  Ratio: {:.3}x", compress_stats.ratio);
    println!("  Bits/byte: {:.3}", compress_stats.bits_per_byte);
    println!(
        "  Time: {:.2}s ({:.0} B/s)\n",
        compress_time,
        original.len() as f64 / compress_time
    );

    // Decompress
    println!("Decompressing...");
    let decompress_start = Instant::now();
    let decompressed = compressor.decompress(&compressed)?;
    let decompress_time = decompress_start.elapsed().as_secs_f64();

    let decompressed_crc = crc32(&decompressed);

    println!(
        "Decompressed: {} bytes, CRC32=0x{:08X}",
        decompressed.len(),
        decompressed_crc
    );
    println!(
        "  Time: {:.2}s ({:.0} B/s)\n",
        decompress_time,
        decompressed.len() as f64 / decompress_time
    );

    // Verify roundtrip integrity
    let success = original == decompressed;

    if success {
        println!("✓ PASS: Roundtrip successful!");
        println!("  Original and decompressed data match exactly.");
    } else {
        println!("✗ FAIL: Roundtrip failed!");
        println!("  Original CRC32: 0x{:08X}", original_crc);
        println!("  Decompressed CRC32: 0x{:08X}", decompressed_crc);

        if original.len() != decompressed.len() {
            println!(
                "  Length mismatch: {} vs {}",
                original.len(),
                decompressed.len()
            );
        } else {
            // Locate first byte difference for debugging
            for (i, (&a, &b)) in original.iter().zip(decompressed.iter()).enumerate() {
                if a != b {
                    println!(
                        "  First difference at byte {}: 0x{:02X} vs 0x{:02X}",
                        i, a, b
                    );
                    break;
                }
            }
        }
    }

    // Write roundtrip output if requested
    if let Some(path) = output_path {
        println!("\nWriting roundtrip output to {}...", path);
        fs::write(path, &decompressed)?;
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Input: {}", input);
    println!("Model: {}", model);
    println!("Coder: {}", coder);
    println!(
        "{} bytes -> {} bytes | bits/byte={:.3} | time={:.2}s",
        original.len(),
        compressed.len(),
        compress_stats.bits_per_byte,
        compress_time + decompress_time
    );

    if !success {
        bail!("Self-test failed: roundtrip mismatch");
    }

    Ok(())
}

/// Calculate and display cross-entropy.
fn cmd_entropy(input: &str, model: &str) -> Result<()> {
    println!("Loading model from {}...", model);
    let mut compressor = Compressor::new(model)?;

    println!("Reading input from {}...", input);
    let data = fs::read(input).context("Failed to read input file")?;

    println!("Calculating cross-entropy for {} bytes...\n", data.len());

    let start = Instant::now();
    let bpb = compressor.cross_entropy(&data)?;
    let elapsed = start.elapsed().as_secs_f64();

    println!("Cross-entropy: {:.4} bits/byte", bpb);
    println!(
        "Theoretical minimum: {:.0} bytes",
        (data.len() as f64 * bpb) / 8.0
    );
    println!(
        "Time: {:.2}s ({:.0} B/s)",
        elapsed,
        data.len() as f64 / elapsed
    );

    Ok(())
}
