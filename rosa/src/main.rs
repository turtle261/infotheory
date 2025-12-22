use std::fs::File;
use std::io::Read;

use rosaplus::RosaPlus;

fn main() {
    let file_path = match std::env::var("ROSAPLUS_EX_PATH") {
        Ok(p) if !p.is_empty() => p,
        _ => {
            println!(
                "ERROR: ROSAPLUS_EX_PATH environment variable must be set to run this example."
            );
            println!("This variable should contain the path to a local training text file.");
            println!("Example: export ROSAPLUS_EX_PATH='/path/to/your/training.txt'");
            std::process::exit(1);
        }
    };

    println!("Using training data file from ROSAPLUS_EX_PATH: {}", file_path);

    let mut f = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) => {
            println!("ERROR: Failed to read text from {}: {}", file_path, e);
            std::process::exit(1);
        }
    };
    let mut text = Vec::new();
    if let Err(e) = f.read_to_end(&mut text) {
        println!("ERROR: Failed to read text from {}: {}", file_path, e);
        std::process::exit(1);
    }
    println!("Loaded text from local file.");

    let mut m = RosaPlus::new(1048576, false, 4, 0);
    m.train_example(&text);
    m.build_lm();

    let prompt = b"ROMEO:";
    let max_tokens = 256;
    let cont = m.generate(prompt, max_tokens).expect("generate failed");
    print!("{}{}\n", std::str::from_utf8(prompt).unwrap(), cont);

    let skip_io = std::env::var("ROSAPLUS_SKIP_IO").unwrap_or_default();
    if skip_io.is_empty() {
        let _ = m.save("rosa-model.json");
        let _ = RosaPlus::load("rosa-model.json");
    }
}
