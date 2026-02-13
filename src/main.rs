use std::io::{self, BufRead};
use std::process;
use id_tokenizer::*;

fn parse_vocab(s: &str) -> Vocab {
    match s {
        "2048" => Vocab::V2048,
        "4096" => Vocab::V4096,
        "8192" => Vocab::V8192,
        "16384" => Vocab::V16384,
        "32768" => Vocab::V32768,
        _ => {
            eprintln!("error: vocab must be 2048, 4096, 8192, 16384, or 32768");
            process::exit(1);
        }
    }
}

fn parse_style(s: &str) -> Style {
    match s {
        "memorable" => Style::Memorable,
        "token" => Style::Token,
        _ => {
            eprintln!("error: style must be 'memorable' or 'token'");
            process::exit(1);
        }
    }
}

fn usage() -> ! {
    eprintln!("Usage:");
    eprintln!("  id-tokenizer encode [type] [value] [--vocab N] [--style S]");
    eprintln!("  id-tokenizer decode [type] [value] [--vocab N] [--style S]");
    eprintln!();
    eprintln!("If type is omitted, auto-detects UUID vs integer.");
    eprintln!("If value is omitted, reads lines from stdin.");
    eprintln!();
    eprintln!("Types:");
    eprintln!("  uuid   Fixed-width UUID encoding (with variant stripping + version check)");
    eprintln!("  int    Variable-length integer (compact, no zero-padding)");
    eprintln!("  u128   Fixed-width 128-bit integer (with bit-mixing)");
    eprintln!("  u64    Fixed-width 64-bit integer (with bit-mixing)");
    eprintln!("  u32    Fixed-width 32-bit integer (with bit-mixing)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  id-tokenizer encode 550e8400-e29b-41d4-a716-446655440000");
    eprintln!("  id-tokenizer encode 42 --style token");
    eprintln!("  echo 12345 | id-tokenizer encode --style token");
    eprintln!("  echo 12345 | id-tokenizer encode u64");
    eprintln!();
    eprintln!("Vocab sizes: 2048 (default), 4096, 8192, 16384, 32768");
    eprintln!("Styles: memorable (default), token");
    process::exit(1);
}

const KNOWN_TYPES: &[&str] = &["uuid", "int", "u128", "u64", "u32"];

fn is_uuid(s: &str) -> bool {
    uuid::Uuid::parse_str(s).is_ok()
}

fn detect_type(value: &str) -> &'static str {
    if is_uuid(value) { "uuid" } else { "int" }
}

fn do_encode(value: &str, kind: &str, vocab: Vocab, style: Style) {
    let phrase = match kind {
        "uuid" => {
            let uuid = uuid::Uuid::parse_str(value).unwrap_or_else(|e| {
                eprintln!("error: invalid UUID: {e}");
                process::exit(1);
            });
            encode_uuid(uuid.as_bytes(), vocab, style)
        }
        "int" => {
            let v: u128 = value.parse().unwrap_or_else(|e| {
                eprintln!("error: invalid integer: {e}");
                process::exit(1);
            });
            encode_int(v, vocab, style)
        }
        "u128" => {
            let v: u128 = value.parse().unwrap_or_else(|e| {
                eprintln!("error: invalid u128: {e}");
                process::exit(1);
            });
            encode_u128(v, vocab, style)
        }
        "u64" => {
            let v: u64 = value.parse().unwrap_or_else(|e| {
                eprintln!("error: invalid u64: {e}");
                process::exit(1);
            });
            encode_u64(v, vocab, style)
        }
        "u32" => {
            let v: u32 = value.parse().unwrap_or_else(|e| {
                eprintln!("error: invalid u32: {e}");
                process::exit(1);
            });
            encode_u32(v, vocab, style)
        }
        _ => unreachable!(),
    };
    println!("{phrase}");
}

fn do_decode(phrase: &str, kind: &str, vocab: Vocab, style: Style) {
    let result = match kind {
        "uuid" => decode_uuid(phrase, vocab, style)
            .map(|b| uuid::Uuid::from_bytes(b).to_string()),
        "int" => decode_int(phrase, vocab, style).map(|v| v.to_string()),
        "u128" => decode_u128(phrase, vocab, style).map(|v| v.to_string()),
        "u64" => decode_u64(phrase, vocab, style).map(|v| v.to_string()),
        "u32" => decode_u32(phrase, vocab, style).map(|v| v.to_string()),
        _ => unreachable!(),
    };
    match result {
        Ok(s) => println!("{s}"),
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn try_decode_auto(phrase: &str, vocab: Vocab, style: Style) {
    // Try UUID first (fixed word count + version validation makes it unambiguous)
    if let Ok(bytes) = decode_uuid(phrase, vocab, style) {
        println!("{}", uuid::Uuid::from_bytes(bytes));
        return;
    }
    // Fall back to variable-length int
    match decode_int(phrase, vocab, style) {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }

    let cmd = &args[1];
    if cmd == "--help" || cmd == "-h" {
        usage();
    }

    // Parse flags from the end, collect positional args
    let mut vocab = Vocab::V2048;
    let mut style = Style::Memorable;
    let mut positional: Vec<&str> = Vec::new();

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--vocab" | "-v" => {
                i += 1;
                if i >= args.len() { usage(); }
                vocab = parse_vocab(&args[i]);
            }
            "--style" | "-s" => {
                i += 1;
                if i >= args.len() { usage(); }
                style = parse_style(&args[i]);
            }
            s if s.starts_with('-') => {
                eprintln!("error: unknown flag '{s}'");
                usage();
            }
            s => positional.push(s),
        }
        i += 1;
    }

    // positional: [type?, value?]
    let (explicit_type, inline_value) = match positional.len() {
        0 => (None, None),
        1 => {
            if KNOWN_TYPES.contains(&positional[0]) {
                (Some(positional[0]), None)
            } else {
                (None, Some(positional[0]))
            }
        }
        2 => {
            if KNOWN_TYPES.contains(&positional[0]) {
                (Some(positional[0]), Some(positional[1]))
            } else {
                eprintln!("error: unknown type '{}'. Known types: uuid, int, u128, u64, u32", positional[0]);
                process::exit(1);
            }
        }
        _ => {
            eprintln!("error: too many arguments");
            usage();
        }
    };

    match cmd.as_str() {
        "encode" => {
            if let Some(value) = inline_value {
                let kind = explicit_type.unwrap_or_else(|| detect_type(value));
                do_encode(value, kind, vocab, style);
            } else {
                // Read from stdin
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    let line = line.unwrap_or_else(|e| {
                        eprintln!("error: {e}");
                        process::exit(1);
                    });
                    let value = line.trim();
                    if value.is_empty() { continue; }
                    let kind = explicit_type.unwrap_or_else(|| detect_type(value));
                    do_encode(value, kind, vocab, style);
                }
            }
        }
        "decode" => {
            if let Some(phrase) = inline_value {
                if let Some(kind) = explicit_type {
                    do_decode(phrase, kind, vocab, style);
                } else {
                    try_decode_auto(phrase, vocab, style);
                }
            } else {
                let stdin = io::stdin();
                for line in stdin.lock().lines() {
                    let line = line.unwrap_or_else(|e| {
                        eprintln!("error: {e}");
                        process::exit(1);
                    });
                    let phrase = line.trim().to_string();
                    if phrase.is_empty() { continue; }
                    if let Some(kind) = explicit_type {
                        do_decode(&phrase, kind, vocab, style);
                    } else {
                        try_decode_auto(&phrase, vocab, style);
                    }
                }
            }
        }
        _ => usage(),
    }
}
