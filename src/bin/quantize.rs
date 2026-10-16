//! `quantize <model.safetensors> <model.q8.gguf>` — the same conversion
//! `quantize_model` runs on a device after a download, for use on a Mac.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: quantize <model.safetensors> <model.q8.gguf>");
        std::process::exit(2);
    }
    if let Err(e) = pocket_tts_ios::quantize_model(args[1].clone(), args[2].clone()) {
        eprintln!("error: {:?}", e);
        std::process::exit(1);
    }
}
