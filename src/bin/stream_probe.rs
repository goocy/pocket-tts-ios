//! Batch render against streamed render of the same sentences, with a
//! fixed seed so both sample the same latents. Reports how long each takes to
//! the first audio and to the last, when each streamed chunk arrived against
//! when playback at 1x would need it, and how far the samples differ.
//!
//!   cargo run --release --bin stream-probe -- <model-dir> [voice-id] [out-dir]

use std::time::Instant;

use candle_core::Device;
use pocket_tts_ios::audio;
use pocket_tts_ios::config::TTSConfig;
use pocket_tts_ios::models::pocket_tts::PocketTTSModel;

const SENTENCES: &[&str] = &[
    "Yes.",
    "The quick brown fox jumps over the lazy dog.",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness.",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args.get(1).expect("model dir");
    let voice = args.get(2).cloned().unwrap_or_default();
    let out_dir = args.get(3).cloned();

    let mut model = PocketTTSModel::load(model_dir, &Device::Cpu).expect("load");
    model
        .configure(TTSConfig {
            voice_id: voice,
            temperature: 0.7,
            top_p: 0.9,
            use_fixed_seed: true,
            seed: 42,
            ..TTSConfig::default()
        })
        .expect("configure");
    let rate = model.sample_rate() as f64;

    // One throwaway render so neither side pays for cold caches.
    let _ = model.synthesize("Warm up.");

    for (index, text) in SENTENCES.iter().enumerate() {
        let started = Instant::now();
        let batch = model.synthesize(text).expect("batch");
        let batch_s = started.elapsed().as_secs_f64();

        let started = Instant::now();
        let mut streamed: Vec<f32> = Vec::new();
        let mut arrivals: Vec<(f64, usize)> = Vec::new();
        model
            .synthesize_true_streaming(text, |samples, _final| {
                streamed.extend_from_slice(samples);
                arrivals.push((started.elapsed().as_secs_f64(), streamed.len()));
                true
            })
            .expect("stream");
        let stream_s = started.elapsed().as_secs_f64();

        // Playback starts when the first chunk lands. Chunk n is needed once
        // everything before it has played; the margin is how early it came.
        let first = arrivals.first().map(|a| a.0).unwrap_or(0.0);
        let mut worst = f64::INFINITY;
        let mut before = 0usize;
        for (at, total) in &arrivals {
            let needed = first + before as f64 / rate;
            worst = worst.min(needed - at);
            before = *total;
        }

        let shared = batch.len().min(streamed.len());
        let mut peak = 0f32;
        let mut sum = 0f64;
        for i in 0..shared {
            let d = (batch[i] - streamed[i]).abs();
            peak = peak.max(d);
            sum += (d as f64) * (d as f64);
        }
        let rms = (sum / shared.max(1) as f64).sqrt();

        println!(
            "RESULT #{index} audio={:.2}s batch: first=last={:.2}s | stream: first={:.2}s last={:.2}s chunks={} worst-margin={:+.2}s | samples batch={} stream={} peak-diff={:.6} rms-diff={:.6}",
            batch.len() as f64 / rate,
            batch_s,
            first,
            stream_s,
            arrivals.len(),
            worst,
            batch.len(),
            streamed.len(),
            peak,
            rms,
        );

        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir).ok();
            for (name, samples) in [("batch", &batch), ("stream", &streamed)] {
                let wav = audio::samples_to_wav(samples, model.sample_rate()).expect("wav");
                std::fs::write(format!("{dir}/{index}-{name}.wav"), wav).ok();
            }
        }
    }
}
