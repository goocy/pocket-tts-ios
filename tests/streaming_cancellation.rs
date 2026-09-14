//! Run with POCKET_TTS_MODEL_DIR pointing to model.safetensors, tokenizer.model,
//! and voices/alba.safetensors: cargo test --release --test streaming_cancellation -- --ignored.
use pocket_tts_ios::{AudioChunk, PocketTTSEngine, TTSEventHandler};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Handler {
    engine: Arc<PocketTTSEngine>,
    chunks: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
    cancel_after_first: bool,
}

impl TTSEventHandler for Handler {
    fn on_audio_chunk(&self, chunk: AudioChunk) {
        if chunk.audio_data.is_empty() {
            return;
        }
        let count = self.chunks.fetch_add(1, Ordering::SeqCst) + 1;
        if self.cancel_after_first && count == 1 {
            self.engine.cancel();
        }
    }
    fn on_progress(&self, _: f32) {}
    fn on_complete(&self) {
        self.completed.fetch_add(1, Ordering::SeqCst);
    }
    fn on_error(&self, message: String) {
        panic!("Synthesis failed: {message}");
    }
}

#[test]
#[ignore = "requires Pocket TTS model files in POCKET_TTS_MODEL_DIR"]
fn cancelling_stops_audio_and_allows_the_next_stream() {
    let model_dir = std::env::var("POCKET_TTS_MODEL_DIR").expect("set POCKET_TTS_MODEL_DIR");
    let engine = Arc::new(PocketTTSEngine::new(model_dir).unwrap());
    let text = "The response button reads the message aloud. Tap it again to stop, or choose another response to switch what you are listening to. The text stays on your phone throughout the process.";
    for cancel_after_first in [true, false] {
        let chunks = Arc::new(AtomicUsize::new(0));
        let completed = Arc::new(AtomicUsize::new(0));
        engine
            .start_true_streaming(
                text.into(),
                Box::new(Handler {
                    engine: engine.clone(),
                    chunks: chunks.clone(),
                    completed: completed.clone(),
                    cancel_after_first,
                }),
            )
            .unwrap();
        if cancel_after_first {
            assert_eq!(chunks.load(Ordering::SeqCst), 1, "audio continued after cancellation");
            assert_eq!(completed.load(Ordering::SeqCst), 0, "cancelled speech reported completion");
        } else {
            assert!(chunks.load(Ordering::SeqCst) > 1, "the next stream stayed cancelled");
            assert!(completed.load(Ordering::SeqCst) > 0, "the next stream did not complete");
        }
    }
}
