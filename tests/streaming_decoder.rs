//! Run with POCKET_TTS_MODEL_DIR pointing to model.safetensors, tokenizer.model,
//! and voices/alba.safetensors: cargo test --release --test streaming_decoder -- --ignored.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use pocket_tts_ios::{
    config::TTSConfig,
    models::{
        mimi::{MimiConfig, MimiDecoder},
        pocket_tts::PocketTTSModel,
    },
};

#[test]
#[ignore = "requires Pocket TTS model files in POCKET_TTS_MODEL_DIR"]
fn streamed_speech_matches_batch_decoding_across_cache_rollover_and_reset() {
    let model_dir = std::env::var("POCKET_TTS_MODEL_DIR").expect("set POCKET_TTS_MODEL_DIR");
    let device = Device::Cpu;
    let mut model = PocketTTSModel::load(&model_dir, &device).unwrap();
    model
        .configure(TTSConfig {
            use_fixed_seed: true,
            seed: 42,
            ..TTSConfig::default()
        })
        .unwrap();
    let model_path = std::path::Path::new(&model_dir).join("model.safetensors");
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device).unwrap() };
    let mean = vb.pp("flow_lm").get((32,), "emb_mean").unwrap();
    let std = vb.pp("flow_lm").get((32,), "emb_std").unwrap();
    let mut decoder = MimiDecoder::new(MimiConfig::default(), vb.pp("mimi")).unwrap();

    for text in [
        "The quick brown fox jumps over the lazy dog.",
        "When the first phrase ends, the voice should stay clear for the rest of this sentence.",
    ] {
        // Use identical TTS-generated latents to isolate streaming decoder errors.
        let (_, values, shape) = model.synthesize_with_latents(text).unwrap();
        let frames = shape[1];
        assert!(
            frames * 16 > 250 + 16,
            "speech must continue past the 250-position cache rollover"
        );
        let latents = Tensor::from_vec(values, &shape, &device)
            .unwrap()
            .broadcast_mul(&std)
            .unwrap()
            .broadcast_add(&mean)
            .unwrap();
        let expected: Vec<f32> = decoder
            .forward_streaming(&latents)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        for batch_size in [1, 4, 7] {
            decoder.reset_decoder_cache();
            let mut state = decoder.init_streaming_state(1, &device).unwrap();
            let mut actual = Vec::new();
            for start in (0..frames).step_by(batch_size) {
                let batch = latents.narrow(1, start, batch_size.min(frames - start)).unwrap();
                let samples: Vec<f32> = decoder
                    .forward_streaming_stateful(&batch, &mut state)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                actual.extend(samples);
            }
            assert_eq!(actual.len(), expected.len(), "streaming changed the speech duration");
            let max_error = actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(
                max_error < 0.0001,
                "batch size {batch_size}, text {text:?}: maximum sample error {max_error}"
            );
        }
    }
}
