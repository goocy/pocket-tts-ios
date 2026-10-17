//! Complete Pocket TTS Model
//!
//! Combines FlowLM transformer (with FlowNet) and Mimi decoder
//! into a complete text-to-speech pipeline.
//!
//! Portions of this file derived from:
//! https://github.com/babybirdprd/pocket-tts
//! Licensed under MIT

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::flowlm::{FlowLM, FlowLMConfig, LatentStreamControl};
use super::mimi::{MimiConfig, MimiDecoder};
use crate::config::TTSConfig;
use crate::error::PocketTTSError;
use crate::modules::embeddings::{VoiceBank, VoiceEmbedding};
use crate::tokenizer::PocketTokenizer;

/// Complete Pocket TTS Model
pub struct PocketTTSModel {
    flowlm: FlowLM,
    mimi: MimiDecoder,
    tokenizer: PocketTokenizer,
    voice_bank: VoiceBank,
    device: Device,
    config: TTSConfig,
    /// The id `config.voice_id` resolved to by `configure` (an empty id is
    /// the bank's first), and that voice's state, read from its file then.
    /// It is the only voice held in memory.
    voice_id: String,
    voice: VoiceEmbedding,
    custom_voice: Option<VoiceEmbedding>,
    /// Pre-loaded noise tensors for correlation testing.
    /// When set, these are used instead of random noise during FlowNet generation.
    /// Each tensor corresponds to one autoregressive step.
    noise_tensors: Option<Vec<Tensor>>,
    /// Where the Mimi decoder lives. Metal when `POCKET_TTS_MIMI_METAL=1`
    /// and the build has the metal feature; otherwise the model device.
    /// The FlowLM stays on the CPU either way: its per-frame matvecs are too
    /// small to amortise GPU dispatch, while Mimi's convolutions over a whole
    /// sentence are exactly what a GPU is for.
    mimi_device: Device,
}

impl PocketTTSModel {
    /// Load model from directory containing all components
    pub fn load<P: AsRef<Path>>(model_dir: P, device: &Device) -> std::result::Result<Self, PocketTTSError> {
        let model_dir = model_dir.as_ref();

        // A model directory holds either the quantized GGUF (written once by
        // `quantize_model`) or the downloaded safetensors. See `quantized_model`.
        let gguf_path = model_dir.join(crate::quantized_model::GGUF_FILE_NAME);
        let model_path = model_dir.join(crate::quantized_model::SAFETENSORS_FILE_NAME);
        let plain: Option<std::collections::HashMap<String, Tensor>> = if gguf_path.exists() {
            Some(
                crate::quantized_model::load_gguf(&gguf_path, device)
                    .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?,
            )
        } else {
            crate::modules::qlinear::clear_registry();
            None
        };
        let vb = match &plain {
            Some(tensors) => VarBuilder::from_tensors(tensors.clone(), DType::F32, device),
            None => unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, device)
                    .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?
            },
        };

        // Load tokenizer (SentencePiece .model format)
        let tokenizer_path = model_dir.join("tokenizer.model");
        let tokenizer = PocketTokenizer::from_file(&tokenizer_path)?;

        // List the voices, and read only the one an unconfigured model speaks
        // in: the bank's first.
        let voices_dir = model_dir.join("voices");
        let voice_bank = VoiceBank::index_dir(&voices_dir)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Failed to list voices: {}", e)))?;
        let voice_id = voice_bank
            .ids()
            .first()
            .cloned()
            .ok_or_else(|| PocketTTSError::ModelLoadFailed(format!("no voice files in {:?}", voices_dir)))?;
        let voice = Self::read_voice(&voice_bank, &voice_id, device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?;

        // Initialize model components
        let flowlm_config = FlowLMConfig::default();
        let flowlm = FlowLM::new(flowlm_config.clone(), vb.pp("flow_lm"), device)
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("FlowLM: {}", e)))?;

        let mimi_config = MimiConfig {
            latent_dim: flowlm_config.latent_dim,
            ..MimiConfig::default()
        };
        let mimi_device = if std::env::var("POCKET_TTS_MIMI_METAL").map(|v| v == "1").unwrap_or(false) {
            match Device::new_metal(0) {
                Ok(d) => {
                    eprintln!("[Mimi] decoder on Metal");
                    d
                },
                Err(e) => {
                    eprintln!("[Mimi] Metal unavailable ({}), decoder on CPU", e);
                    device.clone()
                },
            }
        } else {
            device.clone()
        };
        let vb_mimi = if mimi_device.same_device(device) {
            vb.clone()
        } else if let Some(tensors) = &plain {
            let moved = tensors
                .iter()
                .filter(|(k, _)| k.starts_with("mimi."))
                .map(|(k, v)| Ok((k.clone(), v.to_device(&mimi_device)?)))
                .collect::<candle_core::Result<std::collections::HashMap<String, Tensor>>>()
                .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?;
            VarBuilder::from_tensors(moved, DType::F32, &mimi_device)
        } else {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &mimi_device)
                    .map_err(|e| PocketTTSError::ModelLoadFailed(e.to_string()))?
            }
        };
        let mimi = MimiDecoder::new(mimi_config, vb_mimi.pp("mimi"))
            .map_err(|e| PocketTTSError::ModelLoadFailed(format!("Mimi: {}", e)))?;
        crate::modules::qlinear::clear_registry();

        Ok(Self {
            flowlm,
            mimi,
            mimi_device,
            tokenizer,
            voice_bank,
            device: device.clone(),
            config: TTSConfig::default(),
            voice_id,
            voice,
            custom_voice: None,
            noise_tensors: None,
        })
    }

    /// Configure synthesis parameters
    ///
    /// Also resolves `config.voice_id` against the bank and reads that
    /// voice's file, unless it is the voice already held. A voice that is not
    /// installed is refused here, by name, instead of falling through to a
    /// render with no voice embedding at all.
    pub fn configure(&mut self, config: TTSConfig) -> std::result::Result<(), PocketTTSError> {
        config.validate().map_err(PocketTTSError::InvalidConfig)?;

        let id = if config.voice_id.is_empty() {
            self.voice_bank.ids()[0].clone()
        } else {
            config.voice_id.clone()
        };
        if id != self.voice_id {
            self.voice = Self::read_voice(&self.voice_bank, &id, &self.device)?;
            self.voice_id = id;
        }

        self.config = config;
        Ok(())
    }

    /// The state of the voice listed under `id`, read from its file.
    fn read_voice(bank: &VoiceBank, id: &str, device: &Device) -> std::result::Result<VoiceEmbedding, PocketTTSError> {
        let path = bank.path_of(id).ok_or_else(|| {
            PocketTTSError::InvalidConfig(format!(
                "no voice with id '{}' is installed; installed voices: {}",
                id,
                bank.ids().join(", ")
            ))
        })?;
        VoiceEmbedding::from_file(path, device)
            .map_err(|e| PocketTTSError::InvalidConfig(format!("voice '{}' could not be read from {:?}: {}", id, path, e)))
    }

    /// Set custom voice from reference audio embedding
    pub fn set_custom_voice(&mut self, embedding: VoiceEmbedding) {
        self.custom_voice = Some(embedding);
    }

    /// Clear custom voice (use built-in)
    pub fn clear_custom_voice(&mut self) {
        self.custom_voice = None;
    }

    /// Load pre-captured noise tensors from a directory for correlation testing.
    ///
    /// The directory should contain .npy files named like:
    /// `phrase_XX_noise_step_000.npy`, `phrase_XX_noise_step_001.npy`, etc.
    ///
    /// These tensors are used instead of random noise during FlowNet generation,
    /// eliminating RNG differences between Python and Rust for correlation measurement.
    pub fn load_noise_tensors(
        &mut self,
        noise_dir: &Path,
        phrase_id: &str,
    ) -> std::result::Result<usize, PocketTTSError> {
        let mut tensors = Vec::new();
        let mut step = 0;

        loop {
            let npy_path = noise_dir.join(format!("{}_noise_step_{:03}.npy", phrase_id, step));
            if !npy_path.exists() {
                break;
            }

            // Load .npy file
            let data = std::fs::read(&npy_path).map_err(|e| {
                PocketTTSError::ModelLoadFailed(format!("Failed to read noise file {:?}: {}", npy_path, e))
            })?;

            // Parse .npy header to get shape and find data offset
            let tensor = Self::parse_npy_f32(&data, &self.device).map_err(|e| {
                PocketTTSError::ModelLoadFailed(format!("Failed to parse noise file {:?}: {}", npy_path, e))
            })?;

            tensors.push(tensor);
            step += 1;
        }

        let count = tensors.len();
        if count > 0 {
            eprintln!(
                "[PocketTTS] Loaded {} noise tensors from {:?} for {}",
                count, noise_dir, phrase_id
            );
            self.noise_tensors = Some(tensors);
        } else {
            eprintln!(
                "[PocketTTS] WARNING: No noise tensors found for {} in {:?}",
                phrase_id, noise_dir
            );
            self.noise_tensors = None;
        }
        Ok(count)
    }

    /// Clear pre-loaded noise tensors
    pub fn clear_noise_tensors(&mut self) {
        self.noise_tensors = None;
    }

    /// Parse a NumPy .npy file containing float32 data into a Tensor
    fn parse_npy_f32(data: &[u8], device: &Device) -> std::result::Result<Tensor, String> {
        // Validate magic number
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            return Err("Invalid .npy magic number".to_string());
        }

        let major = data[6];
        let _minor = data[7];

        // Get header length
        let header_len = if major == 1 {
            u16::from_le_bytes([data[8], data[9]]) as usize
        } else if major == 2 {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
        } else {
            return Err(format!("Unsupported .npy version {}", major));
        };

        let header_start = if major == 1 { 10 } else { 12 };
        let data_start = header_start + header_len;

        // Parse header to extract shape
        let header_str =
            std::str::from_utf8(&data[header_start..data_start]).map_err(|e| format!("Invalid header UTF-8: {}", e))?;

        // Extract shape from header dict, e.g., "'shape': (1, 1, 32)"
        let shape = Self::parse_npy_shape(header_str)?;

        // Read float32 data
        let float_data = &data[data_start..];
        let n_floats = float_data.len() / 4;
        let values: Vec<f32> = (0..n_floats)
            .map(|i| {
                f32::from_le_bytes([
                    float_data[i * 4],
                    float_data[i * 4 + 1],
                    float_data[i * 4 + 2],
                    float_data[i * 4 + 3],
                ])
            })
            .collect();

        Tensor::from_vec(values, shape.as_slice(), device).map_err(|e| format!("Tensor creation failed: {}", e))
    }

    /// Parse shape tuple from .npy header string
    fn parse_npy_shape(header: &str) -> std::result::Result<Vec<usize>, String> {
        // Find 'shape': (...) in the header dict
        let shape_start = header
            .find("'shape':")
            .or_else(|| header.find("\"shape\":"))
            .ok_or("No 'shape' field in .npy header")?;

        let paren_start = header[shape_start..].find('(').ok_or("No '(' in shape")? + shape_start;
        let paren_end = header[paren_start..].find(')').ok_or("No ')' in shape")? + paren_start;

        let shape_str = &header[paren_start + 1..paren_end];
        let dims: Vec<usize> = shape_str
            .split(',')
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    trimmed.parse().ok()
                }
            })
            .collect();

        if dims.is_empty() {
            return Err("Empty shape".to_string());
        }

        Ok(dims)
    }

    /// Synthesize text to audio (SYNC MODE)
    ///
    /// **NOTE: Not for current on-device use.** Latency is king for on-device TTS.
    /// Use [`synthesize_true_streaming`] instead, which provides audio chunks
    /// via callback with ~200ms Time To First Audio (TTFA).
    ///
    /// This sync mode exists for:
    /// - Reference/debugging purposes
    /// - Batch processing scenarios where latency doesn't matter
    /// - Comparison testing against streaming mode
    ///
    /// The sync mode processes all tokens at once and returns complete audio,
    /// which has higher latency but uses NON-CAUSAL transformer attention for
    /// potentially different audio characteristics.
    pub fn synthesize(&mut self, text: &str) -> std::result::Result<Vec<f32>, PocketTTSError> {
        eprintln!("[PocketTTS] synthesize called with text len: {}", text.len());
        let prof_start = std::time::Instant::now();

        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;
        eprintln!("[PocketTTS] tokenized to {} tokens: {:?}", token_ids.len(), token_ids);

        // Create tensor
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        eprintln!("[PocketTTS] token tensor shape: {:?}", token_tensor.dims());

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            Some(&self.voice)
        };
        eprintln!("[PocketTTS] voice embedding loaded: {}", voice.is_some());

        // Reset caches for new sequence
        self.flowlm.reset_cache();

        // Generate latents with FlowLM + FlowNet
        // Reference implementation uses lsd_decode_steps = 1 (consistency model)
        // Single step is sufficient as the model is trained with consistency distillation
        let num_flow_steps = 1;
        eprintln!(
            "[PocketTTS] generating latents with {} flow step (consistency model)",
            num_flow_steps
        );
        let seed = if self.config.use_fixed_seed {
            Some(self.config.seed as u64)
        } else {
            None
        };
        let noise_ref = self.noise_tensors.as_deref();
        let prof_lm = std::time::Instant::now();
        let latents = self
            .flowlm
            .generate_latents(&token_tensor, voice, num_flow_steps, self.config.temperature, seed, noise_ref)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;
        let prof_lm_ms = prof_lm.elapsed().as_secs_f64() * 1e3;
        let prof_frames = latents.dim(1).unwrap_or(0);
        eprintln!("[PocketTTS] latents shape: {:?}", latents.dims());

        // DIAGNOSTIC: Log latent statistics to verify FlowLM output quality
        let latents_flat: Vec<f32> = latents
            .flatten_all()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let lat_mean = latents_flat.iter().sum::<f32>() / latents_flat.len() as f32;
        let lat_max = latents_flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lat_min = latents_flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let lat_std =
            (latents_flat.iter().map(|x| (x - lat_mean).powi(2)).sum::<f32>() / latents_flat.len() as f32).sqrt();
        eprintln!(
            "[PocketTTS] latent stats: mean={:.4}, std={:.4}, min={:.4}, max={:.4}",
            lat_mean, lat_std, lat_min, lat_max
        );

        // Denormalize latents before passing to Mimi
        // Python: mimi_decoding_input = latent * emb_std + emb_mean
        let latents = self
            .flowlm
            .denormalize_latents(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Denormalize: {}", e)))?;

        // Decode to audio using streaming mode with batch transformer
        // IMPORTANT: The decoder transformer uses NON-CAUSAL attention, so it must
        // process the full sequence at once (not frame-by-frame with KV cache)
        eprintln!("[PocketTTS] decoding with Mimi (batch transformer, streaming SEANet)...");
        let prof_mimi = std::time::Instant::now();
        let latents = latents
            .to_device(&self.mimi_device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio = self
            .mimi
            .forward_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;
        let audio = audio
            .to_device(&self.device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let prof_mimi_ms = prof_mimi.elapsed().as_secs_f64() * 1e3;
        eprintln!("[PocketTTS] audio tensor shape: {:?}", audio.dims());
        {
            use crate::prof;
            let samples = audio.dim(1).unwrap_or(0);
            eprintln!(
                "[PROF] frames={} audio_ms={:.0} total_ms={:.0} flowlm_ms={:.0} (lm_step={:.0} flownet={:.0}) mimi_ms={:.0} (upsample={:.0} transformer={:.0} seanet={:.0})",
                prof_frames,
                samples as f64 / 24.0,
                prof_start.elapsed().as_secs_f64() * 1e3,
                prof_lm_ms,
                prof::take_ms(&prof::FLOWLM_STEP),
                prof::take_ms(&prof::FLOWNET),
                prof_mimi_ms,
                prof::take_ms(&prof::MIMI_UPSAMPLE),
                prof::take_ms(&prof::MIMI_TRANSFORMER),
                prof::take_ms(&prof::MIMI_SEANET),
            );
        }

        // Note: Amplitude scaling is no longer needed after fixing SEANet to use batch mode
        // which produces correct amplitude (~0.4-0.5) matching Python's streaming output

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Debug: check amplitude to verify audio has signal
        let audio_max = audio_vec.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let audio_mean = audio_vec.iter().map(|s| s.abs()).sum::<f32>() / audio_vec.len() as f32;
        eprintln!(
            "[PocketTTS] final audio samples: {} (expect ~78720 for test phrase)",
            audio_vec.len()
        );
        eprintln!("[PocketTTS] audio max amplitude: {:.4} (expect > 0.01)", audio_max);
        eprintln!("[PocketTTS] audio mean amplitude: {:.4}", audio_mean);

        Ok(audio_vec)
    }

    /// True streaming synthesis - yields audio as soon as possible (PREFERRED METHOD)
    ///
    /// **This is the PREFERRED method for on-device TTS.** Latency is king - this
    /// method achieves ~200ms Time To First Audio (TTFA), delivering audio to the
    /// user as quickly as possible.
    ///
    /// Unlike `synthesize_streaming` which chunks by tokens, this method:
    /// 1. Tokenizes ALL text upfront
    /// 2. Generates latents one at a time (streaming FlowLM)
    /// 3. Decodes and yields audio in small batches DURING generation
    ///
    /// This achieves minimum TTFA by starting audio output as soon as
    /// the first batch of latents is generated.
    ///
    /// The callback receives:
    /// - `samples`: Audio samples for this chunk
    /// - `is_final`: Whether this is the last chunk
    ///
    /// Returns `false` from the callback to stop early.
    pub fn synthesize_true_streaming<F>(
        &mut self,
        text: &str,
        mut chunk_callback: F,
    ) -> std::result::Result<(), PocketTTSError>
    where
        F: FnMut(&[f32], bool) -> bool,
    {
        use std::cell::RefCell;

        // Tokenize ALL text upfront (not chunked)
        let token_ids = self.tokenizer.encode(text)?;

        // Create tensor for all tokens
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            Some(&self.voice)
        };

        // Pre-extract normalization tensors (to avoid borrowing self.flowlm in callback)
        let emb_mean = self.flowlm.emb_mean().clone();
        let emb_std = self.flowlm.emb_std().clone();

        // Reset caches for new sequence
        self.flowlm.reset_cache();
        self.mimi.reset_decoder_cache();

        // One flow step, as `synthesize` uses, so that both paths sample the
        // same latents from the same config.
        let num_flow_steps = 1;
        let seed = if self.config.use_fixed_seed {
            Some(self.config.seed as u64)
        } else {
            None
        };

        // How many latents to gather before decoding them (80 ms of audio
        // each). Measured on a 16 s sentence with two inference threads,
        // batches of 2, 4, 8, 12 and 25 all finished within 5% of each other
        // and of `synthesize`, so there is nothing to buy with a large batch
        // and the small one keeps the listener's lead even. The first batch is
        // separate because it is the one the listener waits for.
        // `POCKET_TTS_STREAM_FIRST` / `POCKET_TTS_STREAM_BATCH` override for measurement.
        let env_frames = |name: &str, default: usize| {
            std::env::var(name).ok().and_then(|s| s.parse().ok()).filter(|&n: &usize| n > 0).unwrap_or(default)
        };
        let first_batch_size = env_frames("POCKET_TTS_STREAM_FIRST", 4);
        let later_batch_size = env_frames("POCKET_TTS_STREAM_BATCH", 4);
        let batches_sent = RefCell::new(0usize);

        // Initialize persistent Mimi streaming state ONCE before streaming starts.
        // This state MUST persist across ALL decode batches for proper overlap-add.
        // Per Python reference: streaming ConvTranspose1d maintains partial buffers
        // that accumulate overlapping output contributions between frames.
        let mimi_state = self
            .mimi
            .init_streaming_state(1, &self.device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let mimi_state = RefCell::new(mimi_state);

        // SAFETY: We use raw pointer to access mimi from within the callback.
        // This is safe because:
        // 1. flowlm and mimi are separate fields with no shared state
        // 2. flowlm is borrowed mutably for generate_latents_streaming
        // 3. mimi is accessed mutably in the callback
        // 4. These are non-overlapping borrows of different struct fields
        // Rust's borrow checker can't prove this is safe, but we know it is.
        let mimi_ptr = &mut self.mimi as *mut MimiDecoder;

        // State for the callback
        let current_batch: RefCell<Vec<Tensor>> = RefCell::new(Vec::new());
        let should_continue = RefCell::new(true);
        let callback_error: RefCell<Option<PocketTTSError>> = RefCell::new(None);

        // NOTE: No crossfade needed - Mimi's streaming ConvTranspose1d state
        // handles overlap-add internally via partial buffers that persist
        // across frames. Adding crossfade would blend DIFFERENT audio content
        // (end of chunk A with start of chunk B) causing artifacts.

        // Generate latents with streaming callback that decodes INLINE
        let result = self.flowlm.generate_latents_streaming(
            &token_tensor,
            voice,
            num_flow_steps,
            self.config.temperature,
            seed,
            None, // noise_tensors: not used in production streaming
            |latent: &Tensor, _step: usize, is_eos: bool| {
                if !*should_continue.borrow() {
                    return LatentStreamControl::Stop;
                }

                // Add latent to current batch
                current_batch.borrow_mut().push(latent.clone());

                // Decode when the batch is full. EOS is not a reason to: the
                // generator runs `frames_after_eos` more steps, and what is
                // left when it stops is flushed below, marked final.
                let _ = is_eos;
                let wanted = if *batches_sent.borrow() == 0 { first_batch_size } else { later_batch_size };
                let batch_ready = current_batch.borrow().len() >= wanted;

                if batch_ready {
                    let batch: Vec<Tensor> = current_batch.borrow_mut().drain(..).collect();

                    // Concatenate batch latents: [1, n, 32]
                    let latents_batch = match Tensor::cat(&batch, 1) {
                        Ok(t) => t,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Denormalize latents before Mimi
                    let denormalized =
                        match latents_batch.broadcast_mul(&emb_std).and_then(|t| t.broadcast_add(&emb_mean)) {
                            Ok(d) => d,
                            Err(e) => {
                                *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                                *should_continue.borrow_mut() = false;
                                return LatentStreamControl::Stop;
                            },
                        };

                    // SAFETY: Access mimi through raw pointer - see safety comment above
                    let mimi = unsafe { &mut *mimi_ptr };

                    // Decode using TRUE streaming with persistent state
                    // This maintains ConvTranspose1d partial buffers across batches
                    let audio = match mimi.forward_streaming_stateful(&denormalized, &mut mimi_state.borrow_mut()) {
                        Ok(a) => a,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Extract samples - no crossfade needed, Mimi handles boundaries
                    let audio_vec: Vec<f32> = match audio.squeeze(0).and_then(|a| a.to_vec1()) {
                        Ok(v) => v,
                        Err(e) => {
                            *callback_error.borrow_mut() = Some(PocketTTSError::InferenceFailed(e.to_string()));
                            *should_continue.borrow_mut() = false;
                            return LatentStreamControl::Stop;
                        },
                    };

                    // Callback with audio chunk directly
                    *batches_sent.borrow_mut() += 1;
                    if !audio_vec.is_empty() && !chunk_callback(&audio_vec, false) {
                        *should_continue.borrow_mut() = false;
                        return LatentStreamControl::Stop;
                    }
                }

                // Always continue - let generate_latents_streaming handle EOS termination
                // with proper frames_after_eos logic (generates additional latents after EOS)
                // Returning Stop here would skip those additional latents!
                LatentStreamControl::Continue
            },
        );

        // Check for callback errors
        if let Some(err) = callback_error.into_inner() {
            return Err(err);
        }

        // Handle FlowLM errors
        result.map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM streaming: {}", e)))?;

        // The tail: whatever the generator produced after the last full batch,
        // including the `frames_after_eos` latents after EOS. The last chunk is
        // always sent, empty if need be, so the receiver hears `is_final`
        // exactly once.
        if !*should_continue.borrow() {
            return Ok(());
        }
        let rest: Vec<Tensor> = current_batch.borrow_mut().drain(..).collect();
        let mut tail: Vec<f32> = Vec::new();
        if !rest.is_empty() {
            let infer = |e: candle_core::Error| PocketTTSError::InferenceFailed(e.to_string());
            let latents = Tensor::cat(&rest, 1).map_err(infer)?;
            let latents = latents
                .broadcast_mul(&emb_std)
                .and_then(|t| t.broadcast_add(&emb_mean))
                .map_err(infer)?;
            let audio = self
                .mimi
                .forward_streaming_stateful(&latents, &mut mimi_state.borrow_mut())
                .map_err(infer)?;
            tail = audio.squeeze(0).and_then(|a| a.to_vec1()).map_err(infer)?;
        }
        chunk_callback(&tail, true);

        Ok(())
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.mimi.sample_rate() as u32
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> u64 {
        117_856_642 // From model manifest
    }

    /// Get model version
    pub fn version(&self) -> &str {
        "1.0.2"
    }

    /// Decode raw latents to audio (for reference testing)
    ///
    /// Takes raw f32 latent data (already denormalized) and decodes it through Mimi.
    /// The latents should be [num_frames, 32] f32 values.
    ///
    /// Note: The latents are NOT denormalized here - they should already be
    /// in the denormalized form (matching Python's mimi_decoding_input).
    pub fn decode_latents(
        &mut self,
        latents_f32: &[f32],
        num_frames: usize,
    ) -> std::result::Result<Vec<f32>, PocketTTSError> {
        let latent_dim = 32;
        let expected_len = num_frames * latent_dim;

        if latents_f32.len() != expected_len {
            return Err(PocketTTSError::InferenceFailed(format!(
                "Expected {} floats for {} frames, got {}",
                expected_len,
                num_frames,
                latents_f32.len()
            )));
        }

        // Create tensor [1, num_frames, 32]
        let latents = Tensor::from_vec(latents_f32.to_vec(), (1, num_frames, latent_dim), &self.device)
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] decode_latents: input shape {:?}", latents.dims());

        // Decode with Mimi (batch transformer, streaming SEANet)
        let audio = self
            .mimi
            .forward_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi decode: {}", e)))?;

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] decode_latents: output {} samples", audio_vec.len());

        Ok(audio_vec)
    }

    /// Synthesize text to audio, also returning raw latents for debugging
    pub fn synthesize_with_latents(
        &mut self,
        text: &str,
    ) -> std::result::Result<(Vec<f32>, Vec<f32>, [usize; 3]), PocketTTSError> {
        eprintln!("[PocketTTS] synthesize_with_latents called with text len: {}", text.len());

        // Tokenize text
        let token_ids = self.tokenizer.encode(text)?;
        eprintln!("[PocketTTS] tokenized to {} tokens: {:?}", token_ids.len(), token_ids);

        // Create tensor
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        // Get voice embedding
        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            Some(&self.voice)
        };

        // Reset caches for new sequence
        self.flowlm.reset_cache();

        // Generate latents with FlowLM + FlowNet
        let num_flow_steps = 1;
        let seed = if self.config.use_fixed_seed {
            Some(self.config.seed as u64)
        } else {
            None
        };
        let latents = self
            .flowlm
            .generate_latents(&token_tensor, voice, num_flow_steps, self.config.temperature, seed, None)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;

        // Get latent shape and data
        let latent_dims = latents.dims();
        let latent_shape = [latent_dims[0], latent_dims[1], latent_dims[2]];
        let latents_flat: Vec<f32> = latents
            .flatten_all()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?
            .to_vec1()
            .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        eprintln!("[PocketTTS] latents shape: {:?}", latent_shape);

        // Denormalize latents before passing to Mimi
        let latents = self
            .flowlm
            .denormalize_latents(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Denormalize: {}", e)))?;

        // Decode to audio using TRUE streaming mode for correct amplitude
        let audio = self
            .mimi
            .forward_true_streaming(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;

        // Convert to Vec<f32>
        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        Ok((audio_vec, latents_flat, latent_shape))
    }

    /// Synthesize audio and dump Mimi decoder intermediates for diagnostic comparison.
    ///
    /// Same pipeline as `synthesize` but uses `forward_streaming_with_dump` to save
    /// per-block .npy files for the first N frames.
    pub fn synthesize_with_mimi_dump(
        &mut self,
        text: &str,
        dump_dir: &std::path::Path,
        dump_frames: usize,
    ) -> std::result::Result<Vec<f32>, PocketTTSError> {
        eprintln!("[PocketTTS] synthesize_with_mimi_dump");

        let token_ids = self.tokenizer.encode(text)?;
        let token_tensor = Tensor::from_vec(
            token_ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
            (1, token_ids.len()),
            &self.device,
        )
        .map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        let voice = if let Some(ref custom) = self.custom_voice {
            Some(custom)
        } else {
            Some(&self.voice)
        };

        self.flowlm.reset_cache();

        let num_flow_steps = 1;
        let seed = if self.config.use_fixed_seed {
            Some(self.config.seed as u64)
        } else {
            None
        };
        let noise_ref = self.noise_tensors.as_deref();
        let latents = self
            .flowlm
            .generate_latents(&token_tensor, voice, num_flow_steps, self.config.temperature, seed, noise_ref)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("FlowLM: {}", e)))?;

        // Denormalize
        let latents = self
            .flowlm
            .denormalize_latents(&latents)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Denormalize: {}", e)))?;

        // Create dump directory
        std::fs::create_dir_all(dump_dir).map_err(|e| PocketTTSError::InferenceFailed(format!("mkdir: {}", e)))?;

        // Decode with dump
        let audio = self
            .mimi
            .forward_streaming_with_dump(&latents, dump_dir, dump_frames)
            .map_err(|e| PocketTTSError::InferenceFailed(format!("Mimi: {}", e)))?;

        let audio = audio.squeeze(0).map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;
        let audio_vec: Vec<f32> = audio.to_vec1().map_err(|e| PocketTTSError::InferenceFailed(e.to_string()))?;

        Ok(audio_vec)
    }
}

impl std::fmt::Debug for PocketTTSModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PocketTTSModel")
            .field("version", &self.version())
            .field("parameter_count", &self.parameter_count())
            .field("sample_rate", &self.sample_rate())
            .field("voice_count", &self.voice_bank.len())
            .finish()
    }
}
