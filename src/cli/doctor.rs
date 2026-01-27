use anyhow::Result;
use crate::embed::detect_best_provider;

pub async fn run() -> Result<()> {
    println!("demongrep system check\n");

    // Model checks
    check_default_model()?;

    // GPU checks
    println!("\nGPU Backends:\n");
    check_cuda();
    check_tensorrt();
    check_coreml();
    check_directml();

    // Ollama check
    println!("\nOllama Backend:\n");
    check_ollama();

    // Recommended provider
    let recommended = detect_best_provider();
    println!("\nRecommended: {:?}", recommended);

    Ok(())
}

fn check_default_model() -> Result<()> {
    use crate::embed::{ModelType, FastEmbedder};
    
    let default_model = ModelType::default();
    println!("Default model: {} ({} dimensions)", default_model.name(), default_model.dimensions());
    
    // Test loading the model
    match FastEmbedder::with_model(default_model) {
        Ok(_) => println!("✅ Default model loaded successfully"),
        Err(e) => println!("❌ Default model failed to load: {}", e),
    }
    
    Ok(())
}

#[cfg(feature = "cuda")]
fn check_cuda() {
    use crate::embed::is_cuda_available;
    match is_cuda_available() {
        true => println!("✅ CUDA: Available"),
        false => println!("❌ CUDA: Not available"),
    }
}

#[cfg(not(feature = "cuda"))]
fn check_cuda() {
    println!("❌ CUDA: Not compiled in (enable with --features cuda)");
}

#[cfg(feature = "tensorrt")]
fn check_tensorrt() {
    use crate::embed::is_tensorrt_available;
    match is_tensorrt_available() {
        true => println!("✅ TensorRT: Available"),
        false => println!("❌ TensorRT: Not available"),
    }
}

#[cfg(not(feature = "tensorrt"))]
fn check_tensorrt() {
    println!("❌ TensorRT: Not compiled in (enable with --features tensorrt)");
}

#[cfg(feature = "coreml")]
fn check_coreml() {
    use crate::embed::is_coreml_available;
    match is_coreml_available() {
        true => println!("✅ CoreML: Available"),
        false => println!("❌ CoreML: Not available"),
    }
}

#[cfg(not(feature = "coreml"))]
fn check_coreml() {
    println!("❌ CoreML: Not compiled in (enable with --features coreml)");
}

#[cfg(feature = "directml")]
fn check_directml() {
    use crate::embed::is_directml_available;
    match is_directml_available() {
        true => println!("✅ DirectML: Available"),
        false => println!("❌ DirectML: Not available"),
    }
}

#[cfg(not(feature = "directml"))]
fn check_directml() {
    println!("❌ DirectML: Not compiled in (enable with --features directml)");
}

#[cfg(feature = "ollama")]
fn check_ollama() {
    use crate::config::Config;

    // Load config to get Ollama URL
    let config = Config::load().unwrap_or_default();
    let ollama_url = &config.embedding.ollama.url;

    println!("✅ Ollama: Compiled in (feature enabled)");
    println!("   Configured URL: {}", ollama_url);
    println!("   Configured model: {}", config.embedding.ollama.model);

    // Check if Ollama is running
    let health_url = format!("{}/api/tags", ollama_url);
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(2))
        .build();

    match agent.get(&health_url).call() {
        Ok(resp) => {
            println!("✅ Ollama server: Running at {}", ollama_url);

            // Parse and show available models
            if let Ok(body) = resp.into_string() {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(models) = json["models"].as_array() {
                        let model_names: Vec<&str> = models
                            .iter()
                            .filter_map(|m| m["name"].as_str())
                            .collect();
                        if !model_names.is_empty() {
                            println!("   Available models: {}", model_names.join(", "));

                            // Check if configured model is available
                            let configured = &config.embedding.ollama.model;
                            let model_available = model_names.iter().any(|n| n.starts_with(configured));
                            if model_available {
                                println!("   ✅ Configured model '{}' is available", configured);
                            } else {
                                println!("   ⚠️  Configured model '{}' not found. Install with: ollama pull {}", configured, configured);
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("❌ Ollama server: Not running or unreachable at {}", ollama_url);
            println!("   Error: {}", e);
            println!("   Start with: ollama serve");
        }
    }
}

#[cfg(not(feature = "ollama"))]
fn check_ollama() {
    println!("❌ Ollama: Not compiled in (enable with --features ollama)");
}
