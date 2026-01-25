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
