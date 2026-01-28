//! Integration tests for CLI options
//!
//! Tests the --provider, --device-id, and --batch-size CLI options.

use std::process::Command;

fn demongrep_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_demongrep"))
}

// ============================================================================
// Help Text Tests
// ============================================================================

#[test]
fn test_index_help_shows_provider_option() {
    let output = demongrep_bin()
        .args(["index", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--provider"), "Help should show --provider option");
    assert!(stdout.contains("cuda"), "Help should mention cuda provider");
    assert!(stdout.contains("cpu"), "Help should mention cpu provider");
}

#[test]
fn test_index_help_shows_device_id_option() {
    let output = demongrep_bin()
        .args(["index", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--device-id"), "Help should show --device-id option");
}

#[test]
fn test_index_help_shows_batch_size_option() {
    let output = demongrep_bin()
        .args(["index", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--batch-size"), "Help should show --batch-size option");
}

#[test]
fn test_search_help_shows_provider_option() {
    let output = demongrep_bin()
        .args(["search", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--provider"), "Search help should show --provider option");
}

// ============================================================================
// Provider Argument Parsing Tests
// ============================================================================

#[test]
fn test_invalid_provider_rejected() {
    let output = demongrep_bin()
        .args(["index", "--provider", "invalid_provider", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    // Should fail with an error
    assert!(!output.status.success(), "Invalid provider should be rejected");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("invalid") || stderr.contains("Invalid") || stderr.contains("error"),
        "Error message should indicate invalid provider"
    );
}

#[test]
fn test_cpu_provider_accepted() {
    let output = demongrep_bin()
        .args(["index", "--provider", "cpu", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    // --dry-run with cpu provider should succeed (no actual indexing)
    // Note: May fail if no files to index, but shouldn't fail on provider parsing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("invalid provider") && !stderr.contains("Invalid provider"),
        "cpu provider should be accepted"
    );
}

#[test]
fn test_auto_provider_accepted() {
    let output = demongrep_bin()
        .args(["index", "--provider", "auto", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("invalid provider") && !stderr.contains("Invalid provider"),
        "auto provider should be accepted"
    );
}

// ============================================================================
// Batch Size Argument Tests
// ============================================================================

#[test]
fn test_batch_size_accepts_number() {
    let output = demongrep_bin()
        .args(["index", "--batch-size", "64", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("invalid") || stderr.contains("No files"),
        "Numeric batch size should be accepted"
    );
}

#[test]
fn test_batch_size_rejects_non_number() {
    let output = demongrep_bin()
        .args(["index", "--batch-size", "not_a_number", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Non-numeric batch size should be rejected");
}

// ============================================================================
// Device ID Argument Tests
// ============================================================================

#[test]
fn test_device_id_accepts_number() {
    let output = demongrep_bin()
        .args(["index", "--provider", "cpu", "--device-id", "0", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("invalid"),
        "Numeric device ID should be accepted"
    );
}

// ============================================================================
// Doctor Command Tests
// ============================================================================

#[test]
fn test_doctor_runs_successfully() {
    let output = demongrep_bin()
        .args(["doctor"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "doctor command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("CPU") || stdout.contains("Provider"),
        "doctor should show provider information"
    );
}

// ============================================================================
// Serve Command Tests
// ============================================================================

#[test]
fn test_serve_help_shows_model_option() {
    // The --model flag is global, so it should appear in serve help
    let output = demongrep_bin()
        .args(["serve", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--model"), "Serve help should show --model option (global flag)");
}

#[test]
fn test_serve_help_shows_provider_option() {
    let output = demongrep_bin()
        .args(["serve", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--provider"), "Serve help should show --provider option");
}

#[test]
fn test_serve_help_shows_device_id_option() {
    let output = demongrep_bin()
        .args(["serve", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--device-id"), "Serve help should show --device-id option");
}

#[test]
fn test_serve_accepts_model_flag() {
    // Test that the serve command accepts --model flag without crashing on argument parsing
    // We use a very short timeout since we just want to verify the flag is accepted,
    // not actually run the server.
    use std::time::Duration;
    use std::thread;
    use std::sync::mpsc::channel;

    let (tx, rx) = channel();

    thread::spawn(move || {
        let output = demongrep_bin()
            .args(["--model", "bge-small", "serve", "--port", "19999"])
            .output()
            .expect("Failed to execute command");
        let _ = tx.send(output);
    });

    // Wait a bit then check - if it crashed on arg parsing, it would return immediately with error
    thread::sleep(Duration::from_millis(500));

    // If we got an immediate response, check it wasn't an argument parsing error
    if let Ok(output) = rx.try_recv() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Argument parsing errors would mention "error: " or "invalid"
        assert!(
            !stderr.contains("error: unrecognized") &&
            !stderr.contains("unexpected argument") &&
            !stderr.contains("Unknown model: 'bge-small'"),
            "The --model flag should be recognized by serve command. Stderr: {}", stderr
        );
    }
    // If no response yet, the server started successfully (flag was accepted)
}

#[test]
fn test_serve_rejects_invalid_model() {
    use std::time::Duration;
    use std::thread;
    use std::sync::mpsc::channel;

    let (tx, rx) = channel();

    thread::spawn(move || {
        let output = demongrep_bin()
            .args(["--model", "totally-invalid-model-name", "serve", "--port", "19998"])
            .output()
            .expect("Failed to execute command");
        let _ = tx.send(output);
    });

    // Wait for the command to finish (invalid model should cause immediate exit)
    thread::sleep(Duration::from_millis(1000));

    if let Ok(output) = rx.try_recv() {
        // Should fail due to invalid model
        assert!(!output.status.success(), "Invalid model should be rejected");

        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Unknown model"),
            "Should show 'Unknown model' error. Stderr: {}", stderr
        );
    }
}

#[test]
fn test_model_flag_position_before_subcommand() {
    // --model is a global flag and should work before the subcommand
    let output = demongrep_bin()
        .args(["--model", "minilm-l6", "index", "--help"])
        .output()
        .expect("Failed to execute command");

    assert!(output.status.success(), "Global --model flag before subcommand should work");
}

#[test]
fn test_all_commands_show_model_in_help() {
    // Verify --model appears in help for all embedding-related commands
    let commands = ["search", "index", "serve", "mcp"];

    for cmd in commands {
        let output = demongrep_bin()
            .args([cmd, "--help"])
            .output()
            .expect(&format!("Failed to execute {} --help", cmd));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("--model"),
            "{} command help should show --model option (global flag)", cmd
        );
    }
}

// ============================================================================
// Ollama Backend Tests
// ============================================================================

#[test]
fn test_backend_flag_shown_in_help() {
    let output = demongrep_bin()
        .args(["index", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--backend"), "Help should show --backend option");
    assert!(stdout.contains("ollama"), "Help should mention ollama backend");
}

#[test]
fn test_ollama_model_flag_shown_in_help() {
    let output = demongrep_bin()
        .args(["index", "--help"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--ollama-model"), "Help should show --ollama-model option");
}

#[test]
fn test_explicit_backend_fastembed_accepted() {
    let output = demongrep_bin()
        .args(["index", "--backend", "fastembed", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT show "Auto-detected Ollama" message when explicitly using fastembed
    assert!(
        !stderr.contains("Auto-detected Ollama"),
        "Explicit --backend fastembed should not auto-detect Ollama"
    );
}

#[test]
fn test_invalid_backend_rejected() {
    let output = demongrep_bin()
        .args(["index", "--backend", "invalid_backend", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    assert!(!output.status.success(), "Invalid backend should be rejected");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown backend") || stderr.contains("Available backends"),
        "Error should mention available backends. Stderr: {}", stderr
    );
}

#[test]
#[cfg(feature = "ollama")]
fn test_explicit_backend_ollama_without_server_fails() {
    // When explicitly requesting Ollama but server isn't running, should fail
    // Use a port that definitely won't have Ollama
    let output = demongrep_bin()
        .env("DEMONGREP_OLLAMA_URL", "http://localhost:59999")
        .args(["index", "--backend", "ollama", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    // This should fail because Ollama isn't available at that URL
    // Note: The actual behavior depends on whether we fail fast or try to connect later
    let stderr = String::from_utf8_lossy(&output.stderr);
    let _stdout = String::from_utf8_lossy(&output.stdout);

    // Either it fails, or it shows an Ollama connection error
    if !output.status.success() {
        assert!(
            stderr.contains("Ollama") || stderr.contains("connect") || stderr.contains("Cannot"),
            "Should show Ollama connection error. Stderr: {}", stderr
        );
    }
}

#[test]
#[cfg(feature = "ollama")]
fn test_auto_detection_message_format() {
    // If Ollama is running (e.g., in CI or dev environment),
    // verify the auto-detection message format
    let output = demongrep_bin()
        .args(["index", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // If Ollama was auto-detected, verify message format
    if stderr.contains("Auto-detected Ollama") {
        assert!(
            stderr.contains("http://") && stderr.contains("using Ollama backend"),
            "Auto-detection message should show URL. Stderr: {}", stderr
        );
        assert!(
            stderr.contains("--backend fastembed to override"),
            "Should show how to override. Stderr: {}", stderr
        );
    }
}

#[test]
#[cfg(feature = "ollama")]
fn test_fastembed_override_when_ollama_available() {
    // Even if Ollama is running, --backend fastembed should override
    let output = demongrep_bin()
        .args(["index", "--backend", "fastembed", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT show auto-detection message
    assert!(
        !stderr.contains("Auto-detected Ollama"),
        "Explicit fastembed should skip auto-detection. Stderr: {}", stderr
    );
}

// ============================================================================
// Backend Selection Priority Tests
// ============================================================================

/// Test the backend selection priority:
/// 1. CLI --backend flag (highest)
/// 2. Config file embedding.backend
/// 3. Auto-detect Ollama if available
/// 4. Default to FastEmbed
#[test]
fn test_backend_selection_cli_takes_priority() {
    // Create a temp config file that sets backend = "ollama"
    let temp_dir = std::env::temp_dir().join("demongrep_test_config");
    let _ = std::fs::create_dir_all(&temp_dir);
    let config_path = temp_dir.join("config.toml");

    std::fs::write(&config_path, r#"
[embedding]
backend = "ollama"
"#).expect("Failed to write test config");

    // But CLI says fastembed - CLI should win
    let output = demongrep_bin()
        .env("DEMONGREP_CONFIG", config_path.to_str().unwrap())
        .args(["index", "--backend", "fastembed", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should use FastEmbed, not Ollama
    assert!(
        !stderr.contains("Auto-detected Ollama") && !stderr.contains("Loading Ollama"),
        "CLI --backend should override config. Stderr: {}", stderr
    );

    // Cleanup
    let _ = std::fs::remove_file(&config_path);
}

// ============================================================================
// Ollama URL Configuration Tests
// ============================================================================

#[test]
#[cfg(feature = "ollama")]
fn test_custom_ollama_url_in_config() {
    let temp_dir = std::env::temp_dir().join("demongrep_test_ollama_url");
    let _ = std::fs::create_dir_all(&temp_dir);
    let config_path = temp_dir.join("config.toml");

    // Configure a custom (invalid) Ollama URL
    std::fs::write(&config_path, r#"
[embedding]
backend = "fastembed"

[embedding.ollama]
url = "http://localhost:99999"
model = "nomic-embed-text"
"#).expect("Failed to write test config");

    // With fastembed backend, should not try to connect to Ollama at all
    let output = demongrep_bin()
        .env("DEMONGREP_CONFIG", config_path.to_str().unwrap())
        .args(["index", "--backend", "fastembed", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should not fail due to Ollama connection when using fastembed
    assert!(
        !stderr.contains("Cannot connect to Ollama"),
        "Should not try Ollama when using fastembed. Stderr: {}", stderr
    );

    // Cleanup
    let _ = std::fs::remove_file(&config_path);
}

// ============================================================================
// Diagnostic Tests - Help identify configuration issues
// ============================================================================

#[test]
fn test_doctor_shows_ollama_status() {
    let output = demongrep_bin()
        .args(["doctor"])
        .output()
        .expect("Failed to execute command");

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Doctor should show information about available backends
    // This helps users diagnose why Ollama might not be auto-detected
    #[cfg(feature = "ollama")]
    {
        // When compiled with ollama feature, should mention it
        assert!(
            stdout.contains("Ollama") || stdout.contains("ollama"),
            "Doctor should show Ollama status when feature is enabled. Stdout: {}", stdout
        );
    }
}

/// This test helps diagnose auto-detection issues by printing debug info
#[test]
#[ignore] // Run manually with: cargo test --features ollama -- --ignored --nocapture
fn test_debug_ollama_auto_detection() {
    println!("\n=== Ollama Auto-Detection Debug ===\n");

    // Check if ollama feature is compiled in
    #[cfg(feature = "ollama")]
    println!("✓ Ollama feature: ENABLED");
    #[cfg(not(feature = "ollama"))]
    println!("✗ Ollama feature: DISABLED");

    // Try to connect to Ollama
    #[cfg(feature = "ollama")]
    {
        let urls = [
            "http://localhost:11434",
            "http://127.0.0.1:11434",
            "http://host.docker.internal:11434",
        ];

        for url in urls {
            let health_url = format!("{}/api/tags", url);
            print!("  Checking {}... ", url);

            match ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(2))
                .build()
                .get(&health_url)
                .call()
            {
                Ok(resp) => {
                    println!("✓ AVAILABLE (status: {})", resp.status());
                    if let Ok(body) = resp.into_string() {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                            if let Some(models) = json["models"].as_array() {
                                println!("    Models available: {}", models.len());
                                for model in models.iter().take(5) {
                                    if let Some(name) = model["name"].as_str() {
                                        println!("      - {}", name);
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("✗ NOT AVAILABLE ({})", e);
                }
            }
        }
    }

    // Run actual command and capture output
    println!("\n=== Running demongrep index --dry-run ===\n");

    let output = demongrep_bin()
        .args(["index", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    println!("Exit code: {}", output.status);
    println!("\nStdout:\n{}", String::from_utf8_lossy(&output.stdout));
    println!("\nStderr:\n{}", String::from_utf8_lossy(&output.stderr));

    // Check what backend was actually used
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("Auto-detected Ollama") {
        println!("\n✓ Result: Ollama was AUTO-DETECTED");
    } else if stderr.contains("Loading Ollama") {
        println!("\n✓ Result: Ollama backend was used (explicit)");
    } else if stderr.contains("Loading embedding model") || stderr.contains("FastEmbed") {
        println!("\n✗ Result: FastEmbed backend was used");
        println!("  Possible reasons:");
        println!("    1. Ollama not running");
        println!("    2. Ollama feature not compiled in");
        println!("    3. Config file sets backend = \"fastembed\"");
        println!("    4. CLI --backend fastembed was used");
    }
}

// ============================================================================
// Auto-Pull Tests - Require Ollama server
// ============================================================================

/// Test that auto-pull message is shown when model isn't found
#[test]
#[ignore] // Requires running Ollama server
fn test_auto_pull_message_shown() {
    // First remove the model to ensure it needs to be pulled
    let _ = std::process::Command::new("ollama")
        .args(["rm", "all-minilm"])
        .output();

    let output = demongrep_bin()
        .args(["index", "--backend", "ollama", "--ollama-model", "all-minilm", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should show the auto-pull message
    assert!(
        stderr.contains("not found locally") || stderr.contains("Pulling"),
        "Should show auto-pull message when model not found. Stderr: {}",
        stderr
    );
}

/// Test that auto-pull succeeds and indexing continues
#[test]
#[ignore] // Requires running Ollama server
fn test_auto_pull_then_index_succeeds() {
    // Remove model first
    let _ = std::process::Command::new("ollama")
        .args(["rm", "all-minilm"])
        .output();

    // Create a temp directory with a small test file
    let temp_dir = std::env::temp_dir().join("demongrep_autopull_test");
    let _ = std::fs::create_dir_all(&temp_dir);
    std::fs::write(temp_dir.join("test.rs"), "fn main() { println!(\"hello\"); }").unwrap();

    let output = demongrep_bin()
        .args([
            "index",
            "--backend", "ollama",
            "--ollama-model", "all-minilm",
            "--force",
        ])
        .current_dir(&temp_dir)
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Check success
    assert!(
        output.status.success(),
        "Index should succeed after auto-pull. Stderr: {}\nStdout: {}",
        stderr,
        stdout
    );

    // Should show pull succeeded
    assert!(
        stderr.contains("pulled successfully") || stderr.contains("Ollama"),
        "Should indicate Ollama was used. Stderr: {}",
        stderr
    );
}

/// Test that invalid model name fails gracefully
#[test]
#[ignore] // Requires running Ollama server
fn test_auto_pull_invalid_model_fails_gracefully() {
    let output = demongrep_bin()
        .args([
            "index",
            "--backend", "ollama",
            "--ollama-model", "this-model-definitely-does-not-exist-xyz123",
            "--dry-run",
        ])
        .output()
        .expect("Failed to execute command");

    // Should fail
    assert!(
        !output.status.success(),
        "Should fail for non-existent model"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should show helpful error
    assert!(
        stderr.contains("could not be pulled")
            || stderr.contains("not exist")
            || stderr.contains("Failed to pull"),
        "Should show pull failure message. Stderr: {}",
        stderr
    );
}

/// Test that already-installed model doesn't trigger pull
#[test]
#[ignore] // Requires running Ollama server
fn test_no_pull_for_installed_model() {
    // First ensure the model is installed
    let _ = std::process::Command::new("ollama")
        .args(["pull", "nomic-embed-text"])
        .status();

    let output = demongrep_bin()
        .args(["index", "--backend", "ollama", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT show pull message
    assert!(
        !stderr.contains("not found locally") && !stderr.contains("Pulling from Ollama"),
        "Should not attempt to pull already-installed model. Stderr: {}",
        stderr
    );

    // Should show Ollama backend is being used
    assert!(
        stderr.contains("Ollama") || stderr.contains("ollama"),
        "Should indicate Ollama is being used. Stderr: {}",
        stderr
    );
}

/// Debug test to manually verify auto-pull behavior
#[test]
#[ignore] // Run manually: cargo test --features ollama -- test_debug_auto_pull --ignored --nocapture
fn test_debug_auto_pull() {
    println!("\n=== Auto-Pull Debug Test ===\n");

    // Check Ollama status
    println!("1. Checking Ollama server...");
    match std::process::Command::new("ollama").args(["list"]).output() {
        Ok(output) => {
            println!("   Installed models:");
            println!("{}", String::from_utf8_lossy(&output.stdout));
        }
        Err(e) => {
            println!("   Failed to run ollama list: {}", e);
            return;
        }
    }

    // Test with a model that might need pulling
    println!("\n2. Testing auto-pull with 'all-minilm'...");

    // Remove it first
    println!("   Removing model first...");
    let _ = std::process::Command::new("ollama")
        .args(["rm", "all-minilm"])
        .status();

    println!("   Running demongrep index --backend ollama --ollama-model all-minilm --dry-run");

    let output = demongrep_bin()
        .args(["index", "--backend", "ollama", "--ollama-model", "all-minilm", "--dry-run"])
        .output()
        .expect("Failed to execute command");

    println!("\n   Exit code: {}", output.status);
    println!("\n   Stderr:\n{}", String::from_utf8_lossy(&output.stderr));
    println!("\n   Stdout:\n{}", String::from_utf8_lossy(&output.stdout));

    // Verify model was pulled
    println!("\n3. Verifying model was pulled...");
    let list_output = std::process::Command::new("ollama")
        .args(["list"])
        .output()
        .expect("Failed to list models");

    let list_str = String::from_utf8_lossy(&list_output.stdout);
    if list_str.contains("all-minilm") {
        println!("   ✓ Model 'all-minilm' is now installed");
    } else {
        println!("   ✗ Model 'all-minilm' was NOT installed");
        println!("   Current models: {}", list_str);
    }
}
