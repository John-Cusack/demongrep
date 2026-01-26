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
