use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const PYOXIDIZER_CONFIG: &str = r#"
# pyoxidizer.bzl
def make_exe():
    dist = default_python_distribution()
    policy = dist.make_python_packaging_policy()
    policy.extension_module_filter = "all"
    policy.resources_location = "filesystem-relative:prefix"

    python_config = dist.make_python_interpreter_config()
    python_config.run_module = "main"

    exe = dist.to_python_executable(
        name = "hexai",
        packaging_policy = policy,
        config = python_config,
    )
    exe.add_python_resources(exe.pip_install(["-r", "requirements.txt"]))
    return exe

def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files

register_target("exe", make_exe)
register_target("install", make_install, depends=["exe"], default=True)
resolve_targets()
"#;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 HexAi Builder – tworzy samodzielny plik wykonywalny z aplikacji Python");

    // 1. Upewnij się, że mamy requirements.txt
    let source_dir = PathBuf::from("./source-code");
    if !source_dir.exists() {
        eprintln!("❌ Katalog 'source-code' nie istnieje! Umieść w nim plik main.py.");
        return Err("Brak katalogu source-code".into());
    }
    let main_py = source_dir.join("main.py");
    if !main_py.exists() {
        eprintln!("❌ Nie znaleziono pliku main.py w katalogu source-code");
        return Err("Brak main.py".into());
    }

    // 2. Przygotuj plik requirements.txt (jeśli nie istnieje)
    let requirements = source_dir.join("requirements.txt");
    if !requirements.exists() {
        println!("📦 Tworzę requirements.txt na podstawie importów...");
        // Proste wykrywanie importów (można rozbudować)
        let content = fs::read_to_string(&main_py)?;
        let imports = vec![
            "fastapi", "uvicorn", "transformers", "accelerate", "bitsandbytes",
            "torch", "ollama", "llama-index", "chromadb", "faster-whisper",
            "diffusers", "duckduckgo-search", "pypdf", "beautifulsoup4",
        ];
        let mut req = String::new();
        for imp in imports {
            if content.contains(&format!("import {}", imp)) || content.contains(&format!("from {}", imp)) {
                req.push_str(&format!("{}\n", imp));
            }
        }
        fs::write(&requirements, req)?;
    }

    // 3. Sprawdź, czy PyOxidizer jest dostępny
    println!("🔍 Sprawdzam obecność PyOxidizer...");
    let pyoxidizer_check = Command::new("pyoxidizer")
        .arg("--version")
        .output();
    if pyoxidizer_check.is_err() {
        println!("📦 PyOxidizer nie znaleziony – instaluję przez pip...");
        let status = Command::new("pip")
            .arg("install")
            .arg("pyoxidizer")
            .status()?;
        if !status.success() {
            eprintln!("❌ Nie udało się zainstalować PyOxidizer. Zainstaluj ręcznie: pip install pyoxidizer");
            return Err("Brak PyOxidizer".into());
        }
    }

    // 4. Skopiuj plik konfiguracyjny do katalogu źródłowego
    let config_path = source_dir.join("pyoxidizer.bzl");
    fs::write(&config_path, PYOXIDIZER_CONFIG)?;
    println!("📝 Utworzono konfigurację: {}", config_path.display());

    // 5. Uruchom PyOxidizer w katalogu source-code
    println!("🏗️  Budowanie aplikacji...");
    let output = Command::new("pyoxidizer")
        .current_dir(&source_dir)
        .arg("build")
        .output()?;

    if !output.status.success() {
        eprintln!("❌ Budowanie nie powiodło się:");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return Err("Błąd budowania".into());
    }

    // 6. Znajdź wynikowy plik binarny
    // PyOxidizer tworzy plik w: build/.../install/hexai (lub hexai.exe)
    let build_dir = source_dir.join("build");
    let target_exe = find_executable(&build_dir, "hexai")?;
    let dest = PathBuf::from(".").join(target_exe.file_name().unwrap());

    // 7. Skopiuj do bieżącego katalogu
    fs::copy(&target_exe, &dest)?;
    println!("✅ Gotowe! Plik wykonywalny: {}", dest.display());

    // 8. Wyczyść (opcjonalnie)
    // fs::remove_file(config_path)?;

    Ok(())
}

/// Rekurencyjnie szuka pliku o nazwie 'hexai' (lub hexai.exe) w katalogu build.
fn find_executable(start_dir: &PathBuf, name: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let target_name = if cfg!(windows) { "hexai.exe" } else { name };
    for entry in walkdir::WalkDir::new(start_dir) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.file_name() == target_name {
            return Ok(entry.into_path());
        }
    }
    Err("Nie znaleziono pliku wykonywalnego".into())
}

// Dodajemy zależność walkdir (dodaj do Cargo.toml)
// Użycie: cargo add walkdir
