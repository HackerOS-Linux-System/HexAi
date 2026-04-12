// HexAi Builder – packages the Python backend into a standalone binary via PyOxidizer
// Usage:  cargo run --release
//         cargo run --release -- --source ./source-code --out ./dist

use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::Command,
};

use walkdir::WalkDir;

// ─────────────────────────── PyOxidizer config ───────────────────────────

const PYOXIDIZER_CONFIG: &str = r#"def make_exe():
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

// ─────────────────────────── CLI args ───────────────────────────

struct Config {
    source_dir: PathBuf,
    out_dir: PathBuf,
    skip_cleanup: bool,
}

impl Config {
    fn from_args() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut source_dir = PathBuf::from("./source-code");
        let mut out_dir = PathBuf::from(".");
        let mut skip_cleanup = false;

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--source" | "-s" => {
                    i += 1;
                    if i < args.len() {
                        source_dir = PathBuf::from(&args[i]);
                    }
                }
                "--out" | "-o" => {
                    i += 1;
                    if i < args.len() {
                        out_dir = PathBuf::from(&args[i]);
                    }
                }
                "--no-cleanup" => skip_cleanup = true,
                "--help" | "-h" => {
                    eprintln!("HexAi Builder");
                    eprintln!("  --source <dir>   Katalog z kodem Python (default: ./source-code)");
                    eprintln!("  --out    <dir>   Katalog docelowy (default: .)");
                    eprintln!("  --no-cleanup     Nie usuwaj plików tymczasowych");
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }
        Self { source_dir, out_dir, skip_cleanup }
    }
}

// ─────────────────────────── Helpers ───────────────────────────

fn print_step(step: u8, total: u8, msg: &str) {
    println!("\n[{step}/{total}] {msg}");
    io::stdout().flush().ok();
}

fn run_cmd(cmd: &str, args: &[&str], cwd: Option<&PathBuf>) -> Result<(), String> {
    let mut builder = Command::new(cmd);
    builder.args(args);
    if let Some(dir) = cwd {
        builder.current_dir(dir);
    }
    let status = builder.status().map_err(|e| format!("Nie można uruchomić `{cmd}`: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("`{cmd}` zakończyło się błędem (kod: {:?})", status.code()))
    }
}

fn find_executable(start: &PathBuf) -> Option<PathBuf> {
    let name = if cfg!(windows) { "hexai.exe" } else { "hexai" };
    for entry in WalkDir::new(start).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() && entry.file_name() == name {
            return Some(entry.into_path());
        }
    }
    None
}

fn auto_detect_requirements(main_py: &PathBuf) -> String {
    let content = fs::read_to_string(main_py).unwrap_or_default();
    let known = [
        "fastapi", "uvicorn", "transformers", "accelerate", "bitsandbytes",
        "torch", "ollama", "llama_index", "chromadb", "rank_bm25",
        "sentence_transformers", "diffusers", "duckduckgo_search", "pypdf",
        "beautifulsoup4", "docker", "redis", "soundfile", "rich",
        "pydantic", "requests",
    ];
    let mut lines = Vec::new();
    for pkg in &known {
        let import_name = pkg.replace('_', "-");
        if content.contains(&format!("import {pkg}"))
            || content.contains(&format!("from {pkg}"))
        {
            lines.push(import_name);
        }
    }
    lines.join("\n")
}

// ─────────────────────────── Main ───────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = Config::from_args();
    const TOTAL: u8 = 7;

    println!("╔══════════════════════════════════════╗");
    println!("║   HexAi Builder – Python → Binary   ║");
    println!("╚══════════════════════════════════════╝");
    println!("  Źródło : {}", cfg.source_dir.display());
    println!("  Cel    : {}", cfg.out_dir.display());

    // ── 1. Validate source dir ──
    print_step(1, TOTAL, "Sprawdzam katalog źródłowy…");
    if !cfg.source_dir.exists() {
        return Err(format!(
            "Katalog '{}' nie istnieje. Utwórz go i umieść w nim main.py.",
            cfg.source_dir.display()
        )
        .into());
    }
    let main_py = cfg.source_dir.join("main.py");
    if !main_py.exists() {
        return Err("Brak pliku main.py w katalogu źródłowym.".into());
    }
    println!("  ✓ main.py znaleziony");

    // ── 2. Generate / verify requirements.txt ──
    print_step(2, TOTAL, "Weryfikuję requirements.txt…");
    let requirements = cfg.source_dir.join("requirements.txt");
    if !requirements.exists() {
        println!("  requirements.txt nie znaleziony – generuję automatycznie…");
        let content = auto_detect_requirements(&main_py);
        if content.is_empty() {
            println!("  ⚠ Nie wykryto żadnych zależności. Sprawdź main.py ręcznie.");
        } else {
            fs::write(&requirements, &content)?;
            println!("  ✓ Wygenerowano requirements.txt ({} pakietów)", content.lines().count());
        }
    } else {
        let lines = fs::read_to_string(&requirements)?
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
            .count();
        println!("  ✓ requirements.txt znaleziony ({lines} pakietów)");
    }

    // ── 3. Check PyOxidizer ──
    print_step(3, TOTAL, "Sprawdzam PyOxidizer…");
    let has_pyoxidizer = Command::new("pyoxidizer")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if has_pyoxidizer {
        println!("  ✓ PyOxidizer dostępny");
    } else {
        println!("  PyOxidizer nie znaleziony – instaluję przez pip…");
        run_cmd("pip", &["install", "pyoxidizer", "--break-system-packages"], None)
            .map_err(|e| format!("Błąd instalacji PyOxidizer: {e}"))?;
        println!("  ✓ PyOxidizer zainstalowany");
    }

    // ── 4. Write pyoxidizer.bzl ──
    print_step(4, TOTAL, "Piszę plik konfiguracyjny pyoxidizer.bzl…");
    let config_path = cfg.source_dir.join("pyoxidizer.bzl");
    fs::write(&config_path, PYOXIDIZER_CONFIG)?;
    println!("  ✓ {}", config_path.display());

    // ── 5. Build ──
    print_step(5, TOTAL, "Buduję plik wykonywalny (może potrwać kilka minut)…");
    run_cmd("pyoxidizer", &["build"], Some(&cfg.source_dir))
        .map_err(|e| format!("Błąd budowania: {e}\nSprawdź logi powyżej."))?;
    println!("  ✓ Budowanie zakończone");

    // ── 6. Locate & copy binary ──
    print_step(6, TOTAL, "Szukam pliku wykonywalnego…");
    let build_dir = cfg.source_dir.join("build");
    let exe_path = find_executable(&build_dir)
        .ok_or("Nie znaleziono pliku wykonywalnego w katalogu build/. Sprawdź logi.")?;
    println!("  ✓ Znaleziono: {}", exe_path.display());

    fs::create_dir_all(&cfg.out_dir)?;
    let dest = cfg.out_dir.join(exe_path.file_name().unwrap());
    fs::copy(&exe_path, &dest)?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest, perms)?;
    }

    println!("  ✓ Skopiowano do: {}", dest.display());

    // ── 7. Cleanup ──
    if !cfg.skip_cleanup {
        print_step(7, TOTAL, "Sprzątam pliki tymczasowe…");
        let _ = fs::remove_file(&config_path);
        println!("  ✓ Gotowe");
    } else {
        print_step(7, TOTAL, "Pomijam sprzątanie (--no-cleanup)");
    }

    println!("\n╔══════════════════════════════════════════╗");
    println!("║  ✅  Sukces!                              ║");
    println!("║  Plik wykonywalny: {}  ║", dest.file_name().unwrap().to_string_lossy());
    println!("╚══════════════════════════════════════════╝");
    println!("\nUruchom: {}", dest.display());

    Ok(())
}
