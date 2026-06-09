use std::{fs, path::Path};

fn main() {
    // Ensure gui/out exists so include_dir! doesn't fail at compile time
    // when the GUI hasn't been built yet.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let gui_out  = Path::new(&manifest).parent().unwrap().join("gui").join("out");

    if !gui_out.exists() {
        fs::create_dir_all(&gui_out).ok();
        // Write a placeholder index.html so the embed doesn't fail
        let placeholder = r#"<!DOCTYPE html>
<html lang="pl">
<head><meta charset="UTF-8"><title>HexAi GUI</title>
<style>
  body { font-family: sans-serif; background: #1a1916; color: #f5f0e8;
         display: flex; align-items: center; justify-content: center;
         height: 100vh; margin: 0; flex-direction: column; gap: 16px; }
  code { background: #2a2825; padding: 4px 12px; border-radius: 6px;
         font-family: monospace; color: #f59e0b; }
  h1 { color: #d97706; }
</style>
</head>
<body>
  <h1>⬡ HexAi</h1>
  <p>GUI nie zostało jeszcze zbudowane.</p>
  <p>Zbuduj GUI:</p>
  <code>cd gui &amp;&amp; npm install &amp;&amp; npm run build</code>
  <p>Następnie przebuduj binarkę:</p>
  <code>cargo build --release -p hexai</code>
</body>
</html>"#;
        fs::write(gui_out.join("index.html"), placeholder).ok();
    }

    // Rerun if gui/out changes
    println!("cargo:rerun-if-changed=../gui/out");
    println!("cargo:rerun-if-changed=../gui/package.json");
}
