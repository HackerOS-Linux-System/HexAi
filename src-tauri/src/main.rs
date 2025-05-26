use tauri::{command, AppHandle};
use std::process::{Command, Stdio};
use std::io::{Read, Write};

#[command]
fn invoke_python(app_handle: AppHandle, action: String, prompt: String) -> String {
    let python_script = app_handle
        .path_resolver()
        .resolve_resource("../backend/main.py")
        .expect("Nie znaleziono skryptu Pythona");

    let input_data = serde_json::json!({
        "action": action,
        "prompt": prompt
    });

    let mut child = Command::new("python3")
        .arg(python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Błąd uruchamiania Pythona");

    // Wysłanie danych do Pythona
    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(input_data.to_string().as_bytes())
            .expect("Błąd zapisu do stdin");
    }

    // Odczyt wyniku
    let mut output = String::new();
    if let Some(stdout) = child.stdout.as_mut() {
        stdout
            .read_to_string(&mut output)
            .expect("Błąd odczytu stdout");
    }

    child.wait().expect("Błąd oczekiwania na Pythona");
    output
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![invoke_python])
        .run(tauri::generate_context!())
        .expect("Błąd uruchamiania aplikacji Tauri");
}
