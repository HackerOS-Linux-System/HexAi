use tauri::Manager;

#[tauri::command]
fn get_api_base() -> String {
    std::env::var("HEXAI_API_BASE")
        .unwrap_or_else(|_| "http://localhost:8000".to_string())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .invoke_handler(tauri::generate_handler![get_api_base])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
