use crate::engine::LlmEngine;
use regex::Regex;
use std::collections::HashMap;

pub const MAX_ITERATIONS: usize = 4;

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, String>,
}

pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let re     = Regex::new(r"\{tool:(\w+)([^}]*)\}").unwrap();
    let arg_re = Regex::new(r#"(\w+)="([^"]+)""#).unwrap();
    re.captures_iter(text).map(|cap| {
        let name = cap[1].to_string();
        let args = arg_re.captures_iter(&cap[2])
            .map(|a| (a[1].to_string(), a[2].to_string()))
            .collect();
        ToolCall { name, args }
    }).collect()
}

pub fn strip_tool_calls(text: &str) -> String {
    let re = Regex::new(r"\{tool:\w+[^}]*\}").unwrap();
    re.replace_all(text, "").trim().to_string()
}

fn tool_weather(args: &HashMap<String, String>) -> String {
    let city = args.get("city").map(|s| s.as_str()).unwrap_or("Warsaw");
    format!("Pogoda w {city}: słonecznie, 22°C, wilgotność 45%.")
}

async fn tool_search_web(args: &HashMap<String, String>, serper_key: Option<&str>) -> String {
    let query = args.get("query").map(|s| s.as_str()).unwrap_or("");
    if query.is_empty() { return "Brak zapytania.".into(); }
    search_web(query, serper_key).await
}

async fn tool_calculate(args: &HashMap<String, String>) -> String {
    let expr = args.get("expr").or_else(|| args.get("expression")).map(|s| s.as_str()).unwrap_or("");
    let clean: String = expr.chars().filter(|c| "0123456789.+-*/() ".contains(*c)).collect();
    match meval::eval_str(&clean) {
        Ok(result) => format!("{expr} = {result}"),
        Err(e)     => format!("Błąd obliczenia: {e}"),
    }
}

async fn tool_datetime(_args: &HashMap<String, String>) -> String {
    format!("Aktualna data i czas: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S %Z"))
}

pub async fn search_web(query: &str, serper_key: Option<&str>) -> String {
    if let Some(key) = serper_key {
        if let Ok(r) = serper_search(query, key).await { return r; }
    }
    ddg_search(query).await.unwrap_or_else(|e| format!("Błąd wyszukiwania: {e}"))
}

async fn serper_search(query: &str, api_key: &str) -> Result<String, reqwest::Error> {
    #[derive(serde::Deserialize)] struct Item { title: String, snippet: String }
    #[derive(serde::Deserialize)] struct Resp { #[serde(default)] organic: Vec<Item> }

    let resp: Resp = reqwest::Client::new()
        .post("https://google.serper.dev/search")
        .header("X-API-KEY", api_key)
        .json(&serde_json::json!({ "q": query, "num": 5 }))
        .send().await?.json().await?;

    Ok(resp.organic.iter()
        .map(|i| format!("• {}: {}", i.title, i.snippet))
        .collect::<Vec<_>>().join("\n"))
}

async fn ddg_search(query: &str) -> anyhow::Result<String> {
    let client = reqwest::Client::builder().user_agent("Mozilla/5.0").build()?;
    let url    = format!("https://lite.duckduckgo.com/lite/?q={}", urlencoding::encode(query));
    let html   = client.get(&url).send().await?.text().await?;
    let doc    = scraper::Html::parse_document(&html);
    let sel    = scraper::Selector::parse("td.result-snippet").unwrap();
    let results: Vec<String> = doc.select(&sel).take(5)
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Ok(if results.is_empty() { format!("Brak wyników dla: {query}") } else { results.join("\n") })
}

pub async fn dispatch(call: &ToolCall, serper_key: Option<&str>) -> String {
    match call.name.as_str() {
        "get_weather" | "weather" => tool_weather(&call.args),
        "search_web"  | "search"  => tool_search_web(&call.args, serper_key).await,
        "calculate"   | "calc"    => tool_calculate(&call.args).await,
        "datetime"    | "now"     => tool_datetime(&call.args).await,
        other => format!("Nieznane narzędzie: '{other}'. Dostępne: weather, search_web, calculate, datetime"),
    }
}

// ── Tool loop ─────────────────────────────────────────────────────

use futures::StreamExt;

pub async fn run_tool_loop_sync(
    engine:     &LlmEngine,
    prompt:     &str,
    history:    &[(String, String)],
    serper_key: Option<&str>,
) -> crate::error::Result<String> {
    let mut working_prompt  = prompt.to_string();
    let mut working_history = history.to_vec();

    for _ in 0..MAX_ITERATIONS {
        let response = engine.generate_sync(&working_prompt, &working_history).await?;
        let calls    = parse_tool_calls(&response);
        if calls.is_empty() { return Ok(response); }

        let mut tool_results = vec![];
        for call in &calls {
            let result = dispatch(call, serper_key).await;
            tool_results.push(format!("[{}] → {}", call.name, result));
        }

        let clean = strip_tool_calls(&response);
        working_history.push((working_prompt.clone(), clean));
        working_prompt = format!(
            "Wyniki narzędzi:\n{}\n\nProszę kontynuuj odpowiedź uwzględniając powyższe wyniki.",
            tool_results.join("\n")
        );
    }
    engine.generate_sync(&working_prompt, &working_history).await
}

pub async fn run_tool_loop_stream(
    engine:     &LlmEngine,
    prompt:     &str,
    history:    &[(String, String)],
    serper_key: Option<&str>,
    tx:         tokio::sync::mpsc::Sender<String>,
) {
    let mut working_prompt  = prompt.to_string();
    let mut working_history = history.to_vec();

    for _ in 0..MAX_ITERATIONS {
        let stream = match engine.generate_stream(&working_prompt, &working_history).await {
            Ok(s)  => s,
            Err(e) => { let _ = tx.send(format!("❌ {e}")).await; return; }
        };
        tokio::pin!(stream);

        let mut full = String::new();
        while let Some(tok) = stream.next().await {
            match tok {
                Ok(t)  => { full.push_str(&t); let _ = tx.send(t).await; }
                Err(e) => { let _ = tx.send(format!("❌ {e}")).await; return; }
            }
        }

        let calls = parse_tool_calls(&full);
        if calls.is_empty() { return; }

        let _ = tx.send("\n\n🔧 Wykonuję narzędzia…\n".to_string()).await;

        let mut tool_results = vec![];
        for call in &calls {
            let result = dispatch(call, serper_key).await;
            let msg    = format!("[{}] → {}\n", call.name, result);
            let _ = tx.send(msg.clone()).await;
            tool_results.push(msg);
        }

        let clean = strip_tool_calls(&full);
        working_history.push((working_prompt.clone(), clean));
        working_prompt = format!(
            "Wyniki narzędzi:\n{}\n\nProszę kontynuuj odpowiedź.",
            tool_results.join("")
        );
    }
}
