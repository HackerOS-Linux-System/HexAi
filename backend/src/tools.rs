use crate::error::Result;
use std::collections::HashMap;

pub fn get_weather(city: &str) -> String {
    format!("Pogoda w {city}: słonecznie, 22°C.")
}

pub async fn search_web(query: &str, serper_key: Option<&str>) -> String {
    if let Some(key) = serper_key {
        if let Ok(result) = serper_search(query, key).await {
            return result;
        }
    }
    ddg_search(query).await.unwrap_or_else(|e| format!("Błąd wyszukiwania: {e}"))
}

async fn serper_search(query: &str, api_key: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Item { title: String, snippet: String }
    #[derive(serde::Deserialize)]
    struct Resp { organic: Vec<Item> }

    let client = reqwest::Client::new();
    let resp: Resp = client
        .post("https://google.serper.dev/search")
        .header("X-API-KEY", api_key)
        .json(&serde_json::json!({ "q": query, "num": 5 }))
        .send()
        .await?
        .json()
        .await?;

    Ok(resp.organic.iter()
        .map(|i| format!("{}: {}", i.title, i.snippet))
        .collect::<Vec<_>>()
        .join("\n"))
}

async fn ddg_search(query: &str) -> anyhow::Result<String> {
    // Simple DuckDuckGo HTML scrape (lite.duckduckgo.com)
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0")
        .build()?;
    let url = format!(
        "https://lite.duckduckgo.com/lite/?q={}",
        urlencoding::encode(query)
    );
    let html = client.get(&url).send().await?.text().await?;
    let doc = scraper::Html::parse_document(&html);
    let sel = scraper::Selector::parse("a.result-link").unwrap_or_else(|_| scraper::Selector::parse("a").unwrap());
    let results: Vec<String> = doc.select(&sel)
        .take(5)
        .map(|el| el.text().collect::<String>())
        .collect();
    Ok(if results.is_empty() { format!("Brak wyników dla: {query}") } else { results.join("\n") })
}

pub async fn dispatch_tool(
    tool: &str,
    args: &HashMap<String, String>,
    serper_key: Option<&str>,
) -> String {
    match tool {
        "get_weather" => get_weather(args.get("city").map(|s| s.as_str()).unwrap_or("Warsaw")),
        "search_web"  => search_web(args.get("query").map(|s| s.as_str()).unwrap_or(""), serper_key).await,
        other         => format!("Narzędzie '{other}' nie istnieje."),
    }
}
