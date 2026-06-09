use dashmap::DashMap;
use std::sync::Arc;

#[derive(Clone, Default)]
pub struct UserProfiler {
    facts: Arc<DashMap<String, Vec<String>>>,
}

impl UserProfiler {
    pub fn new() -> Self { Self::default() }

    pub fn add_fact(&self, user_id: &str, fact: String) {
        let mut entry = self.facts.entry(user_id.to_string()).or_default();
        entry.insert(0, fact);
        if entry.len() > 20 { entry.truncate(20); }
    }

    pub fn get_facts(&self, user_id: &str, limit: usize) -> Vec<String> {
        self.facts
            .get(user_id)
            .map(|v| v.iter().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    pub fn update_from_message(&self, user_id: &str, message: &str) {
        let patterns: &[(&str, &str)] = &[
            (r"preferuję|wolę|używam", "preference"),
            (r"pracuję.*na|używam.*w", "tech_stack"),
        ];
        for (pat, fact_type) in patterns {
            if let Ok(re) = regex::Regex::new(pat) {
                if let Some(cap) = re.find(message) {
                    let snippet = &message[cap.start()..];
                    let fact = format!("{fact_type}: {}", snippet.chars().take(60).collect::<String>());
                    self.add_fact(user_id, fact);
                    break;
                }
            }
        }
    }
}
