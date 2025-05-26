import sys
import json
from grok_api import GrokAssistant
from history import ChatHistory

def main():
    assistant = GrokAssistant()
    history = ChatHistory()

    # Odczyt danych z wejścia (od Tauri)
    input_data = sys.stdin.read()
    try:
        data = json.loads(input_data)
    except json.JSONDecodeError:
        print(json.dumps({"status": "error", "data": "Błąd parsowania danych"}))
        return

    action = data.get("action")
    prompt = data.get("prompt")

    if action == "text":
        result = assistant.generate_text(prompt)
    elif action == "image":
        result = assistant.generate_image(prompt)
    elif action == "history":
        result = {"status": "success", "data": history.get_history()}
    else:
        result = {"status": "error", "data": "Nieznana akcja"}

    # Zapisz do historii (jeśli nie jest to żądanie historii)
    if action in ["text", "image"]:
        history.add_message(prompt, result["data"], action)

    # Wysłanie wyniku z powrotem do Tauri
    print(json.dumps(result))

if __name__ == "__main__":
    main()
