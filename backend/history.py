import json
import os

class ChatHistory:
    def __init__(self):
        self.history_file = os.path.join(os.path.dirname(__file__), "../data/chat_history.json")
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        self.history = self.load_history()

    def load_history(self):
        try:
            with open(self.history_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_history(self):
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def add_message(self, prompt, response, action):
        self.history.append({
            "prompt": prompt,
            "response": response,
            "action": action,
            "timestamp": str(datetime.datetime.now())
        })
        self.save_history()

    def get_history(self):
        return self.history
