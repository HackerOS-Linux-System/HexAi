import os
import requests
from dotenv import load_dotenv
from xai_grok import Grok
from PIL import Image
from io import BytesIO

load_dotenv()

class GrokAssistant:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        self.client = Grok(api_key=self.api_key)
        self.image_dir = os.path.join(os.path.dirname(__file__), "../data/images")
        os.makedirs(self.image_dir, exist_ok=True)

    def generate_text(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model="grok-3",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            return {"status": "success", "data": response.choices[0].message.content}
        except Exception as e:
            return {"status": "error", "data": f"Błąd API: {str(e)}"}

    def generate_image(self, prompt):
        try:
            response = self.client.images.generate(
                model="flux-1",
                prompt=prompt,
                size="1024x1024",
                quality="standard"
            )
            image_url = response.data[0].url
            # Pobierz i zapisz obraz
            image_response = requests.get(image_url)
            image = Image.open(BytesIO(image_response.content))
            image_path = os.path.join(self.image_dir, f"image_{hash(prompt)}.png")
            image.save(image_path)
            return {"status": "success", "data": image_path}
        except Exception as e:
            return {"status": "error", "data": f"Błąd API: {str(e)}"}
