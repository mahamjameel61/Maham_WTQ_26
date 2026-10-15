from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
model = init_chat_model("google_genai:gemini-3.5-flash-lite")
#model = init_chat_model("google_genai:gemini-3.1-flash-lite")
response = model.invoke("give me paragraph on deep learning")
print(response.content[0]["text"])
#print(response.content)