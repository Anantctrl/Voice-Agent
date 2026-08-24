import os 
from  dotenv import load_dotenv
from groq import Groq
# from src.connection import 

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(message: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )

    return response.choices[0].message.content
    



# Only run this test block when executed directly (python llm.py),
# so importing chat() into the voice server does NOT fire a Groq request.
if __name__ == "__main__":
    answer = chat("hello")
    print(answer)