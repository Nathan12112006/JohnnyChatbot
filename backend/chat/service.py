import os
from openai import OpenAI
from utils.intent import classify_intent
from utils.retrieval import retrieve_context
from chat.prompts import SYSTEM_PROMPT, build_prompt

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key == "your_openai_api_key_here":
    raise ValueError("OPENAI_API_KEY environment variable is not set or is placeholder. Please set your actual OpenAI API key.")

client = OpenAI(api_key=api_key)

async def generate_response(message: str):
    intent = classify_intent(message)
    context = retrieve_context(intent)
    prompt = build_prompt(context, message)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )

    reply = response.choices[0].message.content

    suggestions = [
        "What projects has Nathan worked on?",
        "What are Nathan’s top skills?",
        "What is Nathan background in computer science?",
        "How does Nathan approach problem solving?"
    ]

    return reply, suggestions
