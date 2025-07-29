
import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# -----------------------------
from contextlib import asynccontextmanager
from src.search import search 

def clean_llm_output(text: str) -> str:
   
    if not text: return ""
    cleaned_text = re.sub(r'(\*\*|##+\s*)', '', text)
    cleaned_text = re.sub(r'\s*\n\s*', '\n', cleaned_text).strip()
    return cleaned_text

def load_all_models_and_data():
   
    
    print("-> Начата синхронная загрузка данных и моделей...")
    try:
        embeddings = np.load("vectors/embeddings.npy")
        import json
        with open("vectors/chunks.json", "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print("   - Поисковые индексы (векторы и чанки) успешно загружены.")
    except FileNotFoundError:
        print("\nКРИТИЧЕСКАЯ ОШИБКА: Файлы для поиска не найдены. Запустите `python src/create_embeddings.py`\n")
        exit()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("\nКРИТИЧЕСКАЯ ОШИБКА: Не установлен OPENROUTER_API_KEY в .env файле\n")
    
    print("   - Ключ API для OpenRouter загружен.")
    print("-> Загрузка завершена.")
    return {"embeddings": embeddings, "chunks": chunks, "openrouter_api_key": api_key}


ml_models = load_all_models_and_data()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("-> Сервер FastAPI запущен и готов к работе.")
    yield
    print("-> Сервер FastAPI останавливается...")
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="Professional AI Assistant (HyDE)", version="PRO-HYDE-Upgraded")

class SearchQuery(BaseModel): 
    query: str
    top_k: int = 5

@app.post("/ask-ai-assistant", summary="RAG с качественным HyDE", tags=["AI Assistant"])
async def api_rag_assistant(search_query: SearchQuery):
    
    api_key = ml_models["openrouter_api_key"]
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    print(f"[HyDE] Генерация гипотетического ответа для запроса: '{search_query.query}'")
    hyde_prompt = f"Сгенерируй короткий (один абзац) гипотетический ответ на следующий вопрос. Ответ должен быть на русском языке. ВОПРОС: {search_query.query}"
    
    try:
        hyde_response = requests.post(api_url, headers=headers, json={
            "model": "mistralai/mistral-nemo",
            "messages": [{"role": "user", "content": hyde_prompt}]
        })
        hyde_response.raise_for_status()
        hypothetical_answer = hyde_response.json()["choices"][0]["message"]["content"]
        print(f"[HyDE] Качественный гипотетический ответ: {hypothetical_answer[:100]}...")
    except Exception as e:
        print(f"ОШИБКА HyDE: {e}. Продолжаем поиск по оригинальному запросу.")
        hypothetical_answer = search_query.query
    
    print(f"[Поиск] Ищем контекст, используя гипотетический ответ...")
    search_results = search(
        query=hypothetical_answer,
        top_k=search_query.top_k,
        embeddings=ml_models["embeddings"],
        chunks=ml_models["chunks"]
    )
    context = "\n---\\n".join([result['text'] for result in search_results])
    
    system_prompt = """Ты — ИИ-ассистент, эксперт по книге "Грокаем глубокое обучение".
ПРАВИЛА:
1. Твой ответ ДОЛЖЕН БЫТЬ ТОЛЬКО на русском языке.
2. Твой ответ ДОЛЖЕН основываться ИСКЛЮЧИТЕЛЬНО на предоставленном КОНТЕКСТЕ.
3. Если в КОНТЕКСТЕ нет ответа, напиши: "В предоставленных материалах нет ответа на этот вопрос.".
4. Твой ответ должен быть чистым текстом, без Markdown."""
    
    user_prompt = f"""КОНТЕКСТ ИЗ КНИГИ:
{context}
---
Оригинальный ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{search_query.query}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"[Генерация] Отправка улучшенного контекста в Mistral Nemo...")
    try:
        response = requests.post(api_url, headers=headers, json={
            "model": "mistralai/mistral-nemo",
            "messages": messages
        })
        response.raise_for_status()
        raw_answer = response.json()["choices"][0]["message"]["content"]
        clean_answer = clean_llm_output(raw_answer)
        print(f"[Генерация] Финальный ответ успешно сгенерирован.")
        
        return {
            "llm_answer": clean_answer, 
            "source_chunks": search_results
        }
    except Exception as e:
        print(f"ОШИБКА Генерации: {e}")
        raise HTTPException(status_code=500, detail=str(e))