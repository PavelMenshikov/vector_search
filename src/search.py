# Файл: src/search.py

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


print("--- [search.py] Загружается модель-кодировщик sentence-transformer... ---")
MODEL = SentenceTransformer('all-MiniLM-L6-v2')
print("--- [search.py] Модель-кодировщик загружена. ---")


def search(query: str, embeddings: np.ndarray, chunks: list, top_k: int = 3) -> list:
    """
    Принимает текстовый запрос и выполняет поиск по векторам (эмбеддингам).
    
    Args:
        query (str): Текст поискового запроса от пользователя.
        embeddings (np.ndarray): Массив numpy со всеми эмбеддингами из базы.
        chunks (list): Список со всеми текстовыми чанками из базы.
        top_k (int): Количество лучших результатов для возврата.

    Returns:
        list: Список словарей, где каждый словарь представляет найденный результат.
    """
    
    query_embedding = MODEL.encode([query], show_progress_bar=False)

   
    similarities = cosine_similarity(query_embedding, embeddings)

  
    top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]

    
    results = []
    for index in top_k_indices:
        result = {
            "text": chunks[index],
            "score": float(similarities[0][index])
        }
        results.append(result)
    
    return results