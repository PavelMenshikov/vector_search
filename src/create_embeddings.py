from sentence_transformers import SentenceTransformer
import os
import numpy as np
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

print("1. Загрузка ML-модели...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   - Модель загружена.")


input_file_path = os.path.join("data", "book_content.txt")


print(f"2. Чтение сырого текста из '{input_file_path}'...")
try:
    with open(input_file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    print("   - Текст успешно прочитан.")
except FileNotFoundError:
    print(f"Ошибка: Файл '{input_file_path}' не найден. Запустите сначала data_loader.py.")
    exit()


print("3. Нарезка текста на осмысленные чанки...")
text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(full_text)
print(f"   - Текст нарезан на {len(chunks)} чанков.")



print(f"\n4. Создание эмбеддингов для {len(chunks)} чанков...")

embeddings = model.encode(chunks, show_progress_bar=True)
print("   - Эмбеддинги созданы.")



output_folder = "vectors"
os.makedirs(output_folder, exist_ok=True)
np.save(os.path.join(output_folder, "embeddings.npy"), embeddings)


with open(os.path.join(output_folder, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)
    
print(f"Эмбеддинги и чанки сохранены в папку {output_folder}.")