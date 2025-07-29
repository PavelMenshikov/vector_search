import os
import fitz 


SOURCE_PDF_PATH = os.path.join("data", "grokking_deep_learning.pdf")

OUTPUT_TXT_PATH = os.path.join("data", "book_content.txt") 

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Извлекает весь текст из PDF-файла.
    """
    try:
        
        document = fitz.open(pdf_path)
        
        full_text = []
       
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            full_text.append(page.get_text())
        
        print(f"Успешно обработано {len(document)} страниц.")
        return "\n".join(full_text)

    except Exception as e:
        print(f"Ошибка при обработке PDF '{pdf_path}': {e}")
        return ""

if __name__ == "__main__":
    if not os.path.exists(SOURCE_PDF_PATH):
        print(f"Ошибка: Файл не найден по пути {SOURCE_PDF_PATH}")
        print("Пожалуйста, убедитесь, что вы скачали книгу и поместили ее в папку 'data'.")
    else:
        print(f"Начинаю извлечение текста из {SOURCE_PDF_PATH}...")
        book_text = extract_text_from_pdf(SOURCE_PDF_PATH)
        
        if book_text:
            with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
                f.write(book_text)
            print(f"Весь текст успешно извлечен и сохранен в '{OUTPUT_TXT_PATH}'.")