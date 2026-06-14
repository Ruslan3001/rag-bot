"""
Утилита для жесткого лексического поиска по локальному FAISS-индексу.
Позволяет проверить физическое наличие терминов в чанках после индексации.
"""

import sys
import os

# Отключаем прогресс-бары загрузки весов от Hugging Face
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def search_exact(term: str, index_path: str = "faiss_index"):
    print("[*] Загрузка модели и индекса...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    count = 0
    for doc_id, doc in vectorstore.docstore._dict.items():
        if term.lower() in doc.page_content.lower():
            count += 1
            print(f"\n--- Чанк {count} | Источник: {doc.metadata.get('source', 'Неизвестно')} ---")
            print(doc.page_content)
            
    print(f"\n[+] Всего найдено чанков с точным вхождением '{term}': {count}")

if __name__ == "__main__":
    search_term = sys.argv[1] if len(sys.argv) > 1 else "Лысьва"
    search_exact(search_term)