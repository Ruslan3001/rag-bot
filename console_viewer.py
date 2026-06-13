"""
Скрипт для интерактивного просмотра и поиска по индексу прямо в консоли,
без необходимости выгружать данные в CSV.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_interactive_viewer(index_path: str = "faiss_index"):
    print("[*] Загрузка локальной модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"[*] Чтение FAISS индекса из '{index_path}'...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"[+] Индекс загружен. Всего чанков: {len(vectorstore.docstore._dict)}")
    
    while True:
        query = input("\nВведите запрос для поиска (или 'exit' для выхода): ")
        if query.lower() in ['exit', 'выход', 'q']:
            break
            
        results = vectorstore.similarity_search(query, k=10)
        for i, doc in enumerate(results, 1):
            print(f"\n--- Чанк {i} | Источник: {doc.metadata.get('source', 'Неизвестно')} ---")
            print(doc.page_content)

if __name__ == "__main__":
    run_interactive_viewer()