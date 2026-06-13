"""
Скрипт для выгрузки содержимого FAISS индекса в CSV-таблицу.
Позволяет просматривать чанки, метаданные и использовать инструменты 
сортировки и поиска (например, через Excel или Pandas).
"""

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def export_index_to_csv(index_path: str = "faiss_index", output_csv: str = "index_contents.csv"):
    print(f"[*] Загрузка локальной модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print(f"[*] Чтение FAISS индекса из '{index_path}'...")
    # allow_dangerous_deserialization=True необходим для загрузки pickle-файлов в новых версиях LangChain
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("[*] Извлечение документов (чанков)...")
    data = []
    # В FAISS оригинальные тексты хранятся в docstore._dict
    for doc_id, doc in vectorstore.docstore._dict.items():
        row = {
            "chunk_id": doc_id,
            "source": doc.metadata.get("source", "Неизвестно"),
            # Заменяем реальные переносы строк на текстовые '\n',
            # чтобы Excel помещал весь чанк строго в одну строку и одну ячейку.
            "content": doc.page_content.replace('\n', ' \\n ')
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[+] Успешно экспортировано {len(df)} чанков в файл '{output_csv}'.")

if __name__ == "__main__":
    export_index_to_csv()