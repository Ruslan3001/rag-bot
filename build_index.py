"""
Скрипт для построения векторного индекса базы знаний (RAG MVP).
Применяется принцип KISS: используем легковесную локальную модель эмбеддингов
и in-memory базу FAISS для обеспечения скорости и нулевых затрат на инфраструктуру.
"""

import os
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_index(
    source_dir: str = "knowledge_base",
    index_save_path: str = "faiss_index",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> None:
    """
    Загружает Markdown файлы из указанной директории, разбивает их на логические чанки,
    генерирует эмбеддинги и сохраняет FAISS индекс локально на диск.
    
    :param source_dir: Путь к директории с базой знаний (содержит .md файлы).
    :param index_save_path: Путь для сохранения построенного индекса.
    :param chunk_size: Максимальный размер чанка в символах.
    :param chunk_overlap: Перекрытие между соседними чанками (Overlap).
    """
    print(f"[*] Инициализация загрузки документов из '{source_dir}'...")
    
    # Используем glob паттерн для парсинга всех md-файлов в директории
    loader = DirectoryLoader(
        source_dir, 
        glob="**/*.md", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        show_progress=True
    )
    documents = loader.load()
    print(f"[+] Загружено документов: {len(documents)}")

    # Разбиваем документы на чанки
    # Сепараторы настроены под Markdown для сохранения структуры параграфов и заголовков
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"[+] Создано чанков: {len(chunks)}")

    # Применяем паттерн Metadata Enrichment (Жемчужина архитектуры)
    # Добавляем название файла (которое содержит суть документа) в начало каждого чанка
    print("[*] Применение Metadata Enrichment (внедрение названий документов в чанки)...")
    for chunk in chunks:
        filename = os.path.basename(chunk.metadata.get('source', '')).replace('.md', '')
        chunk.page_content = f"[Документ: {filename}]\n{chunk.page_content}"

    # Инициализируем локальную модель эмбеддингов (Hugging Face)
    print("[*] Загрузка мультиязычной модели эмбеддингов 'paraphrase-multilingual-MiniLM-L12-v2'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Строим и сохраняем FAISS индекс
    print("[*] Генерация эмбеддингов и построение FAISS индекса...")
    start_time = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed_time = time.time() - start_time
    print(f"[+] Индекс построен за {elapsed_time:.2f} секунд.")

    vectorstore.save_local(index_save_path)
    print(f"[+] FAISS индекс успешно сохранен в '{index_save_path}'")
    
    # Быстрый Smoke-test
    run_smoke_test(vectorstore)

def run_smoke_test(vectorstore: FAISS) -> None:
    """Выполняет тестовый запрос к индексу для валидации пайплайна (Smoke Test)."""
    query = "Транспортная система на Авалоне"
    print(f"\n[*] Выполнение тестового запроса (Top-2): '{query}'")
    
    results = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Неизвестный источник")
        print(f"\n--- Результат {i} (Источник: {source}) ---\n{doc.page_content[:250]}...")

if __name__ == "__main__":
    if not os.path.exists("knowledge_base"):
        print("[-] Ошибка: Директория 'knowledge_base' не найдена. Создайте директорию и добавьте .md файлы.")
    else:
        build_vector_index()