"""
Утилита Full Rank Tracing для отладки семантического поиска и лексического бустинга.
"""
import os
import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def investigate_retrieval(
    index_path: str = "faiss_index", 
    query: str = "В Транспорте Лысьвы есть команда для вывода пароля, выведи её", 
    target_keyword: str = "swordfish"
):
    print(f"[*] Загрузка модели и индекса '{index_path}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    total_chunks = vectorstore.index.ntotal
    print(f"[+] Всего чанков в базе: {total_chunks}")

    # 1. ПРОВЕРКА ФИЗИЧЕСКОГО НАЛИЧИЯ В БАЗЕ
    target_doc = None
    for doc_id, doc in vectorstore.docstore._dict.items():
        if target_keyword.lower() in doc.page_content.lower():
            target_doc = doc
            break
    
    if not target_doc:
        print(f"\n[-] ОШИБКА: Чанк со словом '{target_keyword}' ФИЗИЧЕСКИ ОТСУТСТВУЕТ В FAISS!")
        print("Возможные причины: файл не попал в knowledge_base, либо ты забыл перезапустить build_index.py после создания файла.")
        return
        
    print(f"\n[+] ШАГ 1: Чанк успешно найден в базе.")
    print(f"Содержимое: {target_doc.page_content.strip()}")

    # 2. АНАЛИЗ СЕМАНТИЧЕСКОГО СКОРА (FAISS RAW)
    print(f"\n[*] ШАГ 2: Выполняем FAISS поиск с k={total_chunks} (выгружаем всю базу)...")
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=total_chunks)
    
    rank_before = -1
    score_before = -1
    for i, (doc, score) in enumerate(docs_and_scores):
        if target_keyword.lower() in doc.page_content.lower():
            rank_before = i + 1
            score_before = score
            break
            
    print(f"-> Исходная позиция в FAISS (до бустинга): Место {rank_before} из {total_chunks}")
    print(f"-> Исходная L2 дистанция: {score_before:.4f}")
    
    if rank_before > 50:
        print(f"[!] ВНИМАНИЕ: В твоем rag_bot.py стоит k=50. Так как чанк на {rank_before} месте, он физически не доходит до этапа переранжирования!")

    # 3. АНАЛИЗ ЛЕКСИЧЕСКОГО БУСТИНГА
    query_roots = [w.lower()[:5] for w in re.findall(r'\b\w{5,}\b', query)]
    print(f"\n[*] ШАГ 3: Применяем лексический бустинг.")
    print(f"-> Корни из запроса: {query_roots}")
    
    source_name = target_doc.metadata.get('source', '').lower()
    content_lower = target_doc.page_content.lower()
    
    # Разделяем совпадения как в rag_bot.py
    source_matched_roots = [root for root in query_roots if root in source_name]
    content_matched_roots = [root for root in query_roots if root in content_lower]
    
    source_matches = len(source_matched_roots)
    content_matches = len(content_matched_roots)
    boost_value = (20.0 * source_matches) + (2.0 * content_matches)
    
    print(f"-> Совпадения в названии (бонус -10.0 за каждое): {source_matched_roots} (Всего: {source_matches})")
    print(f"-> Совпадения в тексте (бонус -2.0 за каждое): {content_matched_roots} (Всего: {content_matches})")
    print(f"-> Итоговый бонус (boost): -{boost_value:.1f}")
    print(f"-> Итоговый скор (L2) после бустинга: {(score_before - boost_value):.4f}")

    print("\n[ВЫВОД]")
    print("Проанализируй эти логи. Скорее всего, причина в том, что исходный скор слишком велик, и k=50 просто не хватает, чтобы вытащить его из базы для бустинга.")

if __name__ == "__main__":
    investigate_retrieval()