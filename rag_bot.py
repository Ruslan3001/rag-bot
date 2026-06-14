"""
Консольный RAG-бот (REPL интерфейс).
Использует:
- Локальный FAISS индекс (эмбеддинги paraphrase-multilingual).
- Облачную LLM (OpenAI) для генерации ответов.
- Техники промптинга: Few-Shot и Chain-of-Thought.
"""
import os
import argparse
import re

# Отключаем прогресс-бары загрузки весов от Hugging Face
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Промпт объединяет в себе инструкции (CoT) и примеры (Few-Shot)
PROMPT_TEMPLATE = """Ты — корпоративный ИИ-помощник QuantumForge. Твоя задача — отвечать на вопросы сотрудников на основе базы знаний.
Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги.

Если в предоставленном контексте нет нужной информации для ответа, честно скажи: "Я не знаю". Не придумывай факты и не ищи информацию во внешней памяти!

ПРИМЕРЫ (Few-Shot & CoT):
Вопрос: Где применялась технология перемещения готовых тоннелей?
Ответ: 
1. Сначала найду в контексте упоминания "технологии перемещения готовых тоннелей".
2. В документе "Транспорт Мицара" сказано, что при строительстве метро под Влтавой применялась уникальная технология перемещения и закрепления готовых участков тоннелей.
3. Следовательно, ответ — При строительстве метро под Влтавой (Транспорт Мицара).

Вопрос: Какая оплата проезда в трамвайной сети Веги?
Ответ: 
1. Ищу в контексте информацию об "оплате проезда" в "трамвайной сети Веги".
2. В документе "Трамвайная сеть Веги" указано: "В дневное время, прежде чем спуститься в подземный вестибюль, пассажир должен приобрести в кассе и погасить компостером посадочный талон".
3. Следовательно, ответ — В дневное время, прежде чем спуститься в подземный вестибюль, пассажир должен приобрести в кассе и погасить компостером посадочный талон.

Вопрос: Как настроить роутер?
Ответ:
1. Ищу в контексте информацию о настройке роутера.
2. В предоставленных документах базы знаний нет упоминаний роутеров или их настройки.
3. Следовательно, ответ — Я не знаю.

КОНТЕКСТ ДЛЯ ОТВЕТА (База Знаний):
{context}

Вопрос пользователя: {question}
Ответ: """

def run_bot(index_path: str = "faiss_index", debug: bool = False):
    # Загружаем переменные окружения из файла .env
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[-] Ошибка: Не задана переменная окружения GOOGLE_API_KEY.")
        print("Установите её в файле .env или терминале: set GOOGLE_API_KEY=ВАШ-КЛЮЧ (Windows) или export GOOGLE_API_KEY=ВАШ-КЛЮЧ (Mac/Linux)")
        return

    print("[*] Загрузка локальной модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("[*] Подключение к FAISS индексу...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print("[*] Инициализация LLM (Gemini)...")
    # Используем gemini-1.5-flash для быстроты и эффективности, Temperature=0 делает ответы детерминированными
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0)

    print("\n[+] Бот готов! Введите ваш запрос (или 'exit' для выхода).")
    while True:
        query = input("\nВы: ")
        if query.lower() in ['exit', 'quit', 'выход', 'q']:
            break
        if not query.strip(): continue
        
        # Используем метод с возвратом скоринга (дистанции L2)
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=8)
        
        # --- ЛЕКСИЧЕСКИЙ БУСТИНГ (ПЕРЕРАНЖИРОВАНИЕ) ---
        # Извлекаем корни длинных слов из запроса (первые 5 букв), чтобы нивелировать падежи (Авалон/Авалона)
        query_roots = [w.lower()[:5] for w in re.findall(r'\b\w{5,}\b', query)]
        
        boosted_docs = []
        for doc, score in docs_and_scores:
            source_name = doc.metadata.get('source', '').lower()
            # Если корень значимого слова из запроса есть в названии документа, даем сильный бонус
            if any(root in source_name for root in query_roots):
                score -= 0.5  # Для метрики L2: чем меньше значение, тем вектор "ближе"
            boosted_docs.append((doc, score))
            
        # Пересортируем выдачу по обновленному скору
        boosted_docs.sort(key=lambda x: x[1])

        # Отделяем документы для передачи в контекст LLM
        retrieved_docs = [doc for doc, score in boosted_docs]
        
        if debug:
            print("\n[DEBUG] Извлеченные чанки из векторной базы:")
            for i, (doc, score) in enumerate(boosted_docs, 1):
                source = doc.metadata.get('source', 'Неизвестно')
                print(f"\n--- Чанк {i} | Источник: {source} | L2 Дистанция: {score:.4f} ---")
                print(doc.page_content)
            
            confirm = input("\n[?] Отправить этот контекст в LLM? (y/n): ")
            if confirm.lower() not in ['y', 'yes', 'да', 'д']:
                print("[-] Отправка отменена. Попробуйте переформулировать запрос.")
                continue

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=query)
        
        print("\nБот: (думает...)")
        response = llm.invoke(prompt)
        # Извлекаем текст, так как Gemini может возвращать список словарей вместо строки
        answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
        print(f"\n{answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Консольный RAG-бот")
    parser.add_argument("--debug", action="store_true", help="Включить режим отладки для проверки извлеченного контекста")
    args = parser.parse_args()
    
    run_bot(debug=args.debug)