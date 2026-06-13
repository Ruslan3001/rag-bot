"""
Консольный RAG-бот (REPL интерфейс).
Использует:
- Локальный FAISS индекс (эмбеддинги paraphrase-multilingual).
- Облачную LLM (OpenAI) для генерации ответов.
- Техники промптинга: Few-Shot и Chain-of-Thought.
"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
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

def run_bot(index_path: str = "faiss_index"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[-] Ошибка: Не задана переменная окружения OPENAI_API_KEY.")
        print("Установите её в терминале: set OPENAI_API_KEY=sk-ВАШ-КЛЮЧ (Windows) или export OPENAI_API_KEY=sk-ВАШ-КЛЮЧ (Mac/Linux)")
        return

    print("[*] Загрузка локальной модели эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("[*] Подключение к FAISS индексу...")
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    print("[*] Инициализация LLM...")
    # Используем gpt-3.5-turbo для быстроты и дешевизны, Temperature=0 делает ответы детерминированными
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0) 

    print("\n[+] Бот готов! Введите ваш запрос (или 'exit' для выхода).")
    while True:
        query = input("\nВы: ")
        if query.lower() in ['exit', 'quit', 'выход', 'q']:
            break
        if not query.strip(): continue
        
        retrieved_docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE).format(context=context, question=query)
        
        print("\nБот: (думает...)")
        response = llm.invoke(prompt)
        print(f"\n{response.content}")

if __name__ == "__main__":
    run_bot()