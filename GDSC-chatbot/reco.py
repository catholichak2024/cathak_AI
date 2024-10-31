from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from operator import itemgetter
import asyncio
import warnings
warnings.filterwarnings("ignore")

import faiss
import pickle
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()


class args:
    prompt = "./prompt.txt"
    embed_model = "BAAI/bge-m3"
    main_model = "gpt-3.5-turbo"

def start_point():
    global store, chain, rag_with_history
    index = faiss.read_index('GDSC_index_fin.bin')

    with open("Retriver/GDSC_store_fin.pkl", "rb") as f:
        store_data = pickle.load(f)

    embeddings_model = HuggingFaceEmbeddings(
            model_name=args.embed_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )

    vectorstore = FAISS(
        index=index,
        docstore=store_data["docstore"],
        index_to_docstore_id=store_data["index_to_docstore_id"],
        embedding_function=embeddings_model
    )

    retriever = vectorstore.as_retriever(score_threshold=0.7,
                                         return_source_documents=False,
                                         search_kwargs={"k": 5}
                                         )

    with open(args.prompt, "r", encoding='utf-8') as f:
        prompt_template = f.read()

    prompt = PromptTemplate(
        input_variables=["before_input", "before_response", "current_input"],
        template=prompt_template
    )

    model = ChatOpenAI(
        model=args.main_model,
        max_tokens=2048,
        temperature=0,
    )

    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    ## 세션 기록을 저장할 딕셔너리
    store = {}

    # 세션 ID를 기반으로 세션 기록을 가져오는 함수
    def get_session_history(session_ids):
        if session_ids not in store:  # 세션 ID가 store에 없는 경우
            # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
            store[session_ids] = ChatMessageHistory()
        return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

    # # 대화를 기록하는 RAG 체인 생성
    rag_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )


def go(message):
    result = rag_with_history.invoke(
        {"question": message},
        config={"configurable": {"session_id": 'rag123'}}
    )
    return result


if __name__ == "__main__":
    start_point()
    print(go('안녕 리코야 나는 사회복지학과인데 교양 추천좀 해줘'))
    print('================')





