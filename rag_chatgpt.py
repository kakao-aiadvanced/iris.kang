## 테스트 
#- 모두 relevance yes : 'agent memory'
#- 부분적으로 relevance no : 'GPT'
#- 모두 relevance no : 'SSD'
import getpass
import os
import sys
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests

# Load environment variables
load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini")
def format_docs(docs):
    """문서 리스트를 하나의 문자열로 포맷"""
    return "\n\n".join(doc.page_content for doc in docs)

# JSON 파서 초기화
parser = JsonOutputParser()

# 관련성 평가 프롬프트
relevance_prompt = PromptTemplate(
    template="""You are an expert information retrieval evaluator.

Your task:
Decide if the retrieved chunk is relevant to the user's query.

Relevance means:
- The chunk contains information that directly answers or is helpful for the user's query.
- The chunk is contextually or semantically related to the user's query.
- Both explicit mentions and context-based connections count.

Instructions:
- Read the user query.
- Read the retrieved chunk.
- Respond with JSON only, using the following format:
    {{ "relevance": "yes" }} if the chunk is relevant
    {{ "relevance": "no" }} if the chunk is not relevant

User Query:
{query}

Retrieved Chunk:
{chunk}

{format_instructions}
""",
    input_variables=["query", "chunk"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# hallucination 평가 프롬프트
hallucination_prompt = PromptTemplate(
    template="""You are an expert fact-checking evaluator.

Your task:
Examine the Generated Answer and compare it to the Source Context. 
Decide if the Generated Answer contains any hallucinated information.

Definition of Hallucination:
- The Generated Answer contains information, claims, or details that are NOT present, contradicted, or unsupported by the Source Context.
- Minor rephrasing of facts is acceptable, but adding unverified or incorrect details is a hallucination.

Instructions:
- If any part of the Generated Answer is NOT supported by the Source Context, respond with:
    {{"hallucination": "yes"}}
- If the Generated Answer is fully supported by the Source Context, respond with:
    {{"hallucination": "no"}}

Source Context:
{source}

Generated Answer:
{answer}

{format_instructions}
""",
    input_variables=["source", "answer"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 평가 체인 생성
relevance_chain = relevance_prompt | llm | parser
hallucination_chain = hallucination_prompt | llm | parser

# 평가 함수들 (체인 정의 후)
def evaluate_chunk_relevance(query, chunk):
    """Evaluate relevance between query and chunk"""
    try:
        result = relevance_chain.invoke({
            "query": query, 
            "chunk": chunk.page_content
        })
        return result
    except Exception as e:
        print(f"Error evaluating relevance: {e}")
        return {"relevance": "error"}

def evaluate_hallucination(source_context, generated_answer):
    """Evaluate if the generated answer contains hallucinations"""
    try:
        result = hallucination_chain.invoke({
            "source": source_context,
            "answer": generated_answer
        })
        return result
    except Exception as e:
        print(f"Error evaluating hallucination: {e}")
        return {"hallucination": "error"}

# 로딩할 URL 정의
url_list = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# 모든 URL에서 문서 로드
loader = WebBaseLoader(
    web_paths=url_list,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 문서 청크 생성
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
splits = text_splitter.split_documents(docs)

# OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Chroma 벡터 저장소 생성
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# retriever 생성
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

# RAG 프롬프트 및 체인 생성
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
def main():
    # 명령어 라인에서 쿼리 받기
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])  # 여러 단어 쿼리 지원
    else:
        test_query = input("쿼리를 입력하세요: ")

    # RAG 시스템 평가
    print("1. RAG 기반 관련성 평가")
    print("="*60)

    print(f"Query: '{test_query}'")
    print("-" * 40)

    # 청크 검색
    retrieved_docs = retriever.invoke(test_query)
    print(f"Retrieved {len(retrieved_docs)} chunks")

    # 각 청크의 관련성 평가
    print("\n### RELEVANCE EVALUATION RESULTS  ###")
    relevant_chunks = []

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"\n[Chunk {i}:]")
        #print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        #print(f"Content preview: {doc.page_content[:150]}...")
        
        # Evaluate relevance
        relevance_result = evaluate_chunk_relevance(test_query, doc)
        print(f"Relevance: {relevance_result.get('relevance')}")
        
        # Keep relevant chunks
        if relevance_result.get('relevance') == 'yes':
            relevant_chunks.append(doc)
        
        print("-" * 40)

    print(f"\nSummary: {len(relevant_chunks)} out of {len(retrieved_docs)} 청크가 관련성이 있습니다.")

    # 관련 청크로 답변 생성
    if relevant_chunks:
        print(f"\n{len(relevant_chunks)} 개의 청크를 사용해 답변 생성중입니다...\n")
        relevant_context = format_docs(relevant_chunks)
        
        # 필터링된 컨텍스트로 RAG 체인 생성
        filtered_rag_result = rag_chain.invoke(test_query)
        print(f"답변: {filtered_rag_result}")
        
        # hallucination 평가
        print("\n2. HALLUCINATION 평가")
        print("="*60)
        
        hallucination_result = evaluate_hallucination(relevant_context, filtered_rag_result)
        
        if hallucination_result.get('hallucination') == 'error':
            print("오류: Hallucination 평가 중 문제가 발생했습니다.")
        else:
            print(f"Hallucination: {hallucination_result}")
            
            # hallucination 감지 시 1회 재시도
            if hallucination_result.get('hallucination') == 'yes':
                print("\nHallucination 감지! 답변을 다시 생성합니다...")
                
                # 답변 생성 재시도
                retry_rag_result = rag_chain.invoke(test_query)
                print(f"재생성된 답변: {retry_rag_result}")
                
                # 재시도 답변의 hallucination 평가
                print("\n재생성된 답변 HALLUCINATION 평가:")
                retry_hallucination_result = evaluate_hallucination(relevant_context, retry_rag_result)
                
                if retry_hallucination_result.get('hallucination') == 'error':
                    print("오류: 재시도 Hallucination 평가 중 문제가 발생했습니다.")
                else:
                    print(f"재시도 Hallucination: {retry_hallucination_result}")
                    
                    if retry_hallucination_result.get('hallucination') == 'yes':
                        print("재시도에도 hallucination이 감지되었습니다.")
                    else:
                        print("재생성된 답변은 hallucination이 없습니다.")
            
    else:
        print("\n관련 있는 청크가 없습니다.")

if __name__ == "__main__":
    main()
