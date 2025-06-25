## 테스트 
#- 모두 relevance yes : 'LLM'
#- 부분적으로 relevance no : 'GPT'
#- 모두 relevance no : 'SSD'

# ============================================================================
# IMPORTS
# ============================================================================
import os
import streamlit as st
from pprint import pprint
from typing import List
from dotenv import load_dotenv

from tavily import TavilyClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon=":orange_heart:",
)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
load_dotenv()

# Tavily API 키를 .env에서 읽어오기
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key:
    os.environ['TAVILY_API_KEY'] = tavily_api_key
else:
    raise ValueError("TAVILY_API_KEY not found in .env file")

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ['OPENAI_API_KEY'] = openai_api_key
else:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# ============================================================================
# GRAPH STATE
# ============================================================================
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        retrieval_attempts: number of attempts to retrieve documents
        hallucination_attempts: number of attempts to check hallucinations
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    retrieval_attempts: int
    hallucination_attempts: int

# ============================================================================
# MODEL AND TOOL INITIALIZATION
# ============================================================================
# Tavily 클라이언트 초기화
tavily = TavilyClient(api_key=tavily_api_key)

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 벡터스토어 및 리트리버 설정
@st.cache_resource
def create_vectorstore():
    urls = [
"http://docs.getwren.ai//",
"http://docs.getwren.ai//cloud/overview",
"http://docs.getwren.ai//oss/overview/introduction",
"https://www.getwren.ai/blog",
"https://wrenaicloud.statuspage.io/",
"https://getwren.ai",
"https://getwren.ai/genbi?utm_source=docs&utm_medium=oss&utm_campaign=vision",
"https://getwren.ai/post/forward-to-2025-powering-the-future-of-enterprise-with-ai-driven-data-intelligence?utm_source=docs&utm_medium=oss&utm_campaign=vision",
"http://docs.getwren.ai//oss/overview/how_wrenai_works",
"http://docs.getwren.ai//oss/overview/cloud_vs_self_host",
"http://docs.getwren.ai//oss/overview/telemetry",
"http://docs.getwren.ai//oss/installation",
"http://docs.getwren.ai//oss/getting_started/sample_data",
"http://docs.getwren.ai//oss/getting_started/own_data",
"http://docs.getwren.ai//oss/concept/security",
"http://docs.getwren.ai//oss/concept/wren_ai_service",
"http://docs.getwren.ai//oss/concept/wren_engine",
"http://docs.getwren.ai//oss/guide/management/org",
"http://docs.getwren.ai//oss/guide/connect/overview",
"http://docs.getwren.ai//oss/guide/boilerplates/overview",
"http://docs.getwren.ai//oss/guide/home/ask",
"http://docs.getwren.ai//oss/guide/modeling/overview",
"http://docs.getwren.ai//oss/guide/knowledge/overview",
"https://wrenai.readme.io/reference/welcome",
"http://docs.getwren.ai//oss/guide/integrations/excel-add-in",
"http://docs.getwren.ai//oss/guide/settings/ds_settings",
"https://getwren.ai/post/fueling-the-next-wave-of-ai-agents-building-the-foundation-for-future-mcp-clients-and-enterprise-data-access?utm_source=docs&utm_medium=oss&utm_campaign=vision",
"http://docs.getwren.ai//oss/engine/concept/what_is_semantics",
"http://docs.getwren.ai//oss/engine/guide/init",
"https://docs.getwren.ai/oss/wren_engine_api",
"http://docs.getwren.ai//oss/ai_service/guide/custom_llm",
    ]
    
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return vectorstore.as_retriever()

retriever = create_vectorstore()

# ============================================================================
# PROMPTS AND CHAINS
# ============================================================================

### Router
router_system = """You are an expert at routing a user question to a vectorstore or web search.
Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks.
You do not need to be stringent with the keywords in the question related to these topics.
Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question.
Return the a JSON with a single key 'datasource' and no premable or explanation. Question to route"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", router_system),
        ("human", "question: {question}"),
    ]
)
question_router = router_prompt | llm | JsonOutputParser()

### Retrieval Grader
retrieval_grader_system = """You are a grader assessing relevance
    of a retrieved document to a user question. If the document contains keywords related to the user question,
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """

retrieval_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retrieval_grader_system),
        ("human", "question: {question}\n\n document: {document} "),
    ]
)
retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()

### Generate
generate_system = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise"""

generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_system),
        ("human", "question: {question}\n\n context: {context} "),
    ]
)
rag_chain = generate_prompt | llm | StrOutputParser()

### Hallucination Grader
hallucination_grader_system = """You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation."""

hallucination_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_grader_system),
        ("human", "documents: {documents}\n\n answer: {generation} "),
    ]
)
hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()

### Answer Grader
answer_grader_system = """You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""

answer_grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_grader_system),
        ("human", "question: {question}\n\n answer: {generation} "),
    ]
)
answer_grader = answer_grader_prompt | llm | JsonOutputParser()

# ============================================================================
# GRAPH NODES
# ============================================================================

def retrieve(state):
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    retrieval_attempts = state.get("retrieval_attempts", 0)
    return {"documents": documents, "question": question}

def relevance_checker(state):
    """
    Determines whether the retrieved documents are relevant to the question
    Only sets web_search flag if ALL documents are not relevant
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    
    # 모든 문서가 관련이 없으면 웹 서치 필요
    if len(filtered_docs) == 0:
        print("---ALL DOCUMENTS NOT RELEVANT - WEB SEARCH NEEDED---")
        web_search = "Yes"
    else:
        print(f"---{len(filtered_docs)} RELEVANT DOCUMENTS FOUND - NO WEB SEARCH NEEDED---")
        web_search = "No"
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def generate_answer(state):
    """
    Generate answer using RAG on retrieved documents
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})

    context_str = "\n".join([d.page_content for d in documents])
    print(f"Generated answer: {generation}")

    # Initialize hallucination attempts if not present
    hallucination_attempts = state.get("hallucination_attempts", 0)
    return {"documents": documents, "question": question, "generation": generation}

def search_tavily(state):
    """
    Web search based based on the question
    """
    print("---WEB SEARCH---")
    print(state)
    question = state["question"]
    documents = None
    if "documents" in state:
      documents = state["documents"]
    retrieval_attempts = state.get("retrieval_attempts", 0)

    # Web search
    docs = tavily.search(query=question)['results']
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    # Append web results to documents
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

def failed_not_relevant(state):
    """
    Handles the case where all documents are not relevant to the question
    """
    print("---FAILED NOT RELEVANT---")
    return {"documents": [], "question": state["question"]}

def failed_hallucination(state):
    """
    Handles the case where the hallucination check fails
    """
    print("---FAILED HALLUCINATION---")
    return {"documents": [], "question": state["question"]}

def regenerate_answer_node(state):
    """
    Regenerate answer
    """
    print("---REGENERATE ANSWER---")
    hallucination_attempts = state.get("hallucination_attempts", 0)
    hallucination_attempts += 1
    state["hallucination_attempts"] = hallucination_attempts
    return generate_answer(state)

# ============================================================================
# CONDITIONAL EDGE FUNCTIONS
# ============================================================================

def decide_to_search_or_generate(state):
    """
    Decide whether to search or generate
    """
    print("---DECISION: SEARCH OR GENERATE---")
    web_search = state["web_search"]
    retrieval_attempts = state.get("retrieval_attempts", 0)

    if web_search == "Yes":
        if retrieval_attempts < 1:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
            return "search_tavily"
        else:
            print("---DECISION: reach to max retrieval attempts, failed not relevant---")
            return "failed_not_relevant"
    else:
        print("---DECISION: GENERATE---")
        return "generate_answer"

def hallucination_checker(state):
    """
    Determines whether the generation is grounded in the document and answers question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# ============================================================================
# WORKFLOW DEFINITION
# ============================================================================

def create_workflow():
    """Create and compile the workflow"""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("relevance_checker", relevance_checker)
    workflow.add_node("search_tavily", search_tavily)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("failed_not_relevant", failed_not_relevant)
    workflow.add_node("failed_hallucination", failed_hallucination)
    workflow.add_node("regenerate_answer", regenerate_answer_node)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "relevance_checker")
    workflow.add_conditional_edges(
        "relevance_checker",
        decide_to_search_or_generate,
        {
            "generate_answer": "generate_answer",
            "search_tavily": "search_tavily",
            "failed_not_relevant": "failed_not_relevant",
        },
    )
    workflow.add_edge("search_tavily", "relevance_checker")
    
    # generate_answer 후에 바로 조건부 엣지로 hallucination_checker 함수 사용
    workflow.add_conditional_edges(
        "generate_answer",
        hallucination_checker,
        {
            "useful": END,
            "not useful": "search_tavily", 
            "not supported": "regenerate_answer",
        },
    )
    workflow.add_conditional_edges(
        "regenerate_answer",
        hallucination_checker,
        {
            "useful": END,
            "not useful": "search_tavily", 
            "not supported": "regenerate_answer",
        },
    )

    return workflow.compile()

# ============================================================================
# STREAMLIT UI (MAIN FUNCTION)
# ============================================================================

def main():
    """Main Streamlit application"""
    # Create workflow
    app = create_workflow()
    
    st.title("RAG Research Assistant powered by OpenAI")
    st.markdown("Ask questions about LLM agents, prompt engineering, and adversarial attacks!")

    # Sidebar
    st.sidebar.header("Settings")
    st.sidebar.markdown("**Model**: GPT-4o-mini")
    st.sidebar.markdown("**Vector Store**: Chroma")
    st.sidebar.markdown("**Web Search**: Tavily")

    # Main input
    input_question = st.text_input(
        "Enter your question",
        value="What is prompt engineering?",
        placeholder="Ask about LLM agents, prompt engineering, or adversarial attacks..."
    )

    col1, col2 = st.columns([1, 4])
    
    with col1:
        generate_report = st.button("🔍 Generate Answer", type="primary")
    
    with col2:
        if st.button("🔄 Clear"):
            st.rerun()

    # Generate answer
    if generate_report and input_question:
        with st.spinner("🤔 Thinking... This may take a moment."):
            try:
                # Create progress placeholder
                progress_placeholder = st.empty()
                result_placeholder = st.empty()
                
                inputs = {"question": input_question}
                steps = []
                
                for output in app.stream(inputs):
                    for key, value in output.items():
                        steps.append(f"✅ Completed: {key}\n")
                        progress_placeholder.markdown("**Progress:**\n" + "\n".join(steps))
                
                final_answer = value.get("generation", "Sorry, I couldn't generate an answer.")
                
                # Clear progress and show result
                progress_placeholder.empty()
                
                st.success("✨ Answer Generated!")
                st.markdown("### 📝 Answer")
                st.markdown(final_answer)
                
                # Show sources if available
                if "documents" in value and value["documents"]:
                    with st.expander("📚 Sources Used"):
                        for i, doc in enumerate(value["documents"][:3]):  # Show top 3 sources
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                            st.markdown("---")
                            
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.markdown("Please try again with a different question.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Test Questions")
    st.sidebar.markdown("- What is prompt engineering?")
    st.sidebar.markdown("- How do LLM agents work?")
    st.sidebar.markdown("- What are adversarial attacks?")
    st.sidebar.markdown("- What is RAG?")
    
    if st.sidebar.button("🔄 Restart App"):
        st.session_state.clear()
        st.rerun()

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()