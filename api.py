import os
import html
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import Response

# --- Imports de Memória e Histórico ---
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
app = FastAPI()

# Armazena o histórico de cada número de telefone na memória RAM
# (Se reiniciar o servidor, apaga. Para produção, usaríamos Redis)
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_conversa = None

def carregar_bot():
    print("--- INICIANDO SERVIDOR COM MEMÓRIA ---")
    caminho_pasta = "documentos"
    
    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)
        return None
        
    print(f"1. Lendo PDFs da pasta '{caminho_pasta}'...")
    try:
        loader = PyPDFDirectoryLoader(caminho_pasta)
        documentos = loader.load()
        if not documentos:
            return None
    except Exception as e:
        print(f"ERRO: {e}")
        return None

    print("2. Processando Embeddings...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_processados = text_splitter.split_documents(documentos)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs_processados, embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # --- PARTE 1: REFORMULADOR DE PERGUNTAS (Contextualização) ---
    # Isso faz o bot entender "E o preço?" baseado na frase anterior
    contextualize_q_system_prompt = """
    Dado um histórico de chat e a última pergunta do usuário (que pode não ter contexto),
    reformule a pergunta para que ela possa ser entendida sozinha.
    NÃO responda a pergunta, apenas reformule-a se necessário. Caso contrário, retorne-a como está.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- PARTE 2: A RESPOSTA FINAL (QA) ---
    qa_system_prompt = """
    Você é um assistente comercial. Use o contexto abaixo para responder.
    Se não souber, diga que não sabe.
    Mantenha a resposta com no máximo 500 caracteres.
    
    Contexto:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"), # Aqui entra o histórico
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # --- PARTE 3: GERENCIADOR DE SESSÃO ---
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    print("3. Bot com Memória Pronto! 🧠")
    return conversational_rag_chain

chain_conversa = carregar_bot()

@app.post("/chat")
# Adicionamos o parâmetro 'From' que o Twilio envia (é o número do zap)
def conversar(Body: str = Form(...), From: str = Form(...)):
    if not chain_conversa:
        return Response(content="Erro: Sem documentos.", media_type="text/plain")
    
    print(f"📩 De: {From} | Diz: {Body}")
    
    try:
        # Passamos o 'session_id' com o número da pessoa
        resultado = chain_conversa.invoke(
            {"input": Body},
            config={"configurable": {"session_id": From}}
        )
        texto_resposta = resultado['answer']
        
        # Corte de segurança
        if len(texto_resposta) > 1500:
            texto_resposta = texto_resposta[:1500] + "..."
            
        print(f"📤 Respondendo: {texto_resposta}")
        
        texto_seguro = html.escape(texto_resposta)
        xml_resposta = f"""<?xml version="1.0" encoding="UTF-8"?>
        <Response>
            <Message>{texto_seguro}</Message>
        </Response>"""
        
        return Response(content=xml_resposta, media_type="application/xml")
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return Response(content="Erro no processamento.", media_type="text/plain")

@app.get("/")
def status():
    return {"status": "Bot com Memória Online 🧠"}