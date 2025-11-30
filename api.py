import os
import html
import re  # Importamos REGEX para caçar o lead no texto
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from fastapi.responses import Response

# Imports do LangChain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import FastEmbedEmbeddings # Mantendo o leve
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

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_conversa = None

def carregar_bot():
    print("--- INICIANDO SERVIDOR VENDEDOR ---")
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

    print("2. Processando Embeddings (FastEmbed)...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_processados = text_splitter.split_documents(documentos)
    
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(docs_processados, embeddings)
    retriever = vectorstore.as_retriever()
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2) # Temp baixa para ser preciso

    # --- PARTE 1: Reformulador ---
    contextualize_q_system_prompt = """
    Dado um histórico de chat e a última pergunta do usuário,
    reformule a pergunta para que ela possa ser entendida sozinha.
    """
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- PARTE 2: O VENDEDOR (Aqui está a mágica) ---
    qa_system_prompt = """
    Você é um Consultor de Vendas Especialista.
    Sua missão é tirar dúvidas baseadas no contexto e CONSEGUIR O CONTATO do cliente.

    REGRAS DE OURO:
    1. Use APENAS o contexto abaixo.
    2. Se o cliente perguntar preço ou detalhes, responda e termine perguntando: "Gostaria de agendar uma demonstração? Qual seu nome?"
    3. SE O CLIENTE FORNECER O NOME OU TELEFONE/EMAIL:
       - Agradeça e diga que um consultor vai entrar em contato.
       - NO FINAL DA MENSAGEM (pule uma linha), escreva ESTRITAMENTE neste formato:
       LEAD_CAPTURADO: [Nome do Cliente] | [Dado de Contato] | [Interesse Resumido]

    Contexto:
    {context}
    """
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    print("3. Bot Vendedor Pronto! 💰")
    return conversational_rag_chain

chain_conversa = carregar_bot()

@app.post("/chat")
def conversar(Body: str = Form(...), From: str = Form(...)):
    if not chain_conversa:
        return Response(content="Erro: Sem documentos.", media_type="text/plain")
    
    print(f"📩 De: {From} | Diz: {Body}")
    
    try:
        resultado = chain_conversa.invoke(
            {"input": Body},
            config={"configurable": {"session_id": From}}
        )
        texto_resposta = resultado['answer']
        
        # --- LÓGICA DE CAPTURA DE LEAD (Espionagem) ---
        # Verifica se o bot soltou a "bandeira" de lead capturado
        if "LEAD_CAPTURADO:" in texto_resposta:
            # 1. Extrai o que vem depois dos dois pontos
            match = re.search(r"LEAD_CAPTURADO:(.*)", texto_resposta)
            if match:
                dados_lead = match.group(1).strip()
                print(f"💰💰💰 NOVO LEAD DETECTADO: {dados_lead} 💰💰💰")
                print(f"Salvando no banco de dados (simulado)...")
                # AQUI entraria o código para salvar no Google Sheets ou Excel
            
            # 2. Limpa a mensagem para o cliente não ver o código interno
            texto_resposta = texto_resposta.replace(match.group(0), "").strip()

        # Corte de segurança e limpeza
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
    return {"status": "Bot Vendedor Online 💰"}