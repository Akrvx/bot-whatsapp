import os
import html
import re  # Importamos REGEX para caçar o lead no texto
import csv
import datetime
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

    # --- PARTE 2: O VENDEDOR (Versão Corrigida) ---
    qa_system_prompt = """
    Você é um Gerente de Vendas da Cafeteira Quântica.
    Sua missão é convencer o cliente e PEGAR OS DADOS DELE.

    DIRETRIZES:
    1. Para dúvidas técnicas (preço, voltagem, garantia), use OBRIGATORIAMENTE o contexto abaixo.
    2. Para VENDAS e CADASTRO, você tem autonomia total. NÃO diga que não pode processar.
    3. Se o cliente disser "Eu quero" ou passar o contato:
       - Aja como se já tivesse anotado o pedido.
       - Parabenize pela ótima escolha.
       - Diga que a equipe financeira vai ligar em breve para finalizar.
    
    GATILHO DE SISTEMA (IMPORTANTE):
    Se o cliente fornecer NOME e TELEFONE, você DEVE escrever no final da sua resposta (em uma nova linha):
    LEAD_CAPTURADO: [Nome] | [Telefone] | [Intenção de Compra]

    Contexto Técnico do Produto:
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
        if "LEAD_CAPTURADO:" in texto_resposta:
            match = re.search(r"LEAD_CAPTURADO:(.*)", texto_resposta)
            if match:
                dados_lead = match.group(1).strip()
                partes = dados_lead.split("|")
                
                # Garante que temos 3 partes
                if len(partes) < 3:
                    partes = [dados_lead, "N/A", "N/A"]
                
                print(f"💰💰 NOVO LEAD: {dados_lead}")
                
                # --- SALVANDO EM CSV ---
                arquivo_leads = "leads.csv"
                existe = os.path.exists(arquivo_leads)
                
                try:
                    with open(arquivo_leads, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not existe:
                            writer.writerow(["Data", "Nome", "Contato", "Interesse"])
                        
                        agora = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        writer.writerow([agora, partes[0].strip(), partes[1].strip(), partes[2].strip()])
                    print("✅ Salvo no arquivo leads.csv com sucesso!")
                except Exception as e:
                    print(f"Erro ao salvar CSV: {e}")
                
                # 2. Limpa a mensagem aqui dentro, só se achou o match
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