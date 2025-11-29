import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# MUDANÇA AQUI: Usamos embeddings locais
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# O Chat continua sendo o Gemini (Nuvem)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

print("--- 1. Carregando e processando conhecimento... ---")

try:
    loader = TextLoader("conhecimento.txt", encoding="utf-8")
    documentos = loader.load()
except Exception as e:
    print(f"ERRO: {e}")
    exit()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs_processados = text_splitter.split_documents(documentos)

# MUDANÇA AQUI: O processamento dos números agora é no seu PC (Grátis e Ilimitado)
print("--- Gerando vetores localmente... ---")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs_processados, embeddings)

print("--- 2. Cérebro atualizado! Iniciando chat... ---")

template = """
Você é um assistente corporativo.
Use APENAS o seguinte contexto para responder à pergunta.
Se a resposta não estiver no contexto, diga "Sinto muito, essa informação não consta na minha base de conhecimento".

Contexto:
{context}

Pergunta:
{input}
"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

def conversar():
    print("Digite 'sair' para encerrar.")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() == 'sair':
            break 
        print("🤖 Consultando base de dados...")
        resposta = retrieval_chain.invoke({"input": pergunta})
        print(f"Bot: {resposta['answer']}")

if __name__ == "__main__":
    conversar()