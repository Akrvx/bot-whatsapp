import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# Adicionamos AIMessage para representar a fala da IA
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# --- CONFIGURAÇÃO DA MEMÓRIA ---
# A lista começa com a personalidade do bot
historico_conversa = [
    SystemMessage(content="Você é um assistente sênior em programação. Ajude com respostas diretas e exemplos de código.")
]

def conversar():
    print("--- Chatbot com Memória Iniciado (Digite 'sair' para encerrar) ---")
    
    while True:
        texto_usuario = input("\nVocê: ")
        if texto_usuario.lower() == 'sair':
            break
        
        # 1. Adiciona o que você disse ao histórico
        historico_conversa.append(HumanMessage(content=texto_usuario))
        
        print("🤖 ...")
        
        # 2. Envia O HISTÓRICO INTEIRO para a IA, não apenas a última frase
        resposta_ai = llm.invoke(historico_conversa)
        
        # 3. Mostra a resposta
        print(f"Bot: {resposta_ai.content}")
        
        # 4. Salva a resposta da IA no histórico também (para ela lembrar o que ela mesma disse)
        historico_conversa.append(AIMessage(content=resposta_ai.content))

if __name__ == "__main__":
    conversar()