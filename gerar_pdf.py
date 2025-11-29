from reportlab.pdfgen import canvas

def criar_pdf_teste():
    c = canvas.Canvas("manual.pdf")
    
    # Título
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 800, "Manual da Cafeteira Quântica 3000")
    
    # Corpo do texto
    c.setFont("Helvetica", 12)
    texto = [
   "Bem-vindo ao futuro do café.",
        "--------------------------------------------------",
        "1. ESPECIFICAÇÕES TÉCNICAS:",
        "   - Voltagem: Bivolt Automático (110v / 220v).", # <--- ADICIONAMOS ISSO
        "   - Potência: 5000 Watts (cuidado ao ligar).",
        "",
        "2. PREÇO:",
        "   O modelo básico custa R$ 5.999,00.",
        "   O modelo Titanium custa R$ 8.500,00 (incluso seguro contra explosões).",
        "",
        "3. INSTRUÇÕES DE SEGURANÇA:",
        "   NUNCA coloque água pesada no reservatório.",
        "   Se a luz vermelha piscar, corra para as colinas.",
        "",
        "4. CONTATO DO SUPORTE:",
        "   Email: socorro@cafequantico.com",
        "   Telefone: 0800-CAFE-DOIDO"
    ]
    
    y = 750
    for linha in texto:
        c.drawString(100, y, linha)
        y -= 20
        
    c.save()
    print("✅ Sucesso! O arquivo 'manual.pdf' foi criado na pasta.")

if __name__ == "__main__":
    criar_pdf_teste()