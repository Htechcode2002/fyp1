from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                               QLineEdit, QPushButton, QLabel, QScrollArea, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from src.core.llm_agent import LLMAgent

class AIWorker(QThread):
    finished = Signal(str, str) # question, answer

    def __init__(self, agent, question):
        super().__init__()
        self.question = question
        self.agent = agent

    def run(self):
        # Use the passed agent which holds history
        answer = self.agent.ask(self.question)
        self.finished.emit(self.question, answer)

class AIChatWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.agent = LLMAgent() # Use default model (dolphin3:latest)
        self.setFixedHeight(400) # Give it some height
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Header
        lbl_title = QLabel("🤖 AI Data Analyst (Powered by Qwen)")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #334155;")
        self.layout.addWidget(lbl_title)
        
        # Chat History
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setStyleSheet("""
            QTextEdit {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
                font-size: 13px;
                color: #1e293b;
            }
        """)
        self.layout.addWidget(self.history)
        
        # Input Area
        input_layout = QHBoxLayout()
        
        self.txt_input = QLineEdit()
        self.txt_input.setPlaceholderText("Ask about your data (e.g. 'How many people crossed yesterday?')")
        self.txt_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                color: #0f172a;
            }
        """)
        self.txt_input.returnPressed.connect(self.send_question)
        
        self.btn_send = QPushButton("Ask AI")
        self.btn_send.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; 
                color: white; 
                border-radius: 6px; 
                padding: 8px 16px; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:disabled { background-color: #94a3b8; }
        """)
        self.btn_send.clicked.connect(self.send_question)
        
        input_layout.addWidget(self.txt_input)
        input_layout.addWidget(self.btn_send)
        
        self.layout.addLayout(input_layout)
        
        # Initial Message
        self.append_message("System", "Ready! Ask me anything about your traffic data.")

    def append_message(self, sender, text):
        color = "#0f172a" if sender == "You" else "#059669" # Green for AI
        formatted = f"<p><b style='color:{color}'>{sender}:</b> {text}</p>"
        self.history.append(formatted)
        
    def send_question(self):
        question = self.txt_input.text().strip()
        if not question:
            return
            
        self.append_message("You", question)
        self.txt_input.clear()
        self.txt_input.setDisabled(True)
        self.btn_send.setDisabled(True)
        self.btn_send.setText("Thinking...")
        
        # Worker
        self.worker = AIWorker(self.agent, question)
        self.worker.finished.connect(self.handle_response)
        self.worker.start()
        
    def handle_response(self, question, answer):
        self.append_message("AI", answer)
        self.txt_input.setDisabled(False)
        self.btn_send.setDisabled(False)
        self.btn_send.setText("Ask AI")
        self.txt_input.setFocus()
