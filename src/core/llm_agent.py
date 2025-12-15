import requests
import json
import re
from src.core.database import DatabaseManager

class LLMAgent:
    def __init__(self, model="dolphin3:latest"):
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.db = DatabaseManager()
        self.context_schema = """
        Table Name: crossing_events
        Columns:
        - id (INT, Primary Key)
        - timestamp (DATETIME, when the event happened)
        - location (VARCHAR, name of the camera location, e.g. 'Main Entrance')
        - line_name (VARCHAR, specific line crossed)
        - count_left (INT, people going left/in)
        - count_right (INT, people going right/out)
        - video_id (VARCHAR, unique ID of the camera source)
        - clothing_color (VARCHAR, detected shirt color)
        
        The 'count_left' and 'count_right' usually represent one person crossing.
        To get total people validation, usually SUM(count_left + count_right).
        """
        self.messages = [] # History: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    def _call_ollama(self, prompt, stream=False):
        try:
            # We construct a full prompt for completion models or use chat API if available.
            # For simplicity, we stick to the generate API but prepend context.
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1} # Lower temp for SQL
            }
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            return f"Error connecting to AI: {e}"

    def ask(self, user_question):
        # Step 1: Decision & SQL Generation
        # We provide recent history context for reference (last 2 turns)
        history_context = ""
        if self.messages:
            recent = self.messages[-2:]
            history_context = "Conversation History:\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

        SYSTEM_RULES = """
        You are a MySQL Database Expert.
        You have FULL PERMISSION to query the 'crossing_events' table.
        
        Goal: Answer user questions by writing SQL queries.
        
        Rules:
        1. If the user asks for data/stats/counts -> Write a SELECT query.
        2. If the user says "Hello" or General Chat -> Return 'NO_SQL'.
        3. Do NOT refuse to answer data questions. 
        4. Output ONLY the SQL (no markdown, no explanations).
        """

        sql_prompt = f"""
        {SYSTEM_RULES}
        
        Schema:
        {self.context_schema}

        {history_context}
        
        User: "{user_question}"
        
        Make a decision:
        - If data is needed: Write the SQL (e.g. SELECT count(*) FROM crossing_events...)
        - If just chat: Write NO_SQL
        """
        
        response_1 = self._call_ollama(sql_prompt)
        cleaned_response = re.sub(r"```sql|```", "", response_1).strip()
        
        final_answer = ""
        
        # Check if NO_SQL
        if "NO_SQL" in cleaned_response.upper() or len(cleaned_response) < 10:
             # Just chat
             chat_prompt = f"""
             You are a helpful assistant for a People Counting Traffic system.
             {history_context}
             User: "{user_question}"
             Reply helpfully and concisely.
             """
             final_answer = self._call_ollama(chat_prompt)
        else:
            # It's SQL
            print(f"[AI Agent] Generated SQL: {cleaned_response}")
            db_result = self.db.execute_safe_query(cleaned_response)
            
            if db_result is None:
                 final_answer = "Sorry, I couldn't execute that query. It might be unsafe or valid data wasn't found."
            elif isinstance(db_result, str):
                 final_answer = f"Database Error: {db_result}"
            else:
                # Step 3: Synthesize Answer
                result_preview = str(db_result)[:2000] 
                final_prompt = f"""
                User Question: "{user_question}"
                Generated SQL: {cleaned_response}
                Database Result: {result_preview}
                
                Task: Answer the user's question based strictly on the database result above.
                - Be concise and friendly.
                - If the result is empty, say no data found.
                """
                final_answer = self._call_ollama(final_prompt)
        
        # update history
        self.messages.append({"role": "user", "content": user_question})
        self.messages.append({"role": "assistant", "content": final_answer})
        
        return final_answer
