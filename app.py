import os
import sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_core import LocalRAGSystem
import time
import json
from datetime import datetime

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()


LOG_FILE = "data/chat_history.log"

def setup_logging():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Obsidian RAG Assistant Chat History\n")
            f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")

def log_interaction(query, context ="", answer="", sources=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] QUERY: {query}\n")
        f.write(f"SOURCES: {', '.join(sources)}\n")
        f.write("CONTEXT:\n")
        f.write(context + "\n\n")
        f.write("RESPONSE:\n")
        f.write(answer + "\n")
        f.write("-" * 80 + "\n\n")


generate_emb_str = input("Generate new embeddings[y/n]: ")
gen_enb_bool = False
if generate_emb_str == "y":
    gen_enb_bool = True

vault_path = config["general"]["vault_path"]
rag_system = LocalRAGSystem(
    vault_path=vault_path,
    update_vault = gen_enb_bool
)

session_id = f"session_{int(time.time())}"
setup_logging()
print("Chat initialized, enter 'exit' to leave.")

while True:
    query = input("\nAlice: ")
    if query.lower() == 'exit':
        break
    
    response, context = rag_system.process_query(session_id, query)
    print(f"\nAI: {response}")
    
    log_interaction(query, context = context, answer = response)