import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("start")
from src.rag_core import RAGSystem
print ("rag imported")
import time
from src.utils.chat_logger import ChatLogger
import src.utils.utils as utils

config = utils.load_config()

chat_logger = ChatLogger()

generate_emb_str = input("Generate new embeddings[y/n]: ")
gen_enb_bool = False
if generate_emb_str == "y":
    gen_enb_bool = True

vault_path = config["general"]["vault_path"]

rag_system = RAGSystem(
    vault_path=vault_path,
    update_vault = gen_enb_bool
)

session_id = f"session_{int(time.time())}"
print("Chat initialized, enter 'exit' to leave.")

while True:
    query = input("\nAlice: ")
    if query.lower() == 'exit':
        break
    
    response, context = rag_system.process_query(session_id, query)
    print(f"\nAI: {response}")
    
    chat_logger.log_interaction(query, context = context, answer = response)