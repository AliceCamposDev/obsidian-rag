
import os
from datetime import datetime
import src.utils.utils as utils

config = utils.load_config()

class ChatLogger():
    
    def __init__(self, log_path: str = "./data/history/dev.log"):
        self.log_path = config["chat_log"]["path"]+"/dev.log"
        self.setup_logging()


    def setup_logging(self) -> None:
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("Obsidian RAG Assistant Chat History\n")
                f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")

    
    def log_interaction(self, query: str, context: str, answer: str) -> None:
        with open(self.log_path, "a", encoding='utf-8', errors='replace') as f:
            f.write("Query: " + query + "\n")
            f.write(context + "\n\n")
            f.write("Answer: " + answer + "\n")
            f.write("-" * 50 + "\n")