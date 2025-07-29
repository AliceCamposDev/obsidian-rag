
import os
from datetime import datetime
import src.utils.utils as utils

config = utils.load_config()

class ChatLogger():
    
    def __init__(self, log_path: str = "./data/history/dev.log"):
        self.log_path = config["chat_log"]["path"]+"/dev.log"
        self.setup_logging()


    def setup_logging(self):
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("Obsidian RAG Assistant Chat History\n")
                f.write(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("-" * 80 + "\n\n")

    def log_interaction(self, query, context ="", answer="", sources=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, "a") as f:
            f.write(f"[{timestamp}] QUERY: {query}\n")
            f.write(f"SOURCES: {', '.join(sources)}\n")
            f.write("CONTEXT:\n")
            f.write(context + "\n\n")
            f.write("RESPONSE:\n")
            f.write(answer + "\n")
            f.write("-" * 80 + "\n\n")