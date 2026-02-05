import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    print("Healthcare Policy RAG Assistant is running...")


if __name__ == "__main__":
    print("Starting Healthcare Policy RAG Assistant...")
    main()
