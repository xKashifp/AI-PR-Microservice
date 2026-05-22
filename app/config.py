from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    # LLM
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-70b-8192"
    GEMINI_API_KEY: str = ""

    # Web3
    ALCHEMY_RPC_URL: str = "https://eth-mainnet.g.alchemy.com/v2/demo"
    ETHERSCAN_API_KEY: str = ""

    # Slack
    SLACK_WEBHOOK_URL: str = ""

    # Paths
    FAISS_INDEX_DIR: str = "./data/faiss_index"
    SQLITE_DB_PATH: str = "./data/pr_intelligence.db"
    MODEL_DIR: str = "./ml/models"

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Sentiment model
    SENTIMENT_MODEL: str = "distilbert-base-uncased-finetuned-sst-2-english"

    model_config = ConfigDict(env_file=".env")

settings = Settings()
