from phi.agent import Agent
from phi.model.openai import OpenAIChat, OpenAILike
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb, SearchType
from phi.playground import Playground, serve_playground_app
from phi.tools.duckduckgo import DuckDuckGo

from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
embeddings = SentenceTransformerEmbedder().get_embedding("The quick brown fox jumps over the lazy dog.")

import os

llm_client = OpenAILike(
    name="glm",
    model="glm-4-9b",
    api_key=os.getenv("ZHIPUAI_API_TOKEN"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

db_uri = "tmp/lancedb"
# Create a knowledge base from a PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use LanceDB as the vector database
    vector_db=LanceDb(table_name="recipes", uri=db_uri, search_type=SearchType.vector, embedder=SentenceTransformerEmbedder()),
)
# Load the knowledge base: Comment out after first run
knowledge_base.load(upsert=True)

rag_agent = Agent(
    # model=OpenAIChat(id="gpt-4o"),
    model=llm_client,
    agent_id="rag-agent",
    knowledge=knowledge_base, # Add the knowledge base to the agent
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[rag_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("rag_agent_dev:app", reload=True, host="0.0.0.0")