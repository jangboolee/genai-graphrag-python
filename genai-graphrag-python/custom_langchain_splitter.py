import asyncio
import os

from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# You will need to install langchain-text-splitters: pip install langchain-text-splitters
from neo4j_graphrag.experimental.components.text_splitters.langchain import (
    LangChainTextSplitterAdapter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM

load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
)
neo4j_driver.verify_connectivity()

llm = OpenAILLM(
    model_name="gpt-5-nano", model_params={"reasoning_effort": "minimal"}
)
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

splitter = LangChainTextSplitterAdapter(
    CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500,
        chunk_overlap=100,
    )
)

kg_builder = SimpleKGPipeline(
    llm=llm,
    driver=neo4j_driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    embedder=embedder,
    from_pdf=True,
    text_splitter=splitter,
)

pdf_file = "./genai-graphrag-python/data/genai-fundamentals_1-generative-ai_1-what-is-genai.pdf"

print(f"Processing {pdf_file}")
result = asyncio.run(kg_builder.run_async(file_path=pdf_file))
print(result.result)
