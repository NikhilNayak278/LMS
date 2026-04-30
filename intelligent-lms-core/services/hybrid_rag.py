import os
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

# Pydantic models for Knowledge Graph Extraction
class Concept(BaseModel):
    name: str = Field(description="Name of the concept (lowercase, snake_case or short phrase)")
    description: str = Field(description="Brief description of the concept")

class Relationship(BaseModel):
    source_concept: str
    target_concept: str
    relationship_type: str = Field(description="Must be PREREQUISITE_OF or RELATED_TO")

class KnowledgeGraphExtraction(BaseModel):
    concepts: List[Concept]
    relationships: List[Relationship]

# Initialize Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

# Initialize DataStax Astra DB Vector Store
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT", "")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")

vector_store = None
if ASTRA_DB_API_ENDPOINT and ASTRA_DB_API_ENDPOINT != "...":
    try:
        vector_store = AstraDBVectorStore(
            embedding=embeddings,
            collection_name="lms_documents",
            api_endpoint=ASTRA_DB_API_ENDPOINT,
            token=ASTRA_DB_APPLICATION_TOKEN,
        )
        print("Connected to DataStax Astra DB.")
    except Exception as e:
        print(f"Error connecting to Astra DB: {e}")

# Initialize Neo4j Graph
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

graph_db = None
if NEO4J_URI and NEO4J_URI != "...":
    try:
        graph_db = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        print("Connected to Neo4j.")
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")

def initialize_neo4j_schema():
    if graph_db:
        try:
            graph_db.query("CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
            print("Neo4j schema initialized.")
        except Exception as e:
            print(f"Error initializing Neo4j schema: {e}")

# LLM for Extraction
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

def ingest_document(text: str, source_name: str):
    """Processes uploaded text, stores chunks in Vector DB, and extracts graph to Neo4j."""
    
    # 1. Store in Vector DB (DataStax)
    if vector_store:
        # Simple chunking (for production, use Langchain's RecursiveCharacterTextSplitter)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        chunks = [c.strip() for c in chunks if c.strip()]
        
        if chunks:
            success_count = 0
            for c in chunks:
                try:
                    doc = Document(page_content=c, metadata={"source": source_name})
                    vector_store.add_documents([doc])
                    success_count += 1
                except Exception as e:
                    print(f"Failed to add chunk to Vector Store. Error: {e}")
            print(f"Successfully added {success_count} out of {len(chunks)} chunks to Vector Store.")
        else:
            print("No valid text chunks to insert into Vector Store.")
    else:
        print("Vector store not configured. Skipping vector insertion.")

    # 2. Extract and Store Knowledge Graph (Neo4j)
    if graph_db:
        print("Extracting knowledge graph from text...")
        start_llm = time.time()
        structured_llm = llm.with_structured_output(KnowledgeGraphExtraction)
        prompt = f"Extract key programming/educational concepts and their prerequisite relationships from the following text:\n\n{text[:5000]}" # Limit text for extraction
        
        try:
            kg: KnowledgeGraphExtraction = structured_llm.invoke(prompt)
            llm_duration = time.time() - start_llm
            print(f"LLM Extraction took {llm_duration:.2f}s")
            
            start_neo4j = time.time()
            
            # Insert into Neo4j
            concepts_data = [{"name": c.name, "description": c.description} for c in kg.concepts]
            if concepts_data:
                graph_db.query(
                    "UNWIND $data AS item MERGE (c:Concept {name: toLower(item.name)}) SET c.description = item.description",
                    {"data": concepts_data}
                )
            
            relationships_data = [{"source_concept": r.source_concept, "target_concept": r.target_concept, "relationship_type": r.relationship_type} for r in kg.relationships]
            if relationships_data:
                graph_db.query(
                    "UNWIND $data AS item MATCH (s:Concept {name: toLower(item.source_concept)}) MATCH (t:Concept {name: toLower(item.target_concept)}) MERGE (s)-[r:RELATED_TO]->(t) SET r.type = item.relationship_type",
                    {"data": relationships_data}
                )
            
            neo4j_duration = time.time() - start_neo4j
            print(f"Neo4j Batch Ingest took {neo4j_duration:.2f}s")
            print(f"Extracted {len(kg.concepts)} concepts and {len(kg.relationships)} relationships to Neo4j.")
        except Exception as e:
            print(f"Error extracting/inserting graph: {e}")
    else:
        print("Neo4j not configured. Skipping graph extraction.")


def hybrid_retrieve(query: str) -> Dict[str, Any]:
    """Retrieves documents from DataStax and prerequisites from Neo4j."""
    
    retrieved_docs = []
    if vector_store:
        results = vector_store.similarity_search(query, k=3)
        retrieved_docs = [r.page_content for r in results]
    else:
        retrieved_docs = ["(Vector RAG not configured)"]

    prerequisites = []
    graph_nodes = {}
    
    if graph_db:
        # 1. Ask LLM to extract keywords from query to search graph
        # For simplicity, we just use the query to find similar nodes (string matching)
        # A robust way is Vector Index on Neo4j or LLM keyword extraction
        
        # Example Cypher: find nodes mentioned in query, get their prerequisites
        cypher_query = """
        MATCH (c:Concept)<-[:PREREQUISITE_OF]-(prereq:Concept)
        WHERE toLower($query) CONTAINS toLower(c.name)
        RETURN prereq.name AS prerequisite, prereq.description AS desc
        LIMIT 5
        """
        try:
            results = graph_db.query(cypher_query, {"query": query})
            prerequisites = [r["prerequisite"] for r in results]
            
            # Also fetch all nodes for memory agent (in a real app, don't fetch all, just relevant ones)
            all_nodes_res = graph_db.query("MATCH (c:Concept) RETURN c.name as name, c.description as desc LIMIT 20")
            for record in all_nodes_res:
                graph_nodes[record["name"]] = {"name": record["name"], "description": record["desc"]}
        except Exception as e:
            print(f"Error querying Neo4j: {e}")
            
    else:
        prerequisites = ["(Graph RAG not configured)"]
        graph_nodes = {"variables": {"name": "Variables", "description": "Mock Data"}}

    return {
        "documents": retrieved_docs,
        "prerequisites": prerequisites,
        "graph_nodes": graph_nodes
    }
