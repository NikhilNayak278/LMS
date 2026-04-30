import time
from typing import Optional
from pydantic import BaseModel
from services.hybrid_rag import llm, vector_store, graph_db

class ChallengeGeneration(BaseModel):
    concept_name: str
    question_text: str
    difficulty: str

class AssessmentResult(BaseModel):
    score: float
    feedback: str
    is_passed: bool
    socratic_hint: str

def get_next_challenge(user_id: str) -> Optional[ChallengeGeneration]:
    if not graph_db:
        print("Graph DB not configured.")
        return None

    # Find a concept the user hasn't mastered, where all its prerequisites (if any) ARE mastered.
    cypher_query = """
    MATCH (c:Concept)
    WHERE NOT EXISTS {
        MATCH (:User {id: $user_id})-[:MASTERED]->(c)
    }
    AND NOT EXISTS {
        MATCH (prereq:Concept)-[:RELATED_TO {type: 'PREREQUISITE_OF'}]->(c)
        WHERE NOT EXISTS {
            MATCH (:User {id: $user_id})-[:MASTERED]->(prereq)
        }
    }
    RETURN c.name AS concept_name, c.description AS description
    LIMIT 1
    """
    
    try:
        results = graph_db.query(cypher_query, {"user_id": user_id})
    except Exception as e:
        print(f"Error querying Neo4j for next challenge: {e}")
        return None

    if not results:
        # User has mastered everything or no concepts exist
        return None
        
    concept_name = results[0]["concept_name"]
    
    # Fetch Ground Truth
    context = ""
    if vector_store:
        try:
            docs = vector_store.similarity_search(concept_name, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error querying Astra DB for context: {e}")
            
    # Generate question
    structured_llm = llm.with_structured_output(ChallengeGeneration)
    prompt = f"""
    Based on the following context, generate a short-answer question to test the student's understanding of the concept '{concept_name}'.
    Return the concept_name, the question_text, and a difficulty level (Easy/Medium/Hard).
    
    Context:
    {context}
    """
    
    try:
        challenge: ChallengeGeneration = structured_llm.invoke(prompt)
        challenge.concept_name = concept_name # enforce correctness
        return challenge
    except Exception as e:
        print(f"Error generating challenge with Gemini: {e}")
        return None

def update_user_progress(user_id: str, concept_name: str, score: float):
    if not graph_db:
        return
        
    if score >= 0.7:
        cypher = """
        MERGE (u:User {id: $user_id})
        MATCH (c:Concept) WHERE toLower(c.name) = toLower($concept_name)
        MERGE (u)-[r:MASTERED]->(c)
        SET r.score = $score, r.timestamp = timestamp(), r.attempts = coalesce(r.attempts, 0) + 1
        WITH u, c
        OPTIONAL MATCH (u)-[s:STRUGGLING_WITH]->(c)
        DELETE s
        """
    else:
        cypher = """
        MERGE (u:User {id: $user_id})
        MATCH (c:Concept) WHERE toLower(c.name) = toLower($concept_name)
        MERGE (u)-[s:STRUGGLING_WITH]->(c)
        SET s.score = $score, s.timestamp = timestamp(), s.attempts = coalesce(s.attempts, 0) + 1
        """
        
    try:
        graph_db.query(cypher, {"user_id": user_id, "concept_name": concept_name, "score": score})
    except Exception as e:
        print(f"Error updating user progress in Neo4j: {e}")

def grade_answer(user_id: str, concept_name: str, question: str, student_answer: str) -> Optional[AssessmentResult]:
    context = ""
    if vector_store:
        try:
            docs = vector_store.similarity_search(concept_name, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        except Exception as e:
            print(f"Error querying Astra DB for context: {e}")
            
    structured_llm = llm.with_structured_output(AssessmentResult)
    prompt = f"""
    Evaluate the student's answer to the question based on the provided ground truth context.
    
    Concept: {concept_name}
    Question: {question}
    Student Answer: {student_answer}
    
    Ground Truth Context:
    {context}
    
    Return a score between 0.0 and 1.0 (where >= 0.7 is passing), brief feedback, whether they passed, and a socratic hint if they failed.
    """
    
    try:
        result: AssessmentResult = structured_llm.invoke(prompt)
        update_user_progress(user_id, concept_name, result.score)
        return result
    except Exception as e:
        print(f"Error grading answer with Gemini: {e}")
        return None
