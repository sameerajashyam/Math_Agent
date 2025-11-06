import json
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime
import re
import numpy as np

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available. Install with: pip install chromadb")

# Import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Install with: pip install sentence-transformers")

logger = logging.getLogger(__name__)

class IncrementalLearningRAG:
    def __init__(self, kb_path: str = "app/knowledge_base/math_data.json"):
        self.kb_path = kb_path
        self.kb_data = self._load_knowledge_base()
        self.learning_enabled = True
        
        # Initialize Vector Database
        self.vector_db_available = CHROMA_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
        self._init_vectordb()
        
        # Fallback to TF-IDF if VectorDB fails
        if not self.vector_db_available:
            self._init_tfidf_fallback()
        
        logger.info(f"✅ Incremental Learning RAG initialized with {len(self.kb_data)} questions")
        logger.info(f"📊 VectorDB: {'✅ Available' if self.vector_db_available else '❌ Using TF-IDF fallback'}")
    
    def _init_vectordb(self):
        """Initialize ChromaDB Vector Database"""
        if not self.vector_db_available:
            return
            
        try:
            # Initialize ChromaDB with persistence
            self.chroma_client = chromadb.PersistentClient(path="./chroma_math_db")
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="math_knowledge_base",
                metadata={"description": "Mathematical questions and solutions database"}
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build vector index
            self._build_vector_index()
            
        except Exception as e:
            logger.error(f"VectorDB initialization failed: {e}")
            self.vector_db_available = False
            self._init_tfidf_fallback()
    
    def _init_tfidf_fallback(self):
        """Initialize TF-IDF as fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            self._build_tfidf_index()
        except ImportError:
            logger.error("scikit-learn not available for TF-IDF fallback")
    
    def _load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base from JSON file"""
        try:
            if os.path.exists(self.kb_path):
                with open(self.kb_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"📁 Loaded {len(data)} questions from {self.kb_path}")
                    return data
            else:
                logger.warning("Knowledge base file not found, starting with empty KB")
                return []
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return []
    
    def _build_vector_index(self):
        """Build vector index in ChromaDB"""
        if not self.vector_db_available or not self.kb_data:
            return
            
        try:
            # Check if collection is already populated
            existing_count = self.collection.count()
            if existing_count > 0:
                logger.info(f"📊 VectorDB already has {existing_count} documents")
                return
            
            documents = []
            metadatas = []
            ids = []
            
            for i, item in enumerate(self.kb_data):
                question = item.get('question', '')
                if question and len(question.strip()) > 0:
                    documents.append(question)
                    metadatas.append({
                        'topic': item.get('topic', 'unknown'),
                        'source': item.get('source', 'unknown'),
                        'learned_at': item.get('learned_at', ''),
                        'complexity': item.get('complexity', 'medium'),
                        'question_length': len(question)
                    })
                    ids.append(f"math_{i}_{hash(question) % 10000}")
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"✅ VectorDB indexed {len(documents)} questions")
                
        except Exception as e:
            logger.error(f"VectorDB indexing failed: {e}")
            self.vector_db_available = False
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for similarity search (fallback)"""
        if not self.kb_data:
            self.question_vectors = None
            return
            
        questions = [item.get('question', '') for item in self.kb_data]
        
        if len(questions) > 0:
            self.question_vectors = self.vectorizer.fit_transform(questions)
            logger.info("✅ TF-IDF index built")
        else:
            self.question_vectors = None
    
    def search_knowledge_base(self, question: str, threshold: float = 0.7) -> Optional[Dict]:
        """Search for similar questions using VectorDB or TF-IDF fallback"""
        if not self.kb_data:
            return None
            
        # Try VectorDB first
        if self.vector_db_available:
            result = self._search_vectordb(question, threshold)
            if result:
                return result
        
        # Fallback to TF-IDF
        return self._search_tfidf(question, threshold)
    
    def _search_vectordb(self, question: str, threshold: float) -> Optional[Dict]:
        """Search using VectorDB similarity search"""
        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_texts=[question],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'] and len(results['documents'][0]) > 0:
                best_match_text = results['documents'][0][0]
                best_metadata = results['metadatas'][0][0]
                best_distance = results['distances'][0][0]
                
                # Convert distance to similarity (ChromaDB uses cosine distance)
                best_similarity = 1 - best_distance
                
                logger.info(f"🔍 VectorDB search - Similarity: {best_similarity:.4f}")
                
                if best_similarity > threshold:
                    # Find the original KB item
                    for item in self.kb_data:
                        if item.get('question') == best_match_text:
                            return {
                                **item,
                                'similarity_score': float(best_similarity),
                                'vector_search': True,
                                'matched_question': best_match_text
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"VectorDB search failed: {e}")
            return None
    
    def _search_tfidf(self, question: str, threshold: float) -> Optional[Dict]:
        """Search using TF-IDF fallback"""
        if not hasattr(self, 'question_vectors') or self.question_vectors is None:
            return None
            
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Transform query to TF-IDF vector
            query_vector = self.vectorizer.transform([question])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.question_vectors).flatten()
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score > threshold:
                best_match = self.kb_data[best_idx]
                return {
                    **best_match,
                    'similarity_score': float(best_score),
                    'vector_search': False,
                    'fallback_used': True
                }
            return None
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return None

    def hybrid_search(self, question: str, llm_agent=None, confidence_threshold: float = 0.85) -> Optional[Dict]:
        """Enhanced Hybrid KB + LLM search with better numerical difference detection"""
        kb_result = self.search_knowledge_base(question, threshold=0.5)  # Lower threshold for detection
        
        if not kb_result:
            return None
        
        confidence = kb_result.get('similarity_score', 0)
        kb_question = kb_result.get('question', '')
        
        logger.info(f"🔍 Hybrid search - KB confidence: {confidence:.4f}")
        logger.info(f"📊 KB question: {kb_question}")
        logger.info(f"📊 User question: {question}")
        
        # Check for numerical differences between KB question and user question
        has_numerical_diff = self._has_numerical_differences(kb_question, question)
        
        if has_numerical_diff:
            logger.info("🔢 Numerical differences detected between KB and user question")
            
            # If LLM is available, use it to generate correct solution
            if llm_agent and llm_agent.is_available():
                logger.info("🔄 Using LLM to generate correct solution for numerical differences")
                return self._enhance_with_llm(question, kb_result, llm_agent)
            else:
                logger.warning("⚠️ Numerical differences detected but LLM not available")
                # Return KB result with warning
                return {
                    **kb_result,
                    'similarity_score': confidence * 0.7,  # Lower confidence for numerical differences
                    'numerical_warning': True,
                    'warning_message': 'Numerical values differ from KB question - answer may be incorrect',
                    'original_kb_values': self._extract_numerical_values(kb_question),
                    'user_question_values': self._extract_numerical_values(question)
                }
        
        # No numerical differences - use KB result if confidence is high enough
        if confidence >= confidence_threshold:
            logger.info(f"✅ High confidence KB match ({confidence:.4f}) - using directly")
            return kb_result
        else:
            logger.info(f"📚 Medium confidence KB match ({confidence:.4f})")
            return kb_result

    def _has_numerical_differences(self, question1: str, question2: str) -> bool:
        """Enhanced numerical difference detection with better logging"""
        try:
            nums1 = self._extract_numerical_values(question1)
            nums2 = self._extract_numerical_values(question2)
            
            logger.info(f"🔢 Number comparison - KB: {nums1}, User: {nums2}")
            
            # If different number of numerical values, definitely different
            if len(nums1) != len(nums2):
                logger.info("❌ Different number of numerical values")
                return True
            
            # If same count but values differ, check if they're significantly different
            for i, (n1, n2) in enumerate(zip(nums1, nums2)):
                if n1 != n2:
                    # Consider any difference as significant for money/rates
                    difference_ratio = abs(n1 - n2) / max(n1, n2)
                    logger.info(f"🔢 Number difference at position {i}: {n1} vs {n2}, ratio: {difference_ratio:.3f}")
                    if difference_ratio > 0.01:  # 1% difference threshold (more sensitive)
                        logger.info("❌ Significant numerical difference detected")
                        return True
            
            logger.info("✅ No significant numerical differences")
            return False
            
        except Exception as e:
            logger.error(f"❌ Numerical difference detection failed: {e}")
            return False

    def _extract_numerical_values(self, text: str) -> List[float]:
        """Extract all numerical values from text"""
        try:
            # Extract all numbers (including decimals and with $ signs)
            numbers = re.findall(r'\$?(\d+\.?\d*)', text)
            # Convert to floats and filter out empty strings
            return [float(x) for x in numbers if x.strip()]
        except Exception as e:
            logger.error(f"❌ Numerical value extraction failed: {e}")
            return []

    def _enhance_with_llm(self, question: str, kb_result: Dict, llm_agent) -> Dict:
        """Enhanced LLM enhancement for numerical differences"""
        try:
            # Use LLM to solve the actual question with correct numbers
            llm_solution = llm_agent.solve_math_problem(question)
            
            if llm_solution:
                logger.info("✅ LLM generated correct solution for numerical differences")
                return {
                    **kb_result,
                    'solution': llm_solution,
                    'source': 'llm_enhanced',
                    'original_kb_confidence': kb_result['similarity_score'],
                    'similarity_score': 0.9,  # High confidence for LLM-calculated solution
                    'numerical_recalculation': True,
                    'recalculation_reason': 'Numerical values differ from KB question'
                }
            else:
                logger.warning("❌ LLM solution generation failed, falling back to KB")
                return {
                    **kb_result,
                    'numerical_warning': True,
                    'warning_message': 'Numerical differences detected but LLM failed - using KB solution with incorrect values'
                }
                
        except Exception as e:
            logger.error(f"❌ LLM enhancement failed: {e}")
            return {
                **kb_result,
                'numerical_warning': True,
                'warning_message': f'Numerical differences detected but enhancement failed: {str(e)}'
            }

    def process_query(self, question: str, solution: Dict, learn: bool = True) -> Dict:
        """
        Process query with automatic learning
        Returns: {found_in_kb: bool, learned: bool, similarity_score: float}
        """
        try:
            # Step 1: Search in KB
            kb_result = self.search_knowledge_base(question)
            
            result = {
                "found_in_kb": kb_result is not None,
                "similarity_score": kb_result.get('similarity_score', 0) if kb_result else 0,
                "learned": False
            }
            
            # Step 2: Auto-learn if new question and learning enabled
            if not result["found_in_kb"] and learn and self.learning_enabled:
                learned = self.add_learned_solution(question, solution, "auto_learned_query")
                result["learned"] = learned
                result["new_kb_size"] = len(self.kb_data)
                
                if learned:
                    logger.info(f"🎓 Auto-learned new question: {question[:50]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {"found_in_kb": False, "learned": False, "error": str(e)}
    
    def add_learned_solution(self, question: str, solution: Dict, source: str = "learned") -> bool:
        """Add a new question-solution pair to KB and VectorDB"""
        try:
            # Check if question already exists (high threshold)
            existing = self.search_knowledge_base(question, threshold=0.9)
            if existing:
                logger.info(f"📚 Question already exists in KB (similarity: {existing.get('similarity_score', 0):.3f})")
                return False
            
            # Create new entry
            new_entry = {
                "question": question,
                "topic": self._classify_topic(question),
                "solution": solution,
                "source": source,
                "learned_at": self._get_timestamp(),
                "learned_from": "user_query"
            }
            
            # Add to KB
            self.kb_data.append(new_entry)
            
            # Add to VectorDB if available
            if self.vector_db_available:
                self._add_to_vectordb(question, new_entry)
            
            # Save to file
            self._save_knowledge_base()
            
            # Rebuild indexes
            if self.vector_db_available:
                # VectorDB is automatically updated
                pass
            else:
                self._build_tfidf_index()
            
            logger.info(f"📖 Added new question to KB: {question[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add learned solution: {e}")
            return False
    
    def _add_to_vectordb(self, question: str, metadata: Dict):
        """Add single document to VectorDB"""
        try:
            doc_id = f"math_{len(self.kb_data)}_{hash(question) % 10000}"
            
            self.collection.add(
                documents=[question],
                metadatas=[{
                    'topic': metadata.get('topic', 'unknown'),
                    'source': metadata.get('source', 'unknown'),
                    'learned_at': metadata.get('learned_at', ''),
                    'complexity': 'medium',
                    'question_length': len(question)
                }],
                ids=[doc_id]
            )
            logger.info(f"✅ Added to VectorDB: {question[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to add to VectorDB: {e}")
    
    def load_dataset_file(self, file_path: str, source: str = "dataset") -> Dict:
        """Load questions from a dataset file"""
        try:
            if not os.path.exists(file_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                elif file_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    return {"status": "error", "message": "Unsupported file format"}
            
            loaded_count = 0
            for item in data:
                # Extract question and solution based on common formats
                question = item.get('question', '') or item.get('problem', '')
                solution = self._parse_solution(item)
                
                if question and solution:
                    success = self.add_learned_solution(question, solution, source)
                    if success:
                        loaded_count += 1
            
            logger.info(f"📚 Loaded {loaded_count} questions from {file_path}")
            return {
                "status": "success",
                "loaded_questions": loaded_count,
                "file_path": file_path,
                "new_kb_size": len(self.kb_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to load dataset file: {e}")
            return {"status": "error", "message": str(e)}
    
    def _parse_solution(self, item: Dict) -> Dict:
        """Parse solution from different dataset formats"""
        try:
            # GS8MK format
            if 'answer' in item:
                answer = item['answer']
                steps = []
                
                # Parse step-by-step solution
                if '####' in answer:
                    lines = answer.split('\n')
                    step_number = 1
                    for line in lines:
                        line = line.strip()
                        if line.startswith('####'):
                            continue  # Skip answer line
                        elif line and line.startswith('#'):
                            steps.append({
                                "step_number": step_number,
                                "explanation": line.replace('#', '').strip(),
                                "equation": self._extract_equation(line)
                            })
                            step_number += 1
                
                return {
                    "steps": steps or [{"step_number": 1, "explanation": answer, "equation": ""}],
                    "final_answer": self._extract_final_answer(answer)
                }
            
            # Custom format
            elif 'solution' in item:
                return item['solution']
            
            # Default format
            else:
                return {
                    "steps": [{"step_number": 1, "explanation": "Solution not available", "equation": ""}],
                    "final_answer": "Answer not provided"
                }
                
        except Exception as e:
            logger.error(f"Failed to parse solution: {e}")
            return {
                "steps": [{"step_number": 1, "explanation": "Error parsing solution", "equation": ""}],
                "final_answer": "Error"
            }
    
    def _extract_equation(self, text: str) -> str:
        """Extract mathematical equations from text"""
        equations = re.findall(r'\d+[\+\-\*\/]\d+[\+\-\*\/\d]*=?\d*', text)
        return equations[0] if equations else ""
    
    def _extract_final_answer(self, answer: str) -> str:
        """Extract final answer from solution text"""
        if '####' in answer:
            lines = answer.split('\n')
            for line in lines:
                if line.startswith('####'):
                    return line.replace('####', '').strip()
        return answer.split('\n')[-1] if '\n' in answer else answer
    
    def _classify_topic(self, question: str) -> str:
        """Classify question topic automatically"""
        question_lower = question.lower()
        
        topic_keywords = {
            'probability': ['probability', 'chance', 'likely', 'random', 'odds', 'ticket', 'dice', 'coin'],
            'algebra': ['solve', 'equation', 'variable', 'x', 'y', 'algebra', 'linear', 'quadratic'],
            'geometry': ['area', 'volume', 'triangle', 'circle', 'angle', 'geometry', 'perimeter'],
            'calculus': ['derivative', 'integral', 'limit', 'calculus', 'differentiate'],
            'arithmetic': ['sum', 'product', 'difference', 'multiple', 'divide', 'add', 'subtract'],
            'work_rate': ['work', 'rate', 'efficiency', 'days', 'hours', 'complete', 'finish'],
            'percentage': ['percent', '%', 'faster', 'slower', 'increase', 'decrease'],
            'speed_time': ['speed', 'time', 'distance', 'km/h', 'mph', 'minutes', 'hours'],
            'word_problem': ['how many', 'each', 'total', 'left', 'remaining', 'together']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                return topic
        
        return "general_math"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
    
    def _save_knowledge_base(self):
        """Save KB to file"""
        try:
            os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
            with open(self.kb_path, 'w') as f:
                json.dump(self.kb_data, f, indent=2)
            logger.info(f"💾 KB saved with {len(self.kb_data)} questions")
        except Exception as e:
            logger.error(f"Failed to save KB: {e}")
    
    def get_kb_stats(self) -> Dict:
        """Get KB statistics"""
        topics = {}
        sources = {}
        
        for item in self.kb_data:
            topic = item.get('topic', 'unknown')
            source = item.get('source', 'unknown')
            
            topics[topic] = topics.get(topic, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        vector_stats = self.get_vector_stats()
            
        return {
            "total_questions": len(self.kb_data),
            "topics": topics,
            "sources": sources,
            "learning_enabled": self.learning_enabled,
            "vector_database": vector_stats
        }
    
    def get_vector_stats(self) -> Dict:
        """Get VectorDB statistics"""
        if not self.vector_db_available:
            return {"status": "not_available", "message": "Using TF-IDF fallback"}
        
        try:
            count = self.collection.count()
            return {
                "status": "active",
                "documents": count,
                "collection": "math_knowledge_base",
                "embedding_model": "all-MiniLM-L6-v2",
                "persistence": True
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def similarity_search(self, question: str, n_results: int = 5) -> List[Dict]:
        """Get multiple similar questions with scores"""
        if not self.vector_db_available:
            return []
            
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            similar_questions = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                similarity = 1 - distance
                
                similar_questions.append({
                    'question': doc,
                    'similarity_score': round(similarity, 4),
                    'topic': metadata.get('topic', 'unknown'),
                    'source': metadata.get('source', 'unknown')
                })
            
            return similar_questions
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def search_by_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """Search questions by topic using metadata filtering"""
        if not self.vector_db_available:
            return []
            
        try:
            results = self.collection.get(
                where={"topic": topic},
                limit=n_results,
                include=['documents', 'metadatas']
            )
            
            topic_questions = []
            for i in range(len(results['documents'])):
                topic_questions.append({
                    'question': results['documents'][i],
                    'topic': results['metadatas'][i].get('topic', 'unknown'),
                    'source': results['metadatas'][i].get('source', 'unknown')
                })
            
            return topic_questions
            
        except Exception as e:
            logger.error(f"Topic search failed: {e}")
            return []

    def export_kb_sample(self, limit: int = 10) -> List[Dict]:
        """Export sample of KB for inspection"""
        return self.kb_data[:limit]
    
    def toggle_learning(self, enabled: bool) -> Dict:
        """Enable/disable automatic learning"""
        self.learning_enabled = enabled
        return {"learning_enabled": self.learning_enabled, "message": f"Learning {'enabled' if enabled else 'disabled'}"}

    def generate_basic_solution(self, question: str) -> Dict:
        """Generate a basic fallback solution"""
        return {
            "solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "explanation": f"Analyzing question: {question}",
                        "equation": ""
                    },
                    {
                        "step_number": 2,
                        "explanation": "This appears to be a mathematical problem that requires step-by-step reasoning",
                        "equation": ""
                    }
                ],
                "final_answer": "Solution would be generated based on mathematical principles"
            }
        }