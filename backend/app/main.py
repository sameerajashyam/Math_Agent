from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from .models import MathQuestion, MathSolution, FeedbackRequest, LearningFeedback, HumanFeedback, FeedbackType, StepFeedback, SystemEvaluation, make_json_safe, safe_float
from .agents.rag_incremental import IncrementalLearningRAG
from .agents.routing_agent import RoutingAgent
from .agents.feedback_agent import FeedbackAgent
from .agents.dspy_optimizer import DSPyOptimizer
from .agents.llm_agent import LLMAgent
from .agents.hitl_feedback_agent import HITLFeedbackAgent
from .agents.web_search_agent import WebSearchAgent

import logging
import json
import os
from typing import Dict, Optional, List
from dotenv import load_dotenv
from datetime import datetime
import math

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Math Agent API - Complete System with AI Gateway & VectorDB & DSPy",
    description="Self-Improving Math Agent with AI Gateway Guardrails, Vector Database, DSPy Optimization & Human-in-the-Loop Learning",
    version="7.0.0"  # Updated version for DSPy integration
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents with enhanced RAG and DSPy optimization
math_rag = IncrementalLearningRAG()
routing_agent = RoutingAgent()
feedback_agent = FeedbackAgent()
dspy_optimizer = DSPyOptimizer()
hitl_agent = HITLFeedbackAgent(rag_system=math_rag)
web_search_agent = WebSearchAgent()

# ==================== GSM8K DATASET LOADING ====================

try:
    from datasets import load_dataset
    GSM8K_AVAILABLE = True
except ImportError:
    logger.warning("📦 'datasets' package not installed. Run: pip install datasets")
    GSM8K_AVAILABLE = False

def load_gsm8k_dataset(limit: int = 500):
    """Load GSM8K dataset automatically"""
    if not GSM8K_AVAILABLE:
        logger.warning("❌ GSM8K not available - install datasets package")
        return 0
        
    try:
        logger.info("📥 Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main")
        
        logger.info(f"📊 Dataset structure: {type(dataset)}")
        logger.info(f"📊 Dataset keys: {list(dataset.keys())}")
        
        train_data = dataset['train']
        logger.info(f"📊 Train data type: {type(train_data)}")
        logger.info(f"📊 Train data length: {len(train_data)}")
        
        # Check first item to understand structure
        if len(train_data) > 0:
            first_item = train_data[0]
            logger.info(f"📊 First item type: {type(first_item)}")
            logger.info(f"📊 First item keys: {list(first_item.keys())}")
            logger.info(f"📊 First item preview: {str(first_item)[:200]}...")
        
        loaded_count = 0
        
        for i in range(min(limit, len(train_data))):
            try:
                item = train_data[i]
                
                # Debug: log what we're getting
                if i == 0:
                    logger.info(f"🔍 Processing first item: {item}")
                
                # Extract question and answer safely
                if isinstance(item, dict):
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                else:
                    logger.warning(f"Unexpected item type at index {i}: {type(item)}")
                    continue
                
                if not question or not answer:
                    logger.warning(f"Skipping item {i} - missing question or answer")
                    continue
                
                # Parse GSM8K format
                if "####" in answer:
                    parts = answer.split("####")
                    reasoning = parts[0].strip() if len(parts) > 0 else ""
                    final_answer = parts[1].strip() if len(parts) > 1 else answer
                else:
                    reasoning = answer
                    final_answer = answer
                
                # Create solution structure
                solution = {
                    "steps": [
                        {
                            "step_number": 1,
                            "explanation": reasoning[:500],
                            "equation": ""
                        }
                    ],
                    "final_answer": final_answer
                }
                
                # Auto-classify topic
                topic = classify_math_topic(question)
                
                # Add to knowledge base
                success = math_rag.add_learned_solution(
                    question=question,
                    solution=solution,
                    source="gsm8k"
                )
                
                if success:
                    loaded_count += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"📊 Loaded {i + 1} questions...")
                    
            except Exception as e:
                logger.warning(f"Failed to process item {i}: {e}")
                continue
        
        logger.info(f"✅ Loaded {loaded_count} GSM8K questions into knowledge base")
        return loaded_count
        
    except Exception as e:
        logger.error(f"❌ Failed to load GSM8K: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0

def classify_math_topic(question: str) -> str:
    """Auto-classify question topic"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['sequence', 'pattern', 'next number', 'fibonacci']):
        return "number_sequences"
    elif any(word in question_lower for word in ['probability', 'chance', 'likely', 'odds', 'dice', 'coin']):
        return "probability"
    elif any(word in question_lower for word in ['percent', '%', 'discount', 'interest']):
        return "percentage"
    elif any(word in question_lower for word in ['ratio', 'proportion']):
        return "ratios"
    elif any(word in question_lower for word in ['speed', 'time', 'distance', 'km/h', 'mph', 'minutes']):
        return "speed_time"
    elif any(word in question_lower for word in ['area', 'volume', 'triangle', 'circle', 'perimeter', 'geometry']):
        return "geometry"
    elif any(word in question_lower for word in ['solve', 'equation', 'x', 'y', 'algebra', 'quadratic']):
        return "algebra"
    elif any(word in question_lower for word in ['work', 'rate', 'efficiency', 'days', 'hours', 'complete']):
        return "work_rate"
    elif any(word in question_lower for word in ['sum', 'product', 'difference', 'multiple', 'divide', 'add', 'subtract']):
        return "arithmetic"
    else:
        return "general_math"

# ==================== ENHANCED SOURCE NAMING SYSTEM ====================

def get_display_source(source: str) -> str:
    """Convert technical source names to user-friendly names"""
    source_map = {
        'knowledge_base': 'knowledge_base',
        'llm_enhanced': 'ai_enhanced',
        'web_search': 'web_search',
        'mcp_web_search': 'mcp_search',
        'equation_solver': 'equation_solver',
        'speed_time_solver': 'speed_time_solver',
        'enhanced_fallback': 'basic_solver',
        'web_search_fallback': 'web_search',
        'dspy_enhanced': 'dspy_optimized',
        'knowledge_base_enhanced': 'knowledge_base',
        'llm_fallback': 'web_search_llm',
        'mcp_web_search': 'mcp_search',
        'enhanced_agent': 'web_search_llm',
        'gsm8k': 'knowledge_base',
        'feedback_learning': 'knowledge_base',
        'hitl_learning': 'knowledge_base',
        'corrected_solution': 'knowledge_base',
        'dspy_optimized': 'dspy_optimized'
    }
    return source_map.get(source, 'web_search_llm')

def enhance_solution_with_display_source(solution_data: Dict) -> Dict:
    """Add display source to solution data"""
    solution_data['display_source'] = get_display_source(solution_data.get('source', ''))
    return solution_data

# ==================== DSPY OPTIMIZATION ENDPOINTS ====================

@app.get("/dspy/status")
async def get_dspy_status():
    """Get DSPy optimization status and statistics"""
    try:
        dspy_stats = dspy_optimizer.get_optimization_stats()
        search_stats = web_search_agent.get_dspy_optimization_stats()
        success_rates = web_search_agent.get_success_rates()
        
        return {
            "dspy_system": "active",
            "optimization_stats": dspy_stats,
            "search_optimization": search_stats,
            "strategy_success_rates": success_rates,
            "features": [
                "feedback_learning",
                "routing_optimization", 
                "strategy_optimization",
                "performance_tracking",
                "pattern_analysis"
            ]
        }
    except Exception as e:
        logger.error(f"DSPy status failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/dspy/learning-history")
async def get_dspy_learning_history(limit: int = 10):
    """Get DSPy learning history"""
    try:
        learning_history = dspy_optimizer.get_learning_history(limit)
        return {
            "learning_history": learning_history,
            "total_entries": len(learning_history),
            "recent_activity": [entry['type'] for entry in learning_history]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/dspy/analyze-feedback-patterns")
async def analyze_feedback_patterns_dspy():
    """Use DSPy to analyze feedback patterns"""
    try:
        feedbacks = feedback_agent.get_all_feedbacks()
        patterns = dspy_optimizer.analyze_feedback_patterns(feedbacks)
        
        return {
            "analysis_type": "dspy_pattern_analysis",
            "patterns_found": patterns.get('patterns_identified', False),
            "insights": patterns.get('insights', {}),
            "suggested_actions": patterns.get('suggested_actions', []),
            "feedbacks_analyzed": patterns.get('feedbacks_analyzed', 0)
        }
    except Exception as e:
        logger.error(f"DSPy pattern analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/dspy/optimize-search-strategy")
async def optimize_search_strategy_dspy(question: str):
    """Use DSPy to optimize search strategy for a specific question"""
    try:
        # Analyze question for DSPy optimization
        topic = classify_math_topic(question)
        complexity = "intermediate"  # Could be enhanced with more sophisticated analysis
        
        # Get performance history for DSPy
        performance_history = web_search_agent.get_success_rates()
        
        # Use DSPy to optimize strategy
        strategy_optimization = dspy_optimizer.optimize_search_strategy(
            question=question,
            topic=topic,
            complexity=complexity,
            performance_history=performance_history
        )
        
        return {
            "question": question,
            "topic": topic,
            "complexity": complexity,
            "strategy_optimization": strategy_optimization,
            "performance_history": performance_history
        }
    except Exception as e:
        logger.error(f"DSPy strategy optimization failed: {e}")
        return {"status": "error", "error": str(e)}

# ==================== ENHANCED FEEDBACK WITH DSPY ====================

@app.post("/feedback/dspy-enhanced")
async def submit_dspy_enhanced_feedback(feedback_data: dict):
    """
    Enhanced feedback with DSPy learning and optimization
    """
    try:
        logger.info(f"🎯 DSPy-enhanced feedback received - Rating: {feedback_data.get('rating', 0)}")
        
        # Process with DSPy learning
        dspy_insights = dspy_optimizer.learn_from_feedback(feedback_data)
        
        # Process with web search agent for strategy optimization
        search_feedback_result = web_search_agent.process_search_feedback(feedback_data)
        
        # Auto-learn from high-rated solutions
        learning_triggered = False
        if feedback_data.get('rating', 0) >= 4 and feedback_data.get('question') and feedback_data.get('original_solution'):
            learning_triggered = math_rag.add_learned_solution(
                question=feedback_data['question'],
                solution=feedback_data['original_solution'],
                source="dspy_enhanced_feedback"
            )
        
        return {
            "status": "processed",
            "dspy_learning_applied": dspy_insights.get('dspy_learned', False),
            "search_optimization": search_feedback_result.get('performance_updated', False),
            "learning_triggered": learning_triggered,
            "dspy_insights": dspy_insights,
            "search_insights": search_feedback_result,
            "message": "Feedback processed with DSPy optimization",
            "kb_size": len(math_rag.kb_data),
            "current_success_rates": web_search_agent.get_success_rates()
        }
        
    except Exception as e:
        logger.error(f"DSPy-enhanced feedback failed: {e}")
        return {"status": "error", "error": str(e)}

# ==================== ENHANCED SOLVE WITH DSPY OPTIMIZATION ====================

@app.post("/solve/dspy-optimized", response_model=MathSolution)
async def solve_math_dspy_optimized(question: MathQuestion):
    """
    Solve math problem with DSPy-optimized routing and search strategies
    """
    try:
        logger.info(f"🎯 DSPy-optimized solving: {question.question}")
        
        # Get relevant feedback history for DSPy optimization
        feedback_history = feedback_agent.get_all_feedbacks()
        relevant_feedback = [fb for fb in feedback_history if any(word in fb.get('question', '').lower() for word in question.question.lower().split())][:5]
        
        # Enhanced routing with DSPy optimization
        routing_result = routing_agent.process_question(question.question, math_rag)
        
        if routing_result['status'] == 'error':
            if 'gateway_action' in routing_result and routing_result['gateway_action'] == 'blocked':
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": routing_result['error_message'],
                        "suggestion": routing_result.get('suggestion', 'Please ask a mathematical question'),
                        "gateway_blocked": True
                    }
                )
            else:
                raise HTTPException(status_code=400, detail=routing_result['error_message'])
        
        sanitized_question = routing_result['sanitized_question']
        route_decision = routing_result['route_decision']
        
        logger.info(f"🎯 DSPy Route: {route_decision['suggested_action']}, Confidence: {route_decision['confidence']:.2f}")
        
        # Use DSPy-optimized web search when needed
        if any(action in route_decision['suggested_action'] for action in ['web_search', 'mcp_search']):
            logger.info("🌐 Using DSPy-optimized web search")
            web_result = web_search_agent.search_math_solution(
                question=sanitized_question,
                kb_confidence=route_decision.get('kb_confidence', 0.0),
                feedback_history=relevant_feedback
            )
            
            if web_result and web_result.get('confidence', 0) > 0.3:
                logger.info(f"✅ DSPy-optimized search successful: {web_result.get('search_method', 'unknown')}")
                
                solution_data = {
                    'question': sanitized_question,
                    'steps': web_result["solution"]["steps"],
                    'final_answer': web_result["solution"]["final_answer"],
                    'source': "dspy_enhanced",
                    'confidence': web_result["confidence"],
                    'dspy_optimized': web_result.get('dspy_optimized', False),
                    'search_method': web_result.get('search_method', 'unknown')
                }
                
                # Add source information if available
                if 'source_url' in web_result:
                    solution_data['source_url'] = web_result['source_url']
                if 'source_title' in web_result:
                    solution_data['source_title'] = web_result['source_title']
                
                # Validate and sanitize output
                output_validation = routing_agent.validate_output(solution_data, sanitized_question)
                if output_validation["status"] != "approved":
                    logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
                    solution_data['gateway_warnings'] = output_validation
                
                sanitized_solution = routing_agent.sanitize_output(solution_data)
                enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
                
                enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
                if output_validation.get("quality_score"):
                    enhanced_solution['quality_score'] = output_validation["quality_score"]
                
                return MathSolution(**enhanced_solution)
        
        # Fallback to standard routing for other cases
        return await solve_math_problem(question)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ DSPy-optimized solving failed: {e}")
        # Fallback to standard solving
        return await solve_math_problem(question)

# ==================== VECTORDB ENDPOINTS ====================

@app.get("/vectordb/stats")
async def get_vectordb_stats():
    """Get VectorDB statistics and health"""
    try:
        vector_stats = math_rag.get_vector_stats()
        kb_stats = math_rag.get_kb_stats()
        
        return {
            "vector_database": "chromadb",
            "status": "active",
            "vector_stats": vector_stats,
            "knowledge_base_stats": kb_stats,
            "features": [
                "vector_similarity_search",
                "metadata_filtering", 
                "persistent_storage",
                "automatic_indexing",
                "topic_based_search"
            ]
        }
    except Exception as e:
        logger.error(f"VectorDB stats failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/vectordb/search-similar")
async def vectordb_similarity_search(question: str, limit: int = 5):
    """Search for similar questions using VectorDB"""
    try:
        similar_questions = math_rag.similarity_search(question, limit)
        
        return {
            "query_question": question,
            "similar_questions": similar_questions,
            "total_found": len(similar_questions),
            "search_method": "vector_similarity",
            "vector_db_used": math_rag.vector_db_available
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/vectordb/search-by-topic")
async def search_by_topic(topic: str, limit: int = 10):
    """Search questions by topic using VectorDB metadata filtering"""
    try:
        topic_questions = math_rag.search_by_topic(topic, limit)
        
        return {
            "topic": topic,
            "questions": topic_questions,
            "total_found": len(topic_questions),
            "search_method": "metadata_filtering",
            "vector_db_used": math_rag.vector_db_available
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/vectordb/add-question")
async def add_question_to_vectordb(learning_data: dict):
    """Add question to VectorDB directly"""
    try:
        question = learning_data.get('question', '')
        solution = learning_data.get('solution', {})
        source = learning_data.get('source', 'direct_add')
        
        if not question or not solution:
            return {"status": "error", "message": "Question and solution required"}
        
        success = math_rag.add_learned_solution(question, solution, source)
        
        if success:
            vector_stats = math_rag.get_vector_stats()
            return {
                "status": "success",
                "message": "Added to VectorDB and knowledge base",
                "question_added": question[:100] + "..." if len(question) > 100 else question,
                "vector_db_documents": vector_stats.get('documents', 'unknown'),
                "kb_size": len(math_rag.kb_data)
            }
        else:
            return {"status": "error", "message": "Failed to add or already exists"}
            
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/vectordb/compare-search")
async def compare_search_methods(question: str):
    """Compare VectorDB vs TF-IDF search results"""
    try:
        # VectorDB search
        vector_results = math_rag.similarity_search(question, 3)
        
        # TF-IDF search (if available)
        tfidf_results = []
        if hasattr(math_rag, 'search_knowledge_base'):
            kb_result = math_rag.search_knowledge_base(question)
            if kb_result:
                tfidf_results = [{
                    'question': kb_result.get('question', ''),
                    'similarity_score': kb_result.get('similarity_score', 0),
                    'topic': kb_result.get('topic', 'unknown'),
                    'source': 'tfidf_search'
                }]
        
        return {
            "query_question": question,
            "vector_db_available": math_rag.vector_db_available,
            "vector_db_results": vector_results,
            "tfidf_results": tfidf_results,
            "comparison": {
                "vector_db_count": len(vector_results),
                "tfidf_count": len(tfidf_results),
                "recommended_method": "vector_db" if math_rag.vector_db_available else "tfidf"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==================== CORE ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "Self-Improving Math Agent API with AI Gateway, VectorDB & DSPy Optimization - Complete System v7.0"}

@app.get("/health")
async def health_check():
    vector_stats = math_rag.get_vector_stats()
    dspy_stats = dspy_optimizer.get_optimization_stats()
    
    return {
        "status": "healthy", 
        "phase": 7, 
        "learning_enabled": True, 
        "ai_gateway": "active",
        "vector_db": vector_stats.get('status', 'unknown'),
        "dspy_optimization": "active" if dspy_stats.get('dspy_available') else "disabled"
    }

@app.post("/solve", response_model=MathSolution)
async def solve_math_problem(question: MathQuestion):
    """
    Complete math problem solving with AI Gateway guardrails and smart routing
    """
    try:
        logger.info(f"🔍 Processing: {question.question}")
        
        # Enhanced routing with RAG integration and AI Gateway
        routing_result = routing_agent.process_question(question.question, math_rag)
        
        if routing_result['status'] == 'error':
            # Check if it's an AI Gateway block
            if 'gateway_action' in routing_result and routing_result['gateway_action'] == 'blocked':
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": routing_result['error_message'],
                        "suggestion": routing_result.get('suggestion', 'Please ask a mathematical question'),
                        "gateway_blocked": True
                    }
                )
            else:
                raise HTTPException(status_code=400, detail=routing_result['error_message'])
        
        sanitized_question = routing_result['sanitized_question']
        route_decision = routing_result['route_decision']
        
        logger.info(f"🎯 Route: {route_decision['suggested_action']}, Confidence: {route_decision['confidence']:.2f}")
        
        # Handle different routing scenarios with AI Gateway output validation
        if route_decision['suggested_action'] == 'use_kb':
            return await _handle_kb_search(sanitized_question, routing_agent)
        
        elif route_decision['suggested_action'] == 'use_kb_with_fallback':
            return await _handle_kb_with_fallback(sanitized_question, routing_agent, route_decision)
        
        elif route_decision['suggested_action'] == 'web_search_available':
            return await _handle_web_search(sanitized_question, routing_agent, route_decision)
        
        elif route_decision['suggested_action'] == 'mcp_web_search':
            return await _handle_mcp_search(sanitized_question, routing_agent, route_decision)
        
        else:
            return await _handle_fallback(sanitized_question, routing_agent)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error solving math problem: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _handle_kb_search(question: str, routing_agent: RoutingAgent) -> MathSolution:
    """Handle KB search with LLM enhancement for numerical differences"""
    # Initialize LLM agent for hybrid search
    llm_agent = LLMAgent()
    
    # Use hybrid search with numerical difference detection
    kb_result = math_rag.hybrid_search(question, llm_agent, confidence_threshold=0.7)  # Lower threshold
    
    if kb_result:
        # PRIORITIZE KB results from feedback_learning source
        source = kb_result.get('source', '')
        if source == 'feedback_learning':
            logger.info("🎓 Using feedback-learned solution (high priority)")
            confidence_boost = 1.0  # Maximum confidence for learned solutions
        else:
            confidence_boost = kb_result.get('similarity_score', 0)
        
        logger.info(f"✅ KB match found with confidence: {confidence_boost:.2f}")
        
        # Check if this was LLM enhanced for numerical differences
        if kb_result.get('source') == 'llm_enhanced':
            logger.info("🔄 Using LLM-enhanced solution for numerical differences")
        
        solution_data = {
            'question': question,
            'steps': kb_result["solution"]["steps"],
            'final_answer': kb_result["solution"]["final_answer"],
            'source': kb_result.get('source', 'knowledge_base'),
            'confidence': confidence_boost,  # Use boosted confidence
            'kb_topic': kb_result.get('topic', 'unknown'),
            'vector_search_used': kb_result.get('vector_search', False)
        }
        
        # Add warning if numerical differences detected but not enhanced
        if kb_result.get('numerical_warning'):
            solution_data['warning'] = kb_result.get('warning_message', 'Numerical differences detected')
            solution_data['original_kb_values'] = kb_result.get('original_kb_values')
            solution_data['user_question_values'] = kb_result.get('user_question_values')
            logger.warning(f"⚠️ {solution_data['warning']}")
            
        # Add recalculation info if LLM was used
        if kb_result.get('numerical_recalculation'):
            solution_data['recalculation_note'] = kb_result.get('recalculation_reason', 'Recalculated with correct values')
            logger.info(f"🔢 {solution_data['recalculation_note']}")
    else:
        logger.info("🔄 No KB match, trying web search")
        return await _handle_web_search_fallback(question, routing_agent, "No KB match")
    
    # Validate and sanitize output with AI Gateway
    output_validation = routing_agent.validate_output(solution_data, question)
    if output_validation["status"] != "approved":
        logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
        # Still return solution but with warnings
        solution_data['gateway_warnings'] = output_validation
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    
    # Add AI Gateway metadata
    enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
    if output_validation.get("quality_score"):
        enhanced_solution['quality_score'] = output_validation["quality_score"]
    
    return MathSolution(**enhanced_solution)

async def _handle_kb_with_fallback(question: str, routing_agent: RoutingAgent, route_decision: Dict) -> MathSolution:
    """Handle medium-confidence KB search with web search fallback"""
    # Initialize LLM agent for hybrid search
    llm_agent = LLMAgent()
    
    # Use hybrid search for better numerical handling
    kb_result = math_rag.hybrid_search(question, llm_agent, confidence_threshold=0.6)
    
    if kb_result and kb_result.get('similarity_score', 0) > 0.5:
        logger.info("✅ Medium-confidence KB match found")
        
        # Boost confidence for feedback-learned solutions
        source = kb_result.get('source', '')
        if source == 'feedback_learning':
            confidence_boost = 1.0
        else:
            confidence_boost = kb_result.get('similarity_score', 0)
            
        solution_data = {
            'question': question,
            'steps': kb_result["solution"]["steps"],
            'final_answer': kb_result["solution"]["final_answer"],
            'source': kb_result.get('source', 'knowledge_base'),
            'confidence': confidence_boost,
            'kb_topic': kb_result.get('topic', 'unknown'),
            'vector_search_used': kb_result.get('vector_search', False)
        }
        
        # Add warning if numerical differences detected
        if kb_result.get('numerical_warning'):
            solution_data['warning'] = kb_result.get('warning_message', 'Numerical differences detected')
            solution_data['original_kb_values'] = kb_result.get('original_kb_values')
            solution_data['user_question_values'] = kb_result.get('user_question_values')
            logger.warning(f"⚠️ {solution_data['warning']}")
            
        # Add recalculation info if LLM was used
        if kb_result.get('numerical_recalculation'):
            solution_data['recalculation_note'] = kb_result.get('recalculation_reason', 'Recalculated with correct values')
            logger.info(f"🔢 {solution_data['recalculation_note']}")
    else:
        logger.info("🔍 Medium confidence, trying web search")
        kb_confidence = route_decision.get('kb_confidence', 0.0)
        web_result = routing_agent.perform_web_search(question, kb_confidence)
        if web_result and web_result.get('confidence', 0) > 0.3:
            logger.info("✅ Web search provided good results")
            solution_data = {
                'question': question,
                'steps': web_result["solution"]["steps"],
                'final_answer': web_result["solution"]["final_answer"],
                'source': web_result["source"],
                'confidence': web_result["confidence"]
            }
            # Add source info if available
            if 'source_url' in web_result:
                solution_data['source_url'] = web_result['source_url']
            if 'source_title' in web_result:
                solution_data['source_title'] = web_result['source_title']
            
            # Add gateway info if available
            if 'gateway_approved' in web_result:
                solution_data['gateway_approved'] = web_result['gateway_approved']
            if 'quality_score' in web_result:
                solution_data['quality_score'] = web_result['quality_score']
        else:
            logger.info("❌ Both KB and web search failed")
            return await _handle_enhanced_fallback(question, routing_agent, 
                f"Medium confidence question. KB: {route_decision['kb_confidence']:.2f}, " +
                f"Complexity: {route_decision['complexity_level']}")
    
    # Validate and sanitize output with AI Gateway
    output_validation = routing_agent.validate_output(solution_data, question)
    if output_validation["status"] != "approved":
        logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
        solution_data['gateway_warnings'] = output_validation
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    
    # Add AI Gateway metadata
    enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
    if output_validation.get("quality_score"):
        enhanced_solution['quality_score'] = output_validation["quality_score"]
    
    return MathSolution(**enhanced_solution)

async def _handle_web_search(question: str, routing_agent: RoutingAgent, route_decision: Dict) -> MathSolution:
    """Handle web search requests with KB context"""
    logger.info("🌐 Performing enhanced web search with LLM")
    
    kb_confidence = route_decision.get('kb_confidence', 0.0)
    web_result = routing_agent.perform_web_search(question, kb_confidence)
    
    if web_result and web_result.get('confidence', 0) > 0.3:
        source = web_result.get('source', 'web_search')
        logger.info(f"✅ Enhanced search successful: {source}")
        
        solution_data = {
            'question': question,
            'steps': web_result["solution"]["steps"],
            'final_answer': web_result["solution"]["final_answer"],
            'source': source,
            'confidence': web_result["confidence"]
        }
        
        # Add source information if available
        if 'source_url' in web_result:
            solution_data['source_url'] = web_result['source_url']
        if 'source_title' in web_result:
            solution_data['source_title'] = web_result['source_title']
        
        # Add gateway info if available
        if 'gateway_approved' in web_result:
            solution_data['gateway_approved'] = web_result['gateway_approved']
        if 'quality_score' in web_result:
            solution_data['quality_score'] = web_result['quality_score']
        if 'gateway_warnings' in web_result:
            solution_data['gateway_warnings'] = web_result['gateway_warnings']
    else:
        logger.info("❌ Enhanced search failed")
        return await _handle_enhanced_fallback(
            question, 
            routing_agent, 
            f"Complex question requiring external knowledge. " +
            f"Complexity: {route_decision['complexity_level']}, " +
            f"Confidence: {route_decision['confidence']:.2f}"
        )
    
    # Additional AI Gateway validation for web search results
    output_validation = routing_agent.validate_output(solution_data, question)
    if output_validation["status"] != "approved":
        logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
        solution_data['gateway_warnings'] = output_validation
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    
    # Ensure gateway metadata is included
    if 'gateway_approved' not in enhanced_solution:
        enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
    if output_validation.get("quality_score") and 'quality_score' not in enhanced_solution:
        enhanced_solution['quality_score'] = output_validation["quality_score"]
    
    return MathSolution(**enhanced_solution)

async def _handle_mcp_search(question: str, routing_agent: RoutingAgent, route_decision: Dict) -> MathSolution:
    """Handle MCP web search requests"""
    logger.info("🔍 Performing MCP web search")
    
    kb_confidence = route_decision.get('kb_confidence', 0.0)
    web_result = routing_agent.perform_web_search(question, kb_confidence)
    
    if web_result and web_result.get('confidence', 0) > 0.3:
        source = 'mcp_web_search' if web_result.get('mcp_used', False) else 'web_search'
        logger.info(f"✅ MCP search successful: {source}")
        
        solution_data = {
            'question': question,
            'steps': web_result["solution"]["steps"],
            'final_answer': web_result["solution"]["final_answer"],
            'source': source,
            'confidence': web_result["confidence"],
            'mcp_used': web_result.get('mcp_used', False)
        }
        
        # Add source information if available
        if 'source_url' in web_result:
            solution_data['source_url'] = web_result['source_url']
        if 'source_title' in web_result:
            solution_data['source_title'] = web_result['source_title']
        
        # Add gateway info
        if 'gateway_approved' in web_result:
            solution_data['gateway_approved'] = web_result['gateway_approved']
        if 'quality_score' in web_result:
            solution_data['quality_score'] = web_result['quality_score']
    else:
        logger.info("❌ MCP search failed, falling back to regular web search")
        return await _handle_web_search(question, routing_agent, route_decision)
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    return MathSolution(**enhanced_solution)

async def _handle_web_search_fallback(question: str, routing_agent: RoutingAgent, reason: str) -> MathSolution:
    """Handle web search as fallback"""
    logger.info(f"🔄 Web search fallback: {reason}")
    
    web_result = routing_agent.perform_web_search(question)
    if web_result and web_result.get('confidence', 0) > 0.3:
        logger.info("✅ Web search fallback successful")
        solution_data = {
            'question': question,
            'steps': web_result["solution"]["steps"],
            'final_answer': web_result["solution"]["final_answer"],
            'source': "web_search_fallback",
            'confidence': web_result["confidence"]
        }
        
        if 'source_url' in web_result:
            solution_data['source_url'] = web_result['source_url']
        
        # Add gateway info if available
        if 'gateway_approved' in web_result:
            solution_data['gateway_approved'] = web_result['gateway_approved']
        if 'quality_score' in web_result:
            solution_data['quality_score'] = web_result['quality_score']
    else:
        return await _handle_enhanced_fallback(question, routing_agent, 
            f"{reason}. Web search also unavailable.")
    
    # Validate and sanitize output with AI Gateway
    output_validation = routing_agent.validate_output(solution_data, question)
    if output_validation["status"] != "approved":
        logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
        solution_data['gateway_warnings'] = output_validation
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    
    enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
    if output_validation.get("quality_score"):
        enhanced_solution['quality_score'] = output_validation["quality_score"]
    
    return MathSolution(**enhanced_solution)

async def _handle_enhanced_fallback(question: str, routing_agent: RoutingAgent, reason: str) -> MathSolution:
    """Enhanced fallback with detailed messaging"""
    logger.info(f"🔄 Enhanced fallback: {reason}")
    
    # Try to generate basic solution
    try:
        fallback_solution = math_rag.generate_basic_solution(question)
    except:
        # Create a basic fallback solution
        fallback_solution = {
            "solution": {
                "steps": [
                    {
                        "step_number": 1,
                        "explanation": "This question requires specialized knowledge",
                        "equation": ""
                    }
                ],
                "final_answer": "Solution not available in current knowledge base"
            }
        }
    
    # Enhance fallback message
    enhanced_steps = fallback_solution["solution"]["steps"].copy()
    enhanced_steps.insert(0, {
        "step_number": 1,
        "explanation": f"ℹ️ {reason}",
        "equation": ""
    })
    
    enhanced_steps.append({
        "step_number": len(enhanced_steps) + 1,
        "explanation": "💡 For complex questions, set up web search API keys for better results",
        "equation": ""
    })
    
    solution_data = {
        'question': question,
        'steps': enhanced_steps,
        'final_answer': fallback_solution["solution"]["final_answer"],
        'source': "enhanced_fallback",
        'confidence': 0.1
    }
    
    # Validate and sanitize output with AI Gateway
    output_validation = routing_agent.validate_output(solution_data, question)
    if output_validation["status"] != "approved":
        logger.warning(f"⚠️ AI Gateway output validation: {output_validation['reason']}")
        solution_data['gateway_warnings'] = output_validation
    
    sanitized_solution = routing_agent.sanitize_output(solution_data)
    enhanced_solution = enhance_solution_with_display_source(sanitized_solution)
    
    enhanced_solution['gateway_approved'] = output_validation["status"] == "approved"
    if output_validation.get("quality_score"):
        enhanced_solution['quality_score'] = output_validation["quality_score"]
    
    return MathSolution(**enhanced_solution)

async def _handle_fallback(question: str, routing_agent: RoutingAgent) -> MathSolution:
    """Basic fallback handler"""
    return await _handle_enhanced_fallback(question, routing_agent, "Default fallback mechanism")

# ==================== AI GATEWAY ENDPOINTS ====================

@app.get("/gateway/analytics")
async def get_gateway_analytics():
    """Get AI Gateway guardrail analytics"""
    try:
        gateway_stats = routing_agent.get_gateway_analytics()
        return {
            "system": "math_ai_gateway",
            "status": "active",
            "analytics": gateway_stats,
            "features": [
                "input_validation",
                "mathematical_intent_detection", 
                "content_safety",
                "pii_protection",
                "output_quality_assessment",
                "educational_focus"
            ]
        }
    except Exception as e:
        logger.error(f"Gateway analytics failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/gateway/test")
async def test_gateway(question: str):
    """Test the AI Gateway with a question"""
    try:
        gateway_test = routing_agent.test_gateway_validation(question)
        return {
            "test_question": question,
            "gateway_result": gateway_test,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gateway/blocked-requests")
async def get_blocked_requests(limit: int = 20):
    """View recently blocked requests (for monitoring)"""
    try:
        analytics = routing_agent.get_gateway_analytics()
        blocked_requests = analytics.get('input_blocks', [])[-limit:] if analytics.get('input_blocks') else []
        
        return {
            "recent_blocks": blocked_requests,
            "total_blocks": len(analytics.get('input_blocks', [])),
            "block_rate": analytics.get('block_rate', 0)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==================== ROUTING ANALYTICS ENDPOINTS ====================

@app.get("/routing/analytics")
async def get_routing_analytics():
    """Get comprehensive routing analytics"""
    try:
        routing_stats = routing_agent.get_routing_stats()
        gateway_analytics = routing_agent.get_gateway_analytics()
        
        return {
            "routing_system": "enhanced_with_ai_gateway",
            "routing_stats": routing_stats,
            "gateway_analytics": gateway_analytics,
            "mcp_available": routing_stats.get('mcp_available', False),
            "confidence_thresholds": {
                "high": routing_stats.get('high_confidence_threshold'),
                "medium": routing_stats.get('medium_confidence_threshold'),
                "low": routing_stats.get('low_confidence_threshold')
            }
        }
    except Exception as e:
        logger.error(f"Routing analytics failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/routing/test-decision")
async def test_routing_decision(question: str):
    """Test routing decision for a specific question"""
    try:
        routing_test = routing_agent.test_routing_decision(question, math_rag)
        return routing_test
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/routing/numerical-analysis")
async def analyze_numerical_sensitivity(question: str):
    """Analyze numerical sensitivity of a question"""
    try:
        analysis = routing_agent.analyze_numerical_sensitivity(question)
        return analysis
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==================== GSM8K DATASET ENDPOINTS ====================

@app.post("/knowledge-base/load-gsm8k")
async def load_gsm8k_endpoint(limit: int = 500):
    """Load GSM8K dataset manually"""
    try:
        if not GSM8K_AVAILABLE:
            return {
                "status": "error",
                "message": "GSM8K not available. Install: pip install datasets"
            }
            
        loaded_count = load_gsm8k_dataset(limit)
        return {
            "status": "success",
            "loaded_questions": loaded_count,
            "new_kb_size": len(math_rag.kb_data),
            "message": f"Loaded {loaded_count} GSM8K questions"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/gsm8k-status")
async def gsm8k_status():
    """Check GSM8K dataset status"""
    gsm8k_count = sum(1 for item in math_rag.kb_data if item.get('source') == 'gsm8k')
    
    return {
        "gsm8k_available": GSM8K_AVAILABLE,
        "gsm8k_questions_loaded": gsm8k_count,
        "total_questions": len(math_rag.kb_data),
        "installation_command": "pip install datasets" if not GSM8K_AVAILABLE else "already_installed"
    }

# ==================== GS8MK SPECIFIC ENDPOINTS ====================

@app.post("/knowledge-base/load-gs8mk")
async def load_gs8mk_dataset():
    """Load the full GS8MK dataset"""
    try:
        # Path to your GS8MK dataset file
        gs8mk_path = "app/knowledge_base/gs8mk_dataset.json"
        
        if not os.path.exists(gs8mk_path):
            return {
                "status": "error", 
                "message": f"GS8MK dataset file not found at {gs8mk_path}",
                "suggestion": "Create gs8mk_dataset.json file with GS8MK questions"
            }
        
        result = math_rag.load_dataset_file(gs8mk_path)
        
        return {
            "status": "success",
            "message": "GS8MK dataset loaded",
            "loaded_questions": result.get("loaded_questions", 0),
            "new_kb_size": len(math_rag.kb_data),
            "file_path": gs8mk_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/gs8mk")
async def test_gs8mk_questions():
    """Test GS8MK question routing and solutions"""
    test_questions = [
        "What is the next number in this sequence? 2, 6, 12, 20, 30, ?",
        "A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. If you draw two marbles without replacement, what is the probability that both are red?",
        "A car travels from City A to City B at 60 km/h and returns at 40 km/h. What is the average speed for the entire trip?",
        "If 3x + 7 = 22 and 2y - 5 = 15, what is the value of x + y?",
        "The ratio of boys to girls in a class is 3:5. If there are 24 students total, how many girls are there?"
    ]
    
    results = []
    for question in test_questions:
        try:
            solution = await solve_math_problem(MathQuestion(question=question))
            results.append({
                "question": question,
                "source": solution.source,
                "display_source": get_display_source(solution.source),
                "confidence": solution.confidence,
                "answer": solution.final_answer,
                "gateway_approved": getattr(solution, 'gateway_approved', True),
                "quality_score": getattr(solution, 'quality_score', 0.8)
            })
        except Exception as e:
            results.append({
                "question": question,
                "error": str(e)
            })
    
    return {
        "test_type": "GS8MK_Questions",
        "results": results,
        "summary": f"Tested {len(results)} GS8MK-style questions"
    }

@app.get("/debug/gs8mk-check")
async def debug_gs8mk_check():
    """Check if GS8MK questions are in KB"""
    gs8mk_questions = [
        "What is the next number in this sequence? 2, 6, 12, 20, 30, ?",
        "A bag contains 5 red marbles, 3 blue marbles, and 2 green marbles. If you draw two marbles without replacement, what is the probability that both are red?",
        "A car travels from City A to City B at 60 km/h and returns at 40 km/h. What is the average speed for the entire trip?",
        "If 3x + 7 = 22 and 2y - 5 = 15, what is the value of x + y?",
        "The ratio of boys to girls in a class is 3:5. If there are 24 students total, how many girls are there?"
    ]
    
    results = []
    for question in gs8mk_questions:
        kb_result = math_rag.search_knowledge_base(question)
        results.append({
            "question": question[:50] + "...",
            "in_kb": bool(kb_result),
            "confidence": kb_result.get('similarity_score', 0) if kb_result else 0,
            "would_use_kb": kb_result.get('similarity_score', 0) > 0.7 if kb_result else False
        })
    
    return {
        "total_kb_questions": len(math_rag.kb_data),
        "gs8mk_check": results
    }

# ==================== INCREMENTAL LEARNING ENDPOINTS ====================

@app.post("/solve/learn")
async def solve_and_learn(question: MathQuestion, learn: bool = True):
    """
    Solve math problem with automatic learning from new queries
    """
    try:
        logger.info(f"🔍 Processing with learning: {question.question}")
        
        # First, try to solve using existing routing
        solution = await solve_math_problem(question)
        
        # Convert solution to the format needed for learning
        solution_dict = {
            "steps": solution.steps,
            "final_answer": solution.final_answer
        }
        
        # Auto-learn if this is a new question
        learning_result = math_rag.process_query(question.question, solution_dict, learn)
        
        # Add learning info to response
        response_data = solution.dict()
        response_data["learning"] = learning_result
        
        return response_data
        
    except Exception as e:
        logger.error(f"Solve and learn failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/load-file")
async def load_dataset_file(file_path: str):
    """Load questions from a dataset file (JSON or JSONL)"""
    try:
        result = math_rag.load_dataset_file(file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/knowledge-base/learn-from-solution")
async def learn_from_solution(question: str, solution: Dict):
    """Manually add a solution to KB"""
    try:
        success = math_rag.add_learned_solution(question, solution, "manual_learning")
        return {
            "status": "learned" if success else "already_exists",
            "question": question[:100] + "..." if len(question) > 100 else question,
            "kb_size": len(math_rag.kb_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning/toggle")
async def toggle_learning(enabled: bool = True):
    """Enable/disable automatic learning"""
    try:
        result = math_rag.toggle_learning(enabled)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== FEEDBACK & LEARNING ENDPOINTS ====================

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Basic feedback collection"""
    logger.info(f"📊 Basic feedback - Rating: {feedback.rating}")
    return {
        "status": "feedback_received", 
        "message": "Thank you for your feedback!",
        "rating": feedback.rating
    }

@app.post("/feedback/enhanced")
async def submit_enhanced_feedback(feedback: LearningFeedback):
    """
    Enhanced feedback with AUTO-LEARNING capabilities
    """
    try:
        result = feedback_agent.process_feedback(feedback)
        
        # AUTO-LEARNING: Add high-rated solutions to KB automatically
        if feedback.user_rating >= 4:  # Only learn from good solutions (4-5 stars)
            learning_result = math_rag.process_query(
                feedback.question, 
                feedback.original_solution.dict(), 
                learn=True
            )
            result["incremental_learning"] = learning_result
            logger.info(f"📚 Auto-learning from feedback: {learning_result}")
        
        # Run optimization if we have enough feedback
        total_feedbacks = result.get("total_feedbacks", 0)
        if total_feedbacks >= 10 and total_feedbacks % 10 == 0:
            optimization_result = dspy_optimizer.analyze_feedback_patterns(
                feedback_agent.get_all_feedbacks()
            )
            result["optimization"] = optimization_result
        
        return result
        
    except Exception as e:
        logger.error(f"Enhanced feedback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HUMAN-IN-THE-LOOP ENDPOINTS ====================

@app.post("/feedback/hitl")
async def human_in_the_loop_feedback(feedback: HumanFeedback):
    """
    Comprehensive Human-in-the-Loop feedback processing
    """
    try:
        logger.info(f"👤 HITL feedback received - Type: {feedback.feedback_type}")
        
        result = hitl_agent.process_human_feedback(feedback)
        
        return {
            "status": "success",
            "message": "Feedback processed successfully",
            "feedback_id": result.get("feedback_id"),
            "learning_applied": result.get("learning_applied"),
            "improvements": result.get("improvements"),
            "next_actions": result.get("next_actions")
        }
        
    except Exception as e:
        logger.error(f"HITL feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/correct-solution")
async def correct_solution(
    question: str,
    original_solution: Dict,
    corrected_solution: Dict,
    user_notes: Optional[str] = None
):
    """
    Endpoint for solution correction
    """
    # Ensure the solution data is properly formatted
    original_solution_safe = make_json_safe(original_solution)
    corrected_solution_safe = make_json_safe(corrected_solution)
    
    # Create MathSolution objects with safe data
    try:
        original_math_solution = MathSolution(**original_solution_safe)
    except Exception as e:
        logger.error(f"Error creating original MathSolution: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid original solution format: {str(e)}")
    
    feedback = HumanFeedback(
        question=question,
        original_solution=original_math_solution,
        feedback_type=FeedbackType.CORRECTION,
        corrected_solution=corrected_solution_safe,
        user_notes=user_notes
    )
    
    return await human_in_the_loop_feedback(feedback)

@app.post("/feedback/rate-solution")
async def rate_solution(
    question: str,
    original_solution: Dict, 
    rating: int,
    feedback_notes: Optional[str] = None
):
    """
    Endpoint for solution rating
    """
    # Ensure the solution data is properly formatted
    original_solution_safe = make_json_safe(original_solution)
    
    try:
        original_math_solution = MathSolution(**original_solution_safe)
    except Exception as e:
        logger.error(f"Error creating original MathSolution: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid original solution format: {str(e)}")
    
    feedback = HumanFeedback(
        question=question,
        original_solution=original_math_solution,
        feedback_type=FeedbackType.RATING,
        rating=rating,
        user_notes=feedback_notes
    )
    
    return await human_in_the_loop_feedback(feedback)

@app.get("/hitl/analytics")
async def get_hitl_analytics():
    """
    Get HITL system analytics
    """
    try:
        analytics = hitl_agent.get_feedback_analytics()
        evaluation = hitl_agent.get_system_evaluation()
        
        return {
            "feedback_analytics": analytics,
            "system_evaluation": evaluation,
            "hitl_status": "active"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hitl/learning-cycles")
async def get_learning_cycles(limit: int = 10):
    """
    Get recent learning cycles
    """
    try:
        cycles_data = hitl_agent.get_learning_cycles(limit)
        
        return {
            "total_cycles": cycles_data.get("total_cycles", 0),
            "recent_cycles": cycles_data.get("recent_cycles", []),
            "status": cycles_data.get("status", "success")
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Enhanced solve endpoint with feedback collection
@app.post("/solve/with-feedback")
async def solve_with_feedback_collection(question: MathQuestion):
    """
    Solve math problem with built-in feedback collection
    """
    try:
        solution = await solve_math_problem(question)
        
        # Add feedback collection information
        solution_dict = solution.dict()
        solution_dict["feedback_interface"] = {
            "message": "Help improve the system! Provide feedback on this solution:",
            "available_feedback_types": [
                {"type": "correction", "endpoint": "/feedback/correction"},
                {"type": "rating", "endpoint": "/feedback/rate-solution"},
                {"type": "general", "endpoint": "/feedback/hitl"}
            ],
            "quick_rating": {
                "message": "Rate this solution (1-5):",
                "endpoint": "/feedback/rate-solution"
            }
        }
        
        return solution_dict
        
    except Exception as e:
        logger.error(f"Feedback-enabled solution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== KNOWLEDGE BASE MANAGEMENT ====================

@app.post("/learning/add-solution")
async def add_solution_to_kb(question: str, solution: Dict, topic: str = ""):
    """Manually add a solution to knowledge base"""
    try:
        success = math_rag.add_learned_solution(question, solution, "manual_addition")
        return {
            "status": "added" if success else "already_exists",
            "question": question[:100] + "..." if len(question) > 100 else question,
            "kb_size": len(math_rag.kb_data),
            "topic": topic or "auto_classified"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/explore")
async def explore_knowledge_base(
    search: str = "",
    topic: str = "",
    limit: int = 20,
    offset: int = 0
):
    """Explore the knowledge base with search and filtering"""
    try:
        # Get all questions from RAG
        all_questions = math_rag.kb_data
        
        filtered_data = all_questions
        
        # Apply search filter
        if search:
            filtered_data = [
                item for item in filtered_data 
                if search.lower() in item.get('question', '').lower()
                or search.lower() in item.get('topic', '').lower()
            ]
        
        # Apply topic filter
        if topic:
            filtered_data = [
                item for item in filtered_data 
                if topic.lower() in item.get('topic', '').lower()
            ]
        
        # Paginate results
        paginated_data = filtered_data[offset:offset + limit]
        
        return {
            "total_questions": len(all_questions),
            "filtered_count": len(filtered_data),
            "showing": f"{offset + 1}-{min(offset + limit, len(filtered_data))} of {len(filtered_data)}",
            "questions": [
                {
                    "id": i + offset,
                    "question": item.get('question', '')[:100] + "..." if len(item.get('question', '')) > 100 else item.get('question', ''),
                    "topic": item.get('topic', 'unknown'),
                    "source": item.get('source', 'unknown'),
                    "learned_at": item.get('learned_at', 'unknown'),
                    "has_solution": 'solution' in item
                }
                for i, item in enumerate(paginated_data)
            ],
            "all_topics": list(set(item.get('topic', 'unknown') for item in all_questions))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/knowledge-base/search-similar")
async def search_similar_questions(question: str, limit: int = 5):
    """Search for questions similar to the given one"""
    try:
        # Use VectorDB for similarity search if available
        if math_rag.vector_db_available:
            similar_questions = math_rag.similarity_search(question, limit)
        else:
            # Fallback to manual search
            similar_questions = []
            for item in math_rag.kb_data[:limit*2]:
                if item.get('question') != question:
                    question_words = set(question.lower().split())
                    item_words = set(item.get('question', '').lower().split())
                    common_words = question_words.intersection(item_words)
                    similarity = len(common_words) / max(len(question_words), 1)
                    
                    if similarity > 0.3:
                        similar_questions.append({
                            "question": item.get('question', ''),
                            "topic": item.get('topic', 'unknown'),
                            "similarity_score": round(similarity, 3),
                            "source": item.get('source', 'unknown')
                        })
            
            similar_questions.sort(key=lambda x: x['similarity_score'], reverse=True)
            similar_questions = similar_questions[:limit]
        
        return {
            "input_question": question,
            "similar_questions": similar_questions,
            "search_method": "vector_db" if math_rag.vector_db_available else "manual"
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/knowledge-base/test-match")
async def test_kb_match(question: str):
    """Test if a specific question matches in knowledge base"""
    try:
        kb_result = math_rag.search_knowledge_base(question)
        
        return {
            "test_question": question,
            "kb_match_found": bool(kb_result),
            "kb_confidence": kb_result.get('similarity_score', 0) if kb_result else 0,
            "matched_question": kb_result.get('question', '') if kb_result else None,
            "matched_topic": kb_result.get('topic', '') if kb_result else None,
            "would_use_kb": kb_result.get('similarity_score', 0) > 0.7 if kb_result else False,
            "vector_search_used": kb_result.get('vector_search', False) if kb_result else False
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== SYSTEM STATUS & ANALYTICS ====================

@app.get("/analytics/performance")
async def get_performance_analytics():
    """Get system performance analytics"""
    try:
        performance_report = feedback_agent.get_performance_report()
        optimization_history = dspy_optimizer.get_learning_history()
        kb_stats = math_rag.get_kb_stats()
        routing_stats = routing_agent.get_routing_stats()
        gateway_analytics = routing_agent.get_gateway_analytics()
        vector_stats = math_rag.get_vector_stats()
        dspy_stats = dspy_optimizer.get_optimization_stats()
        search_stats = web_search_agent.get_success_rates()
        
        # Make all data JSON safe
        safe_performance_report = make_json_safe(performance_report)
        safe_optimization_history = make_json_safe(optimization_history[-5:])
        safe_kb_stats = make_json_safe(kb_stats)
        safe_vector_stats = make_json_safe(vector_stats)
        safe_routing_stats = make_json_safe(routing_stats)
        safe_gateway_analytics = make_json_safe(gateway_analytics)
        safe_dspy_stats = make_json_safe(dspy_stats)
        safe_search_stats = make_json_safe(search_stats)
        
        return {
            "performance_report": safe_performance_report,
            "dspy_optimization_history": safe_optimization_history,
            "knowledge_base_stats": safe_kb_stats,
            "vector_database_stats": safe_vector_stats,
            "routing_stats": safe_routing_stats,
            "gateway_analytics": safe_gateway_analytics,
            "dspy_optimization_stats": safe_dspy_stats,
            "search_strategy_success": safe_search_stats,
            "total_feedbacks": feedback_agent._get_total_feedbacks(),
            "system_health": "excellent" if len(performance_report) > 0 else "learning"
        }
        
    except Exception as e:
        logger.error(f"Analytics failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/status")
async def learning_status():
    """Get learning system status"""
    kb_stats = math_rag.get_kb_stats()
    routing_stats = routing_agent.get_routing_stats()
    vector_stats = math_rag.get_vector_stats()
    dspy_stats = dspy_optimizer.get_optimization_stats()
    
    # Make all data JSON safe
    safe_kb_stats = make_json_safe(kb_stats)
    safe_vector_stats = make_json_safe(vector_stats)
    safe_dspy_stats = make_json_safe(dspy_stats)
    
    return {
        "phase": 7,
        "learning_system": "active",
        "knowledge_base": safe_kb_stats,
        "vector_database": safe_vector_stats,
        "dspy_optimization": {
            "status": "active" if dspy_stats.get('dspy_available') else "disabled",
            "optimizations_applied": dspy_stats.get('total_optimizations', 0),
            "learning_cycles": dspy_stats.get('feedback_learnings', 0)
        },
        "incremental_learning": {
            "enabled": math_rag.learning_enabled,
            "strategy": "learn_from_new_queries",
            "auto_save": True,
            "duplicate_prevention": True
        },
        "ai_gateway": {
            "status": "active",
            "input_guardrails": "enabled",
            "output_validation": "enabled",
            "educational_focus": "enabled"
        },
        "components": {
            "feedback_analysis": "active",
            "dspy_optimization": "active" if dspy_stats.get('dspy_available') else "disabled", 
            "performance_tracking": "active",
            "self_improvement": "active",
            "incremental_learning": "active",
            "auto_kb_expansion": "active",
            "ai_gateway_guardrails": "active",
            "vector_database": "active" if math_rag.vector_db_available else "fallback",
            "dspy_search_optimization": "active"
        },
        "capabilities": {
            "learn_from_feedback": True,
            "optimize_routing": True,
            "track_performance": True,
            "suggest_improvements": True,
            "auto_add_to_kb": True,
            "grow_knowledge_base": True,
            "load_dataset_files": True,
            "ai_gateway_protection": True,
            "vector_similarity_search": math_rag.vector_db_available,
            "dspy_optimized_search": dspy_stats.get('dspy_available', False)
        },
        "status": "continuously_learning"
    }

@app.get("/api/status")
async def api_status():
    """Check API key status"""
    tavily_key = bool(os.getenv('TAVILY_API_KEY'))
    serper_key = bool(os.getenv('SERPER_API_KEY'))
    openai_key = bool(os.getenv('OPENAI_API_KEY'))
    
    return {
        "web_search_apis": {
            "tavily": "configured" if tavily_key else "not_configured",
            "serper": "configured" if serper_key else "not_configured",
            "openai": "configured" if openai_key else "not_configured",
            "real_search_available": tavily_key or serper_key,
            "llm_available": openai_key
        },
        "ai_gateway": {
            "status": "active",
            "guardrails": "enabled",
            "educational_focus": "mathematics_only"
        },
        "vector_database": {
            "status": "active" if math_rag.vector_db_available else "fallback",
            "type": "chromadb" if math_rag.vector_db_available else "tfidf"
        },
        "dspy_optimization": {
            "status": "active" if dspy_optimizer.lm else "disabled",
            "language_model": "openai" if dspy_optimizer.lm else "not_configured"
        },
        "environment": os.getenv('APP_ENV', 'development'),
        "suggestion": "Configure API keys in .env file for full functionality" if not (tavily_key or serper_key or openai_key) else "API keys configured"
    }

@app.get("/system/status")
async def system_status():
    """Get complete system status"""
    tavily_key = bool(os.getenv('TAVILY_API_KEY'))
    openai_key = bool(os.getenv('OPENAI_API_KEY'))
    kb_stats = math_rag.get_kb_stats()
    routing_stats = routing_agent.get_routing_stats()
    vector_stats = math_rag.get_vector_stats()
    dspy_stats = dspy_optimizer.get_optimization_stats()
    
    # Make all data JSON safe
    safe_kb_stats = make_json_safe(kb_stats)
    safe_vector_stats = make_json_safe(vector_stats)
    safe_dspy_stats = make_json_safe(dspy_stats)
    
    return {
        "phase": 7,
        "status": "operational",
        "knowledge_base": safe_kb_stats,
        "vector_database": safe_vector_stats,
        "dspy_optimization": safe_dspy_stats,
        "components": {
            "rag_system": "active",
            "routing_agent": "active", 
            "web_search": "real_api" if tavily_key else "mock_mode",
            "llm_enhancement": "active" if openai_key else "disabled",
            "equation_solver": "active",
            "ai_gateway_guardrails": "active",
            "feedback_system": "active",
            "learning_engine": "active",
            "incremental_kb_growth": "active",
            "mcp_integration": "active" if routing_stats.get('mcp_available') else "disabled",
            "vector_database": "active" if math_rag.vector_db_available else "fallback",
            "dspy_optimization": "active" if dspy_optimizer.lm else "disabled"
        },
        "capabilities": {
            "basic_math": "excellent",
            "equation_solving": "active",
            "advanced_math": "with_web_search",
            "llm_enhancement": "enabled" if openai_key else "disabled",
            "routing_intelligence": "high",
            "ai_gateway_guardrails": "enabled",
            "self_improvement": "enabled",
            "auto_kb_expansion": "enabled",
            "educational_focus": "mathematics_only",
            "vector_similarity_search": math_rag.vector_db_available,
            "dspy_optimized_search": dspy_optimizer.lm is not None
        }
    }

# ==================== DEBUG & TESTING ENDPOINTS ====================

@app.get("/routing/debug")
async def debug_routing(question: str = "Solve 2x + 5 = 15"):
    """Debug routing decisions"""
    routing_result = routing_agent.process_question(question, math_rag)
    
    kb_result = math_rag.search_knowledge_base(question)
    kb_confidence = kb_result["similarity_score"] if kb_result else 0.0
    
    # Test web search if needed
    web_result = None
    if routing_result['status'] == 'success' and routing_result['route_decision']['needs_web_search']:
        web_result = routing_agent.perform_web_search(question, kb_confidence)
    
    return {
        "question": question,
        "routing_analysis": routing_result,
        "kb_search": {
            "confidence": kb_confidence,
            "match_found": bool(kb_result),
            "matched_question": kb_result.get('question') if kb_result else None,
            "vector_search_used": kb_result.get('vector_search', False) if kb_result else False
        },
        "web_search": web_result if web_result else {"status": "not_triggered"},
        "equation_solver": {
            "status": "active",
            "capabilities": ["linear_equations", "system_of_equations"]
        },
        "ai_gateway": {
            "status": "active",
            "input_validation": "passed",
            "educational_focus": "enabled"
        },
        "vector_database": {
            "status": "active" if math_rag.vector_db_available else "fallback",
            "documents": math_rag.get_vector_stats().get('documents', 0) if math_rag.vector_db_available else 0
        },
        "dspy_optimization": {
            "status": "active" if dspy_optimizer.lm else "disabled"
        },
        "summary": {
            "final_decision": routing_result['route_decision']['suggested_action'],
            "reasoning": routing_result['route_decision'].get('reasoning', 'Not specified')
        }
    }

@app.get("/knowledge-base/stats")
async def get_knowledge_base_stats():
    """Get KB statistics"""
    try:
        stats = math_rag.get_kb_stats()
        safe_stats = make_json_safe(stats)
        return {
            **safe_stats,
            "status": "loaded",
            "equation_solver": "active",
            "web_search_available": True,
            "learning_enabled": True,
            "ai_gateway": "active",
            "vector_database": "active" if math_rag.vector_db_available else "fallback",
            "dspy_optimization": "active" if dspy_optimizer.lm else "disabled"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/learning/kb-preview")
async def preview_knowledge_base(limit: int = 10):
    """Preview knowledge base contents"""
    try:
        sample = math_rag.export_kb_sample(limit)
        safe_sample = make_json_safe(sample)
        return {
            "total_questions": len(math_rag.kb_data),
            "preview": safe_sample,
            "vector_db_available": math_rag.vector_db_available
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/question-source")
async def debug_question_source(question: str):
    """Debug endpoint to see exactly why a question uses KB vs web search"""
    try:
        # Check KB first
        kb_result = math_rag.search_knowledge_base(question)
        kb_confidence = kb_result.get('similarity_score', 0) if kb_result else 0
        
        # Check routing decision
        routing_result = routing_agent.process_question(question, math_rag)
        
        # Check AI Gateway validation
        gateway_test = routing_agent.test_gateway_validation(question)
        
        return {
            "question": question,
            "kb_confidence": kb_confidence,
            "kb_match_found": bool(kb_result),
            "vector_search_used": kb_result.get('vector_search', False) if kb_result else False,
            "routing_decision": routing_result['route_decision'],
            "ai_gateway_validation": gateway_test,
            "final_source": "knowledge_base" if kb_confidence > 0.7 else "web_search",
            "reason": "High KB confidence" if kb_confidence > 0.7 else "Low KB confidence, needs web search"
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== NEW NUMERICAL ANALYSIS ENDPOINTS ====================

@app.get("/debug/numerical-analysis")
async def debug_numerical_analysis(question: str):
    """Debug numerical difference detection"""
    try:
        # Initialize LLM agent
        llm_agent = LLMAgent()
        
        # Test KB search
        kb_result = math_rag.search_knowledge_base(question)
        
        # Test numerical difference detection
        if kb_result:
            kb_question = kb_result.get('question', '')
            has_diff = math_rag._has_numerical_differences(kb_question, question)
            kb_numbers = math_rag._extract_numerical_values(kb_question)
            user_numbers = math_rag._extract_numerical_values(question)
            
            # Test hybrid search
            hybrid_result = math_rag.hybrid_search(question, llm_agent)
            
            return {
                "question": question,
                "kb_question": kb_question,
                "has_numerical_differences": has_diff,
                "kb_numbers": kb_numbers,
                "user_numbers": user_numbers,
                "numbers_match": kb_numbers == user_numbers,
                "kb_confidence": kb_result.get('similarity_score', 0),
                "hybrid_search_used_llm": hybrid_result.get('source') == 'llm_enhanced' if hybrid_result else False,
                "final_source": hybrid_result.get('source') if hybrid_result else 'no_match',
                "llm_available": llm_agent.is_available(),
                "vector_search_used": kb_result.get('vector_search', False)
            }
        else:
            return {"error": "No KB match found"}
            
    except Exception as e:
        return {"error": str(e)}

@app.get("/test/numerical-difference")
async def test_numerical_difference(question1: str = "Tina makes $18.00 an hour", question2: str = "Tina makes $19.00 an hour"):
    """Test numerical difference detection between two questions"""
    try:
        has_diff = math_rag._has_numerical_differences(question1, question2)
        nums1 = math_rag._extract_numerical_values(question1)
        nums2 = math_rag._extract_numerical_values(question2)
        
        return {
            "question1": question1,
            "question2": question2,
            "has_numerical_differences": has_diff,
            "numbers1": nums1,
            "numbers2": nums2,
            "difference_ratios": [abs(n1 - n2) / max(n1, n2) for n1, n2 in zip(nums1, nums2)] if len(nums1) == len(nums2) else []
        }
    except Exception as e:
        return {"error": str(e)}

# ==================== ENHANCED LEARNING ENDPOINTS ====================

def remove_solution_from_kb(question: str) -> bool:
    """Remove a solution from knowledge base"""
    try:
        initial_size = len(math_rag.kb_data)
        
        # Filter out the question
        math_rag.kb_data = [
            item for item in math_rag.kb_data 
            if item.get('question') != question
        ]
        
        if len(math_rag.kb_data) < initial_size:
            # Save and rebuild index
            math_rag._save_knowledge_base()
            if math_rag.vector_db_available:
                # Note: VectorDB removal would require additional implementation
                logger.info("⚠️ VectorDB removal not implemented - manual cleanup may be needed")
            else:
                math_rag._build_tfidf_index()
            logger.info(f"✅ Removed question from KB: {question[:50]}...")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to remove solution: {e}")
        return False

@app.post("/feedback/correction")
async def submit_correction(correction_data: dict):
    """Submit correction for wrong solutions"""
    try:
        question = correction_data.get('question', '')
        wrong_solution = correction_data.get('wrong_solution', {})
        correct_solution = correction_data.get('correct_solution', {})
        user_notes = correction_data.get('user_notes', '')
        
        logger.info(f"🔧 Correction submitted for: {question[:50]}...")
        
        # 1. Check if wrong solution is in KB
        kb_match = math_rag.search_knowledge_base(question)
        was_in_kb = kb_match is not None
        
        correction_actions = []
        
        if was_in_kb:
            # 2. Remove wrong solution from KB
            removal_success = remove_solution_from_kb(question)
            if removal_success:
                correction_actions.append("wrong_solution_removed")
                logger.info(f"🗑️ Removed wrong solution from KB: {question[:50]}...")
        
        # 3. Add corrected solution to KB
        if correct_solution and 'steps' in correct_solution and 'final_answer' in correct_solution:
            learning_success = math_rag.add_learned_solution(
                question=question,
                solution=correct_solution,
                source="corrected_solution"
            )
            if learning_success:
                correction_actions.append("correct_solution_added")
                logger.info(f"✅ Added corrected solution to KB: {question[:50]}...")
        
        return {
            "status": "processed",
            "correction_applied": len(correction_actions) > 0,
            "actions": correction_actions,
            "was_in_kb": was_in_kb,
            "message": "Correction processed successfully",
            "kb_size": len(math_rag.kb_data),
            "vector_db_updated": learning_success and math_rag.vector_db_available
        }
        
    except Exception as e:
        logger.error(f"Correction failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/learning/add-to-kb")
async def add_to_kb_direct(learning_data: dict):
    """Direct endpoint to add solutions to KB - SIMPLE VERSION"""
    try:
        question = learning_data.get('question', '')
        solution = learning_data.get('solution', {})
        
        if not question or not solution:
            return {"status": "error", "message": "Question and solution required"}
        
        # Use RAG system to add to KB - WITHOUT confidence parameter
        success = math_rag.add_learned_solution(
            question=question,
            solution=solution,
            source="direct_learning"
        )
        
        if success:
            logger.info(f"🎓 Direct learning: Added '{question[:50]}...' to KB")
            return {
                "status": "success", 
                "message": "Added to knowledge base",
                "kb_size": len(math_rag.kb_data),
                "vector_db_updated": math_rag.vector_db_available
            }
        else:
            return {"status": "error", "message": "Failed to add to KB"}
            
    except Exception as e:
        logger.error(f"Direct learning failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/learning/check-question")
async def check_question_in_kb(question: str):
    """Check if a question exists in KB"""
    try:
        kb_match = math_rag.search_knowledge_base(question)
        
        return {
            "question": question,
            "in_kb": kb_match is not None,
            "confidence": kb_match.get('similarity_score', 0) if kb_match else 0,
            "kb_size": len(math_rag.kb_data),
            "vector_search_used": kb_match.get('vector_search', False) if kb_match else False
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/feedback/simple")
async def simple_feedback_with_learning(feedback_data: dict):
    """Simple feedback endpoint that accepts raw JSON and triggers learning"""
    try:
        rating = feedback_data.get('rating', 0)
        question = feedback_data.get('question', '')
        solution = feedback_data.get('original_solution', {})
        
        logger.info(f"📊 Simple feedback - Rating: {rating}")
        
        # LEARNING TRIGGER: Add high-quality solutions to KB automatically
        learning_triggered = False
        improvements = []
        
        if rating >= 4 and question and solution:
            # Ensure solution is in correct format
            if isinstance(solution, dict) and 'steps' in solution and 'final_answer' in solution:
                learning_triggered = math_rag.add_learned_solution(
                    question=question,
                    solution=solution,
                    source="feedback_learning"
                )
                if learning_triggered:
                    improvements = ["knowledge_base_updated", "system_learning"]
                    logger.info(f"🎓 Learning triggered for: {question[:50]}...")
                    
                    # Record learning cycle for simple feedback too
                    try:
                        hitl_agent._record_learning_cycle(feedback_data, improvements)
                    except Exception as e:
                        logger.error(f"Failed to record simple feedback cycle: {e}")
        
        return {
            "status": "processed", 
            "rating": rating,
            "learning_triggered": learning_triggered,
            "improvements": improvements,
            "message": "Feedback processed" + (" and learned!" if learning_triggered else ""),
            "kb_size": len(math_rag.kb_data),
            "vector_db_updated": learning_triggered and math_rag.vector_db_available
        }
        
    except Exception as e:
        logger.error(f"Simple feedback failed: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/feedback/orchestrated")
async def orchestrated_feedback(feedback_data: dict):
    """Enhanced feedback with complete learning orchestration"""
    try:
        rating = feedback_data.get('rating', 0)
        question = feedback_data.get('question', '')
        solution = feedback_data.get('original_solution', {})
        
        logger.info(f"🎯 Orchestrated feedback - Rating: {rating}")
        
        # Use HITL agent for comprehensive learning
        learning_result = hitl_agent.apply_learning_from_feedback(feedback_data, math_rag)
        
        return {
            "status": "processed",
            "rating": rating,
            "learning_applied": learning_result.get("learning_applied", False),
            "kb_updated": learning_result.get("kb_updated", False),
            "improvements": learning_result.get("improvements", []),
            "message": learning_result.get("message", "Feedback processed"),
            "kb_size": len(math_rag.kb_data)
        }
        
    except Exception as e:
        logger.error(f"Orchestrated feedback failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/learning/cycles")
async def get_learning_cycles(limit: int = 10):
    """Get all learning cycles with details"""
    try:
        cycles_data = hitl_agent.get_learning_cycles(limit)
        
        # Add KB stats to response
        kb_stats = math_rag.get_kb_stats()
        
        # Make data JSON safe
        safe_cycles_data = make_json_safe(cycles_data)
        safe_kb_stats = make_json_safe(kb_stats)
        
        return {
            **safe_cycles_data,
            "kb_stats": safe_kb_stats,
            "learning_velocity": f"{len(cycles_data.get('recent_cycles', []))} cycles recently"
        }
        
    except Exception as e:
        logger.error(f"Learning cycles endpoint failed: {e}")
        return {"status": "error", "error": str(e)}

@app.get("/learning/analytics")
async def get_comprehensive_learning_analytics():
    """Get comprehensive learning analytics"""
    try:
        feedback_analytics = hitl_agent.get_feedback_analytics()
        system_evaluation = hitl_agent.get_system_evaluation()
        kb_stats = math_rag.get_kb_stats()
        learning_cycles = hitl_agent.get_learning_cycles(20)  # Last 20 cycles
        
        # Calculate learning metrics
        total_learned = kb_stats.get('sources', {}).get('feedback_learning', 0) + kb_stats.get('sources', {}).get('hitl_learning', 0) + kb_stats.get('sources', {}).get('corrected_solution', 0)
        total_questions = kb_stats.get('total_questions', 0)
        learning_rate = total_learned / total_questions if total_questions > 0 else 0
        
        # Make all data JSON safe
        safe_feedback_analytics = make_json_safe(feedback_analytics)
        safe_system_evaluation = make_json_safe(system_evaluation)
        safe_kb_stats = make_json_safe(kb_stats)
        safe_learning_cycles = make_json_safe(learning_cycles)
        
        return {
            "feedback_analytics": safe_feedback_analytics,
            "system_evaluation": safe_system_evaluation,
            "knowledge_base": safe_kb_stats,
            "learning_cycles": safe_learning_cycles,
            "learning_metrics": {
                "questions_learned": total_learned,
                "learning_rate": round(learning_rate, 4),
                "kb_growth": total_questions,
                "learning_effectiveness": system_evaluation.get('learning_effectiveness', 0)
            },
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Learning analytics failed: {e}")
        return {"status": "error", "error": str(e)}

# ==================== EXISTING TEST ENDPOINTS ====================

@app.get("/test/mcp-search")
async def test_mcp_search(query: str = "work rate problems mathematics"):
    """Test MCP search functionality"""
    try:
        from app.agents.enhanced_math_agent import EnhancedMathAgent
        agent = EnhancedMathAgent()
        
        results = agent._search_via_mcp(query)
        return {
            "status": "success",
            "query": query,
            "results_preview": results[:500] + "..." if results else "No results",
            "results_length": len(results) if results else 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/test/equation-solver")
async def test_equation_solver(question: str = "a+b=10 a-b=20"):
    """Test the equation solver directly"""
    try:
        from app.agents.web_search_agent import WebSearchAgent
        agent = WebSearchAgent()
        
        result = agent.search_math_solution(question)
        
        return {
            "question": question,
            "result": result,
            "solver_used": result.get('source') == 'equation_solver' if result else False
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve/enhanced", response_model=MathSolution)
async def solve_math_enhanced(question: MathQuestion):
    """Enhanced math solving with Agentic-RAG architecture (MCP + DSPy)"""
    try:
        # Try to import enhanced components
        try:
            from app.agents.enhanced_math_agent import EnhancedMathAgent
            enhanced_agent = EnhancedMathAgent()
            
            # Use enhanced agent if available
            result = enhanced_agent.process_question(question.question, math_rag)
            
            if result and result.get('status') != 'error':
                logger.info("✅ Enhanced agent processed successfully")
                solution_data = {
                    'question': question.question,
                    'steps': result['solution']['steps'],
                    'final_answer': result['solution']['final_answer'],
                    'source': result.get('source', 'enhanced_agent'),
                    'confidence': result.get('confidence', 0.8)
                }
                
                # Add educational context if available
                if result.get('educational_context'):
                    solution_data['educational_tips'] = result['solution'].get('learning_objectives', [])
                
                enhanced_solution = enhance_solution_with_display_source(solution_data)
                return MathSolution(**enhanced_solution)
                
        except ImportError as e:
            logger.warning(f"Enhanced agent not available, falling back to standard: {e}")
        
        # Fallback to standard routing
        return await solve_math_problem(question)
        
    except Exception as e:
        logger.error(f"Enhanced math solving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize benchmark components
try:
    from app.benchmark.jee_bench_loader import JEEBenchLoader
    from app.benchmark.evaluation_engine import JEEBenchEvaluationEngine
    jee_loader = JEEBenchLoader()
    evaluation_engine = None
except ImportError:
    logger.warning("JEE Bench components not available")
    jee_loader = None
    evaluation_engine = None

@app.post("/benchmark/jee/load")
async def load_jee_benchmark(limit: int = 50):
    """Load JEE Bench dataset for benchmarking"""
    try:
        if jee_loader is None:
            return {"status": "error", "message": "JEE Bench components not available"}
            
        success = jee_loader.load_dataset(limit=limit)
        
        if success:
            questions = jee_loader.get_questions()
            return {
                "status": "success",
                "loaded_questions": len(questions),
                "sample_questions": jee_loader.export_sample(5),
                "message": f"JEE Bench dataset loaded with {len(questions)} questions"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to load JEE Bench dataset"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark/jee/run")
async def run_jee_benchmark(question_limit: int = 10):
    """Run JEE Bench benchmark evaluation with error handling"""
    try:
        global evaluation_engine
        
        if jee_loader is None:
            return {"status": "error", "message": "JEE Bench components not available"}
        
        # Initialize evaluation engine if not exists
        if evaluation_engine is None:
            evaluation_engine = JEEBenchEvaluationEngine(
                math_agent=None,  # Will use main solve function
                routing_agent=routing_agent,
                rag_system=math_rag
            )
        
        # Load questions if not already loaded
        if not jee_loader.questions:
            jee_loader.load_dataset(limit=question_limit)
        
        questions = jee_loader.get_questions()[:question_limit]
        solutions = jee_loader.get_solutions()[:question_limit]
        
        logger.info(f"🚀 Starting JEE Bench evaluation with {len(questions)} questions")
        
        # Run benchmark
        metrics = await evaluation_engine.run_benchmark(questions, solutions)
        report = evaluation_engine.generate_report()
        
        # Ensure all float values are safe for JSON
        safe_report = make_json_safe(report)
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jee_benchmark_{timestamp}.json"
        evaluation_engine.export_results(filename)
        
        return {
            "status": "completed",
            "benchmark_report": safe_report,
            "results_file": filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark/jee/results")
async def get_benchmark_results():
    """Get latest benchmark results"""
    try:
        if evaluation_engine is None:
            return {"status": "no_results", "message": "No benchmark results available"}
        
        report = evaluation_engine.generate_report()
        safe_report = make_json_safe(report)
        return {
            "status": "success",
            "latest_results": safe_report
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/benchmark/jee/sample-questions")
async def get_sample_questions(count: int = 5):
    """Get sample JEE questions for testing"""
    try:
        if jee_loader is None or not jee_loader.questions:
            return {"status": "error", "message": "JEE Bench not loaded"}
        
        sample = jee_loader.export_sample(count)
        safe_sample = make_json_safe(sample)
        return {
            "sample_questions": safe_sample,
            "total_available": len(jee_loader.questions)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/benchmark/jee/compare-routing")
async def compare_routing_strategies(question_count: int = 10):
    """Compare routing strategies on JEE questions"""
    try:
        if jee_loader is None:
            return {"status": "error", "message": "JEE Bench components not available"}
            
        if not jee_loader.questions:
            jee_loader.load_dataset(limit=question_count)
        
        questions = jee_loader.get_questions()[:question_count]
        routing_analysis = []
        
        for q in questions:
            routing_result = routing_agent.process_question(q['question'], math_rag)
            
            routing_analysis.append({
                'question_id': q['id'],
                'question_topic': q['topic'],
                'question_difficulty': q['difficulty'],
                'routing_decision': routing_result['route_decision']['suggested_action'],
                'routing_confidence': routing_result['route_decision']['confidence'],
                'routing_reason': routing_result['route_decision'].get('reasoning', '')
            })
        
        safe_analysis = make_json_safe(routing_analysis)
        
        return {
            "routing_analysis": safe_analysis,
            "summary": {
                "total_questions": len(routing_analysis),
                "kb_decisions": sum(1 for r in routing_analysis if 'kb' in r['routing_decision']),
                "web_decisions": sum(1 for r in routing_analysis if 'web' in r['routing_decision']),
                "avg_confidence": np.mean([r['routing_confidence'] for r in routing_analysis])
            }
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Startup event to initialize with GSM8K data
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        # Log initial KB size
        initial_size = len(math_rag.kb_data)
        logger.info(f"🎯 System started with {initial_size} questions in knowledge base")
        
        # Auto-load GSM8K dataset if KB is small
        if initial_size < 100 and GSM8K_AVAILABLE:
            loaded_count = load_gsm8k_dataset(limit=500)  # Load 500 questions to start
            if loaded_count > 0:
                logger.info(f"📚 Knowledge base now has {len(math_rag.kb_data)} questions")
            else:
                logger.info("ℹ️ Using default knowledge base")
        elif not GSM8K_AVAILABLE:
            logger.info("📦 Install 'datasets' package for automatic GSM8K loading: pip install datasets")
                
        # Also try to load GS8MK dataset if file exists
        gs8mk_path = "app/knowledge_base/gs8mk_dataset.json"
        if os.path.exists(gs8mk_path):
            result = math_rag.load_dataset_file(gs8mk_path)
            logger.info(f"📚 Loaded GS8MK dataset: {result}")
        
        # Log AI Gateway status
        logger.info("🛡️ AI Gateway Guardrails: ACTIVE")
        logger.info("🎯 Enhanced Routing with MCP: ACTIVE")
        logger.info("📊 Vector Database: ACTIVE" if math_rag.vector_db_available else "📊 Vector Database: FALLBACK MODE")
        logger.info("🧠 DSPy Optimization: ACTIVE" if dspy_optimizer.lm else "🧠 DSPy Optimization: DISABLED")
        logger.info("🚀 Math Agent System v7.0 with VectorDB & DSPy: READY")
                
    except Exception as e:
        logger.warning(f"Startup initialization note: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)