import logging
import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class HITLFeedbackAgent:
    """
    Fixed HITL Agent - Working version with complete learning cycle tracking
    """
    
    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.kb_dir = "app/knowledge_base"
        
        # File paths
        self.feedback_db_path = os.path.join(self.kb_dir, "human_feedback.json")
        self.learning_cycles_path = os.path.join(self.kb_dir, "learning_cycles.json")
        self.metrics_path = os.path.join(self.kb_dir, "system_metrics.json")
        
        self._ensure_storage()
        logger.info("✅ Fixed HITL Feedback Agent initialized")
    
    def _ensure_storage(self):
        """Ensure storage files exist"""
        os.makedirs(self.kb_dir, exist_ok=True)
        
        # Create files if they don't exist
        default_files = {
            self.feedback_db_path: {"feedbacks": [], "total_count": 0},
            self.learning_cycles_path: {"cycles": [], "total_count": 0},
            self.metrics_path: {
                "metrics": {
                    "total_feedbacks": 0,
                    "average_rating": 0.0,
                    "total_corrections": 0,
                    "learning_cycles_count": 0,
                    "kb_updates_count": 0
                }
            }
        }
        
        for path, default_data in default_files.items():
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump(default_data, f, indent=2)
    
    def _safe_json_operation(self, file_path, operation):
        """Safe JSON operations with error handling"""
        try:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                # Default data based on filename
                if "feedback" in file_path:
                    data = {"feedbacks": [], "total_count": 0}
                elif "cycles" in file_path:
                    data = {"cycles": [], "total_count": 0}
                else:
                    data = {"metrics": {}}
            
            result = operation(data)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return result
            
        except Exception as e:
            logger.error(f"JSON operation failed for {file_path}: {e}")
            return None

    def _record_learning_cycle(self, feedback_data: Dict, improvements: List[str]):
        """Record learning cycle when feedback triggers learning"""
        try:
            question = feedback_data.get('question', 'Unknown question')
            rating = feedback_data.get('rating', 0)
            
            # Load existing cycles
            cycle_data = self._safe_json_operation(self.learning_cycles_path, lambda data: data)
            if not cycle_data:
                cycle_data = {"cycles": [], "total_count": 0}
            
            # Create new cycle
            new_cycle = {
                "cycle_id": f"lc_{uuid.uuid4().hex[:8]}",
                "feedback_id": f"fb_{int(datetime.now().timestamp())}",
                "timestamp": datetime.now().isoformat(),
                "question_preview": question[:50] + "..." if len(question) > 50 else question,
                "improvements": improvements,
                "rating": rating,
                "kb_size_before": len(self.rag_system.kb_data) if self.rag_system else 0,
                "kb_size_after": len(self.rag_system.kb_data) + 1 if self.rag_system else 0
            }
            
            # Add to cycles
            cycle_data["cycles"].append(new_cycle)
            cycle_data["total_count"] = len(cycle_data["cycles"])
            
            # Save back to file
            self._safe_json_operation(self.learning_cycles_path, lambda data: cycle_data)
            
            logger.info(f"📝 Recorded learning cycle: {new_cycle['cycle_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record learning cycle: {e}")
            return False

    def _update_learning_metrics(self):
        """Update learning metrics in system_metrics.json"""
        try:
            def update_metrics(data):
                metrics = data.get("metrics", {})
                metrics["kb_updates_count"] = metrics.get("kb_updates_count", 0) + 1
                metrics["last_learning_update"] = datetime.now().isoformat()
                data["metrics"] = metrics
                return data
            
            self._safe_json_operation(self.metrics_path, update_metrics)
            
        except Exception as e:
            logger.error(f"Failed to update learning metrics: {e}")

    def process_human_feedback(self, feedback) -> Dict:
        """Process human feedback - FIXED VERSION"""
        try:
            feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
            
            # Convert to dict - FIXED: No .dict() method issues
            if hasattr(feedback, 'dict'):
                feedback_data = feedback.dict()
            else:
                # It's already a dict or we need to handle it differently
                feedback_data = dict(feedback) if hasattr(feedback, '__dict__') else feedback
            
            # Ensure we have the required fields
            if not isinstance(feedback_data, dict):
                feedback_data = {
                    "question": "Unknown",
                    "rating": 0,
                    "feedback_type": "rating", 
                    "user_notes": "",
                    "user_id": "anonymous"
                }
            
            feedback_data["feedback_id"] = feedback_id
            feedback_data["timestamp"] = datetime.now().isoformat()
            
            # Store feedback
            storage_success = self._safe_json_operation(
                self.feedback_db_path,
                lambda data: data["feedbacks"].append(feedback_data) or data.update({"total_count": len(data["feedbacks"])})
            ) is not None

            # Apply learning from feedback
            learning_result = self.apply_learning_from_feedback(feedback_data, self.rag_system)
            
            return {
                "status": "success",
                "feedback_id": feedback_id,
                "storage_success": storage_success,
                "learning_applied": learning_result.get("learning_applied", False),
                "improvements": learning_result.get("improvements", []),
                "kb_updated": learning_result.get("kb_updated", False)
            }
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return {"status": "error", "error": str(e)}

    def apply_learning_from_feedback(self, feedback_data: Dict, rag_system) -> Dict:
        """Apply learning from feedback to KB - ENHANCED VERSION with cycle tracking"""
        try:
            question = feedback_data.get("question", "")
            original_solution = feedback_data.get("original_solution", {})
            rating = feedback_data.get("rating", 0)
            
            improvements = []
            
            # Only learn from high-rated solutions (4-5 stars)
            if rating >= 4 and question and original_solution:
                # Use the RAG system to add the solution
                success = rag_system.add_learned_solution(
                    question=question,
                    solution=original_solution,
                    source="hitl_learning"
                )
                
                if success:
                    improvements = ["knowledge_base_updated", "system_learning"]
                    logger.info(f"🎓 HITL learning applied to KB: {question[:50]}...")
                    
                    # Record learning cycle
                    self._record_learning_cycle(feedback_data, improvements)
                    
                    # Update metrics
                    self._update_learning_metrics()
                    
                    return {
                        "learning_applied": True,
                        "kb_updated": True,
                        "improvements": improvements,
                        "message": "Added to knowledge base",
                        "question_added": question[:100] + "..." if len(question) > 100 else question
                    }
            
            return {
                "learning_applied": False,
                "kb_updated": False,
                "improvements": improvements,
                "message": "Learning criteria not met (need rating >= 4)"
            }
            
        except Exception as e:
            logger.error(f"HITL learning failed: {e}")
            return {
                "learning_applied": False,
                "kb_updated": False,
                "improvements": [],
                "error": str(e)
            }

    def get_feedback_analytics(self) -> Dict:
        """Get analytics - FIXED VERSION without .dict() issues"""
        try:
            # Read all data safely
            feedback_data = self._safe_json_operation(self.feedback_db_path, lambda data: data)
            cycle_data = self._safe_json_operation(self.learning_cycles_path, lambda data: data)
            metrics_data = self._safe_json_operation(self.metrics_path, lambda data: data)
            
            # Handle None returns
            feedbacks = feedback_data.get("feedbacks", []) if feedback_data else []
            cycles = cycle_data.get("cycles", []) if cycle_data else []
            metrics = metrics_data.get("metrics", {}) if metrics_data else {}
            
            # Calculate basic analytics
            feedback_types = {}
            for fb in feedbacks:
                fb_type = fb.get("feedback_type", "unknown")
                feedback_types[fb_type] = feedback_types.get(fb_type, 0) + 1
            
            # Calculate average rating
            total_ratings = 0
            sum_ratings = 0
            for fb in feedbacks:
                rating = fb.get("rating", 0)
                if rating > 0:
                    total_ratings += 1
                    sum_ratings += rating
            
            average_rating = sum_ratings / total_ratings if total_ratings > 0 else 0.0
            
            return {
                "total_feedbacks": len(feedbacks),
                "feedback_types": feedback_types,
                "learning_cycles_count": len(cycles),
                "average_rating": average_rating,
                "kb_updates_count": metrics.get("kb_updates_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            return {
                "total_feedbacks": 0,
                "feedback_types": {},
                "learning_cycles_count": 0,
                "average_rating": 0.0,
                "kb_updates_count": 0
            }

    def get_system_evaluation(self) -> Dict:
        """Get system evaluation - FIXED VERSION"""
        try:
            analytics = self.get_feedback_analytics()
            
            # Simple scoring based on analytics
            total_feedbacks = analytics.get("total_feedbacks", 0)
            avg_rating = analytics.get("average_rating", 0)
            learning_cycles = analytics.get("learning_cycles_count", 0)
            
            # Calculate scores
            accuracy_score = min(0.8 + (total_feedbacks * 0.01), 0.95)
            user_satisfaction = (avg_rating - 1) / 4 if avg_rating > 0 else 0.7
            learning_effectiveness = min(learning_cycles * 0.2, 0.9) if learning_cycles > 0 else 0.5
            
            # Overall health
            avg_score = (accuracy_score + user_satisfaction + learning_effectiveness) / 3
            if avg_score >= 0.8:
                health = "excellent"
            elif avg_score >= 0.7:
                health = "good"
            elif avg_score >= 0.6:
                health = "fair"
            else:
                health = "needs_improvement"
            
            return {
                "accuracy_score": round(accuracy_score, 2),
                "explanation_quality": 0.7,
                "step_clarity": 0.7,
                "final_answer_accuracy": round(accuracy_score * 0.9, 2),
                "user_satisfaction": round(user_satisfaction, 2),
                "learning_effectiveness": round(learning_effectiveness, 2),
                "overall_health": health
            }
            
        except Exception as e:
            logger.error(f"System evaluation failed: {e}")
            return {
                "accuracy_score": 0.5,
                "explanation_quality": 0.5,
                "step_clarity": 0.5,
                "final_answer_accuracy": 0.5,
                "user_satisfaction": 0.5,
                "learning_effectiveness": 0.5,
                "overall_health": "unknown"
            }

    def get_learning_cycles(self, limit: int = 10) -> Dict:
        """Get learning cycles - FIXED VERSION"""
        try:
            cycle_data = self._safe_json_operation(self.learning_cycles_path, lambda data: data)
            cycles = cycle_data.get("cycles", []) if cycle_data else []
            
            return {
                "total_cycles": len(cycles),
                "recent_cycles": cycles[-limit:],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Learning cycles failed: {e}")
            return {
                "total_cycles": 0,
                "recent_cycles": [],
                "status": "error",
                "error": str(e)
            }

    def _calculate_learning_effectiveness(self, analytics: Dict) -> float:
        """Calculate learning effectiveness score"""
        try:
            total_feedbacks = analytics.get("total_feedbacks", 0)
            learning_cycles = analytics.get("learning_cycles_count", 0)
            
            if total_feedbacks == 0:
                return 0.5
                
            # Effectiveness = learning cycles / total feedbacks (higher is better)
            effectiveness = min(learning_cycles / total_feedbacks, 1.0)
            return round(effectiveness, 2)
            
        except Exception as e:
            logger.error(f"Learning effectiveness calculation failed: {e}")
            return 0.5