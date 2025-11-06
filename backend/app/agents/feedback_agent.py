import logging
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
from ..models import LearningFeedback

logger = logging.getLogger(__name__)

class FeedbackAgent:
    def __init__(self):
        self.feedback_file = "app/knowledge_base/feedback_data.json"
        self.metrics_file = "app/knowledge_base/performance_metrics.json"
        self._ensure_feedback_files()
        
    def _ensure_feedback_files(self):
        """Ensure feedback files exist"""
        os.makedirs("app/knowledge_base", exist_ok=True)
        
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump({"feedbacks": [], "total_count": 0}, f)
        
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'w') as f:
                json.dump({"metrics": {}}, f)
    
    def process_feedback(self, feedback: LearningFeedback) -> Dict:
        """Process user feedback and extract learning insights"""
        try:
            logger.info(f"📊 Processing feedback - Rating: {feedback.user_rating}")
            
            # Store the feedback
            self._store_feedback(feedback)
            
            # Analyze for patterns
            analysis = self._analyze_feedback(feedback)
            
            # Update performance metrics
            self._update_metrics(analysis)
            
            # Generate improvement suggestions
            suggestions = self._generate_improvement_suggestions(feedback, analysis)
            
            return {
                "status": "processed",
                "analysis": analysis,
                "suggestions": suggestions,
                "total_feedbacks": self._get_total_feedbacks()
            }
            
        except Exception as e:
            logger.error(f"Feedback processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _store_feedback(self, feedback: LearningFeedback):
        """Store feedback in JSON file"""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            
            feedback_dict = {
                "question": feedback.question,
                "user_rating": feedback.user_rating,
                "user_feedback": feedback.user_feedback,
                "source": feedback.original_solution.source,
                "confidence": feedback.original_solution.confidence,
                "timestamp": feedback.timestamp.isoformat(),
                "corrected_solution": feedback.corrected_solution
            }
            
            data["feedbacks"].append(feedback_dict)
            data["total_count"] = len(data["feedbacks"])
            
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
    
    def _analyze_feedback(self, feedback: LearningFeedback) -> Dict:
        """Analyze feedback for patterns"""
        rating = feedback.user_rating
        source = feedback.original_solution.source
        confidence = feedback.original_solution.confidence
        
        analysis = {
            "rating_category": "positive" if rating >= 4 else "neutral" if rating == 3 else "negative",
            "source_performance": source,
            "confidence_vs_rating": "high_confidence_good_rating" if confidence > 0.7 and rating >= 4 else
                                   "high_confidence_bad_rating" if confidence > 0.7 and rating <= 2 else
                                   "low_confidence_good_rating" if confidence < 0.5 and rating >= 4 else
                                   "low_confidence_bad_rating" if confidence < 0.5 and rating <= 2 else "mixed",
            "needs_improvement": rating <= 3,
            "question_complexity": self._assess_complexity(feedback.question)
        }
        
        return analysis
    
    def _assess_complexity(self, question: str) -> str:
        """Assess question complexity"""
        question_lower = question.lower()
        
        complex_terms = ['quantum', 'fourier', 'laplace', 'differential', 'partial derivative']
        intermediate_terms = ['calculus', 'derivative', 'integral', 'matrix', 'vector']
        
        if any(term in question_lower for term in complex_terms):
            return "advanced"
        elif any(term in question_lower for term in intermediate_terms):
            return "intermediate"
        else:
            return "basic"
    
    def _update_metrics(self, analysis: Dict):
        """Update performance metrics"""
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            question_type = analysis["question_complexity"]
            
            if question_type not in metrics:
                metrics[question_type] = {
                    "total_ratings": 0,
                    "sum_ratings": 0,
                    "positive_feedbacks": 0,
                    "negative_feedbacks": 0
                }
            
            metrics[question_type]["total_ratings"] += 1
            metrics[question_type]["sum_ratings"] += analysis.get("user_rating", 3)
            
            if analysis["rating_category"] == "positive":
                metrics[question_type]["positive_feedbacks"] += 1
            elif analysis["rating_category"] == "negative":
                metrics[question_type]["negative_feedbacks"] += 1
            
            data["metrics"] = metrics
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
    
    def _generate_improvement_suggestions(self, feedback: LearningFeedback, analysis: Dict) -> List[str]:
        """Generate improvement suggestions based on feedback"""
        suggestions = []
        
        if analysis["needs_improvement"]:
            if analysis["confidence_vs_rating"] == "high_confidence_bad_rating":
                suggestions.append("High confidence but low rating - consider adjusting confidence thresholds")
            
            if feedback.original_solution.source == "knowledge_base":
                suggestions.append("KB solution received low rating - consider enhancing knowledge base")
            
            if feedback.original_solution.source == "web_search":
                suggestions.append("Web search result needs improvement - consider better result filtering")
        
        return suggestions[:3]
    
    def _get_total_feedbacks(self) -> int:
        """Get total number of feedbacks"""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            return data.get("total_count", 0)
        except:
            return 0
    
    def get_all_feedbacks(self) -> List[Dict]:
        """Get all stored feedbacks"""
        try:
            with open(self.feedback_file, 'r') as f:
                data = json.load(f)
            return data.get("feedbacks", [])
        except:
            return []
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            report = {}
            
            for question_type, stats in metrics.items():
                total = stats["total_ratings"]
                if total > 0:
                    report[question_type] = {
                        "average_rating": stats["sum_ratings"] / total,
                        "success_rate": stats["positive_feedbacks"] / total * 100,
                        "total_feedbacks": total
                    }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {}