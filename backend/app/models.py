from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import math

class FeedbackType(str, Enum):
    CORRECTION = "correction"
    RATING = "rating"
    EXPLANATION_IMPROVEMENT = "explanation_improvement"
    STEP_CORRECTION = "step_correction"
    FINAL_ANSWER_CORRECTION = "final_answer_correction"
    GENERAL_FEEDBACK = "general_feedback"

class MathQuestion(BaseModel):
    question: str
    user_id: Optional[str] = None

class Step(BaseModel):
    step_number: int
    explanation: str
    equation: Optional[str] = None
    
    # Fix: Add get method to prevent 'Step' object has no attribute 'get' error
    def get(self, key, default=None):
        return getattr(self, key, default)
    
    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        # Ensure all fields are properly serialized
        return {k: v for k, v in data.items() if v is not None}

class MathSolution(BaseModel):
    question: str
    steps: List[Step]
    final_answer: str
    source: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    display_source: Optional[str] = None
    gateway_approved: Optional[bool] = True
    quality_score: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            # Handle NaN values in confidence scores
            float: lambda v: 0.0 if v != v else v  # Convert NaN to 0.0
        }
    
    # Fix: Add get method for compatibility
    def get(self, key, default=None):
        return getattr(self, key, default)

class StepFeedback(BaseModel):
    step_number: int
    original_explanation: str
    corrected_explanation: Optional[str] = None
    original_equation: Optional[str] = None
    corrected_equation: Optional[str] = None
    feedback_notes: Optional[str] = None
    is_correct: bool = True

class HumanFeedback(BaseModel):
    question: str
    original_solution: MathSolution
    feedback_type: FeedbackType
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 scale
    corrected_solution: Optional[Dict] = None
    step_feedback: Optional[List[StepFeedback]] = None
    explanation_feedback: Optional[str] = None
    final_answer_correction: Optional[str] = None
    user_notes: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class FeedbackRequest(BaseModel):
    question: str
    solution: MathSolution
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = None
    user_id: Optional[str] = None

class LearningFeedback(BaseModel):
    question: str
    original_solution: MathSolution
    user_rating: int = Field(..., ge=1, le=5)
    user_feedback: Optional[str] = None
    corrected_solution: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class LearningCycle(BaseModel):
    cycle_id: str
    feedback_id: str
    question: str
    original_solution: Dict
    human_feedback: HumanFeedback
    corrected_solution: Optional[Dict] = None
    learning_applied: bool = False
    confidence_impact: float = 0.0
    improvements_made: List[str] = []
    kb_updated: bool = False
    timestamp: datetime = Field(default_factory=datetime.now)

class PerformanceMetrics(BaseModel):
    question_type: str
    success_rate: float = Field(ge=0.0, le=1.0)
    average_rating: float = Field(ge=0.0, le=5.0)
    total_feedback: int
    improvement_suggestions: List[str]
    
    class Config:
        json_encoders = {
            float: lambda v: 0.0 if v != v else v  # Handle NaN values
        }

class SystemEvaluation(BaseModel):
    accuracy_score: float = Field(ge=0.0, le=1.0)
    explanation_quality: float = Field(ge=0.0, le=1.0)
    step_clarity: float = Field(ge=0.0, le=1.0)
    final_answer_accuracy: float = Field(ge=0.0, le=1.0)
    user_satisfaction: float = Field(ge=0.0, le=1.0)
    learning_effectiveness: float = Field(ge=0.0, le=1.0)
    overall_health: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            float: lambda v: 0.0 if v != v else v  # Handle NaN values
        }

class FeedbackAnalytics(BaseModel):
    total_feedbacks: int
    feedback_types: Dict[str, int]
    severity_distribution: Dict[str, int]
    average_rating: float = Field(ge=0.0, le=5.0)
    learning_cycles_count: int
    kb_updates_count: int
    
    class Config:
        json_encoders = {
            float: lambda v: 0.0 if v != v else v  # Handle NaN values
        }

# Helper function to safely convert to float
def safe_float(value, default=0.0):
    """Safely convert value to float, handling NaN and None"""
    try:
        result = float(value)
        return default if math.isnan(result) else result
    except (TypeError, ValueError):
        return default

# Helper function to make any object JSON safe
def make_json_safe(obj):
    """Recursively make object safe for JSON serialization"""
    if isinstance(obj, float):
        return safe_float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    else:
        try:
            # Try to convert to string if it's a complex object
            return str(obj)
        except:
            return "unserializable_object"