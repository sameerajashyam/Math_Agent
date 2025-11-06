# app/benchmark/jee_bench_loader.py
import json
import logging
import os
import sys
from typing import List, Dict, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

class JEEBenchLoader:
    """JEE Bench dataset loader and processor with enhanced error handling"""
    
    def __init__(self):
        self.dataset = None
        self.questions = []
        self.solutions = []
        self.loaded = False
        
    def load_dataset(self, split: str = "test", limit: int = 100) -> bool:
        """Load JEE Bench dataset from HuggingFace with fallback options"""
        try:
            logger.info("📥 Loading JEE Bench dataset from HuggingFace...")
            
            # Try to import datasets
            try:
                from datasets import load_dataset
            except ImportError:
                logger.warning("🤗 datasets package not installed. Installing...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
                from datasets import load_dataset
            
            # Load the dataset
            self.dataset = load_dataset("lighteval/JEE-Bench", split=split)
            logger.info(f"📊 Dataset loaded: {len(self.dataset)} items")
            
            # Process questions and solutions
            self.questions = []
            self.solutions = []
            
            successful_loaded = 0
            for i, item in enumerate(self.dataset):
                if successful_loaded >= limit:
                    break
                    
                try:
                    question = self._extract_question(item)
                    solution = self._extract_solution(item)
                    
                    if question and solution and len(question.strip()) > 10:
                        self.questions.append({
                            'id': f"jee_{successful_loaded + 1}",
                            'question': question,
                            'metadata': self._extract_metadata(item),
                            'difficulty': self._classify_difficulty(question),
                            'topic': self._classify_topic(question)
                        })
                        self.solutions.append({
                            'id': f"jee_{successful_loaded + 1}",
                            'solution': solution,
                            'expected_answer': self._extract_expected_answer(item)
                        })
                        successful_loaded += 1
                        
                        # Log first question for verification
                        if successful_loaded == 1:
                            logger.info(f"🔍 First question sample: {question[:100]}...")
                            
                except Exception as e:
                    logger.warning(f"⚠️ Skipping item {i}: {e}")
                    continue
            
            if successful_loaded == 0:
                logger.warning("🤔 No questions loaded from HuggingFace, trying fallback...")
                return self._load_fallback_data(limit)
            
            self.loaded = True
            logger.info(f"✅ Successfully loaded {successful_loaded} JEE Bench questions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load JEE Bench from HuggingFace: {e}")
            return self._load_fallback_data(limit)
    
    def _load_fallback_data(self, limit: int) -> bool:
        """Load fallback sample data when HuggingFace fails"""
        try:
            logger.info("🔄 Loading fallback JEE sample data...")
            
            self.questions = [
                {
                    "id": "jee_1",
                    "question": "If x² + y² = 25 and xy = 12, find the value of x + y",
                    "metadata": {"subject": "algebra", "type": "problem_solving"},
                    "difficulty": "medium",
                    "topic": "algebra"
                },
                {
                    "id": "jee_2", 
                    "question": "Find the derivative of f(x) = 3x⁴ - 2x³ + 5x - 7",
                    "metadata": {"subject": "calculus", "type": "derivative"},
                    "difficulty": "easy",
                    "topic": "calculus"
                },
                {
                    "id": "jee_3",
                    "question": "Solve the equation: 2sin²x - 3sinx + 1 = 0 for x in [0, 2π]",
                    "metadata": {"subject": "trigonometry", "type": "equation_solving"},
                    "difficulty": "medium",
                    "topic": "trigonometry"
                },
                {
                    "id": "jee_4",
                    "question": "A bag contains 5 red and 3 blue balls. Two balls are drawn at random. Find the probability that both are red.",
                    "metadata": {"subject": "probability", "type": "probability"},
                    "difficulty": "medium",
                    "topic": "probability"
                },
                {
                    "id": "jee_5",
                    "question": "Find the value of ∫(3x² + 2x + 1) dx from 0 to 2",
                    "metadata": {"subject": "calculus", "type": "integration"},
                    "difficulty": "easy",
                    "topic": "calculus"
                },
                {
                    "id": "jee_6",
                    "question": "The sum of the first n terms of an AP is 3n² + 5n. Find the 25th term.",
                    "metadata": {"subject": "algebra", "type": "sequences"},
                    "difficulty": "medium",
                    "topic": "algebra"
                },
                {
                    "id": "jee_7",
                    "question": "Find the equation of the tangent to the curve y = x³ - 3x² + 2 at the point where x = 1.",
                    "metadata": {"subject": "calculus", "type": "tangent"},
                    "difficulty": "hard",
                    "topic": "calculus"
                },
                {
                    "id": "jee_8",
                    "question": "If A and B are two events such that P(A) = 0.4, P(B) = 0.3 and P(A∩B) = 0.2, find P(A∪B).",
                    "metadata": {"subject": "probability", "type": "probability"},
                    "difficulty": "easy",
                    "topic": "probability"
                },
                {
                    "id": "jee_9",
                    "question": "Solve the differential equation: dy/dx = x² + y²",
                    "metadata": {"subject": "calculus", "type": "differential_equation"},
                    "difficulty": "hard",
                    "topic": "calculus"
                },
                {
                    "id": "jee_10",
                    "question": "Find the area of the triangle with vertices (1,2), (3,4), and (5,1).",
                    "metadata": {"subject": "geometry", "type": "coordinate_geometry"},
                    "difficulty": "medium",
                    "topic": "geometry"
                }
            ]
            
            self.solutions = [
                {
                    "id": "jee_1",
                    "solution": {
                        "steps": [
                            {
                                "step_number": 1,
                                "explanation": "We know that (x + y)² = x² + y² + 2xy",
                                "equation": "(x + y)² = x² + y² + 2xy"
                            },
                            {
                                "step_number": 2,
                                "explanation": "Substitute given values: x² + y² = 25 and xy = 12",
                                "equation": "(x + y)² = 25 + 2×12 = 25 + 24 = 49"
                            },
                            {
                                "step_number": 3,
                                "explanation": "Take square root of both sides",
                                "equation": "x + y = ±√49 = ±7"
                            }
                        ],
                        "final_answer": "x + y = 7 or x + y = -7"
                    },
                    "expected_answer": "7 or -7"
                },
                {
                    "id": "jee_2",
                    "solution": {
                        "steps": [
                            {
                                "step_number": 1,
                                "explanation": "Apply power rule for differentiation: d/dx(xⁿ) = n·xⁿ⁻¹",
                                "equation": "f'(x) = d/dx(3x⁴) - d/dx(2x³) + d/dx(5x) - d/dx(7)"
                            },
                            {
                                "step_number": 2,
                                "explanation": "Differentiate each term separately",
                                "equation": "f'(x) = 12x³ - 6x² + 5 - 0"
                            }
                        ],
                        "final_answer": "f'(x) = 12x³ - 6x² + 5"
                    },
                    "expected_answer": "12x³ - 6x² + 5"
                },
                {
                    "id": "jee_4",
                    "solution": {
                        "steps": [
                            {
                                "step_number": 1,
                                "explanation": "Total number of balls = 5 red + 3 blue = 8 balls",
                                "equation": "Total = 8"
                            },
                            {
                                "step_number": 2,
                                "explanation": "Number of ways to choose 2 red balls from 5",
                                "equation": "C(5,2) = 5!/(2!·3!) = 10"
                            },
                            {
                                "step_number": 3,
                                "explanation": "Total number of ways to choose any 2 balls from 8",
                                "equation": "C(8,2) = 8!/(2!·6!) = 28"
                            },
                            {
                                "step_number": 4,
                                "explanation": "Probability = favorable outcomes / total outcomes",
                                "equation": "P = 10/28 = 5/14"
                            }
                        ],
                        "final_answer": "5/14"
                    },
                    "expected_answer": "5/14"
                }
            ]
            
            # Ensure we don't exceed limit
            self.questions = self.questions[:limit]
            self.solutions = self.solutions[:min(limit, len(self.solutions))]
            
            self.loaded = True
            logger.info(f"✅ Loaded {len(self.questions)} fallback JEE questions")
            return True
            
        except Exception as e:
            logger.error(f"❌ Fallback data loading failed: {e}")
            return False
    
    def _extract_question(self, item) -> str:
        """Extract question text from dataset item"""
        try:
            if 'input' in item and item['input']:
                return str(item['input'])
            elif 'question' in item and item['question']:
                return str(item['question'])
            elif 'text' in item and item['text']:
                return str(item['text'])
            elif 'problem' in item and item['problem']:
                return str(item['problem'])
            return ""
        except:
            return ""
    
    def _extract_solution(self, item) -> Dict:
        """Extract solution from dataset item"""
        try:
            solution = {
                'steps': [],
                'final_answer': ''
            }
            
            # Extract final answer
            if 'output' in item and item['output']:
                solution['final_answer'] = str(item['output'])
            elif 'answer' in item and item['answer']:
                solution['final_answer'] = str(item['answer'])
            elif 'target' in item and item['target']:
                solution['final_answer'] = str(item['target'])
            
            # Create basic step structure
            if solution['final_answer']:
                solution['steps'] = [
                    {
                        'step_number': 1,
                        'explanation': f"Solution approach for JEE problem leading to: {solution['final_answer']}",
                        'equation': ''
                    }
                ]
            else:
                # Create empty steps if no final answer
                solution['steps'] = [
                    {
                        'step_number': 1,
                        'explanation': "Step-by-step solution for the given problem",
                        'equation': ''
                    }
                ]
            
            return solution
        except:
            return {'steps': [], 'final_answer': ''}
    
    def _extract_expected_answer(self, item) -> str:
        """Extract expected answer for evaluation"""
        try:
            if 'output' in item and item['output']:
                return str(item['output'])
            elif 'answer' in item and item['answer']:
                return str(item['answer'])
            elif 'target' in item and item['target']:
                return str(item['target'])
            return ""
        except:
            return ""
    
    def _extract_metadata(self, item) -> Dict:
        """Extract metadata from dataset item"""
        metadata = {}
        try:
            if 'subject' in item:
                metadata['subject'] = item['subject']
            if 'type' in item:
                metadata['type'] = item['type']
            if 'year' in item:
                metadata['year'] = item['year']
            if 'difficulty' in item:
                metadata['difficulty'] = item['difficulty']
        except:
            pass
        return metadata
    
    def _classify_difficulty(self, question: str) -> str:
        """Classify question difficulty based on content"""
        if not question:
            return "medium"
            
        question_lower = question.lower()
        
        hard_indicators = ['prove', 'theorem', 'complex', 'advanced', 'derivative', 'integral', 
                          'differential', 'limit', 'continuous', 'differentiable', 'vector',
                          'matrix', 'eigenvalue', 'determinant']
        
        medium_indicators = ['solve', 'equation', 'find', 'calculate', 'expression', 'value',
                           'probability', 'ratio', 'percentage', 'area', 'volume']
        
        if any(word in question_lower for word in hard_indicators):
            return "hard"
        elif any(word in question_lower for word in medium_indicators):
            return "medium"
        else:
            return "easy"
    
    def _classify_topic(self, question: str) -> str:
        """Classify math topic based on question content"""
        if not question:
            return "general_math"
            
        question_lower = question.lower()
        
        topic_indicators = {
            "calculus": ['calculus', 'derivative', 'integral', 'limit', 'differentiable', 
                        'continuous', 'differential', 'tangent', 'curve'],
            "algebra": ['algebra', 'equation', 'polynomial', 'matrix', 'determinant',
                       'eigenvalue', 'linear', 'quadratic', 'root'],
            "geometry": ['geometry', 'triangle', 'circle', 'angle', 'area', 'volume',
                        'perimeter', 'coordinate', 'vertex', 'vertices'],
            "probability": ['probability', 'statistics', 'distribution', 'random',
                           'chance', 'likely', 'odds', 'dice', 'coin'],
            "trigonometry": ['trigonometry', 'sin', 'cos', 'tan', 'angle', 'triangle',
                           'sine', 'cosine', 'tangent']
        }
        
        for topic, indicators in topic_indicators.items():
            if any(word in question_lower for word in indicators):
                return topic
                
        return "general_math"
    
    def get_questions(self) -> List[Dict]:
        """Get all loaded questions"""
        return self.questions
    
    def get_solutions(self) -> List[Dict]:
        """Get all loaded solutions"""
        return self.solutions
    
    def get_question_by_id(self, question_id: str) -> Optional[Dict]:
        """Get specific question by ID"""
        for q in self.questions:
            if q['id'] == question_id:
                return q
        return None
    
    def get_solution_by_id(self, question_id: str) -> Optional[Dict]:
        """Get specific solution by ID"""
        for s in self.solutions:
            if s['id'] == question_id:
                return s
        return None
    
    def export_sample(self, count: int = 5) -> List[Dict]:
        """Export sample questions for testing"""
        samples = []
        for i in range(min(count, len(self.questions))):
            sample = self.questions[i].copy()
            # Add solution preview if available
            solution = self.get_solution_by_id(sample['id'])
            if solution:
                sample['solution_preview'] = solution.get('expected_answer', '')[:100]
            samples.append(sample)
        return samples
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded dataset"""
        if not self.loaded:
            return {"status": "not_loaded"}
        
        topics = {}
        difficulties = {}
        
        for q in self.questions:
            topic = q.get('topic', 'unknown')
            difficulty = q.get('difficulty', 'unknown')
            
            topics[topic] = topics.get(topic, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        return {
            "status": "loaded",
            "total_questions": len(self.questions),
            "total_solutions": len(self.solutions),
            "topics_distribution": topics,
            "difficulty_distribution": difficulties,
            "has_huggingface_data": self.dataset is not None
        }
    
    def save_to_file(self, filename: str = "jee_bench_questions.json"):
        """Save loaded questions to file for later use"""
        try:
            data = {
                "questions": self.questions,
                "solutions": self.solutions,
                "metadata": {
                    "total_questions": len(self.questions),
                    "loaded_at": pd.Timestamp.now().isoformat(),
                    "source": "huggingface" if self.dataset else "fallback"
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved {len(self.questions)} questions to {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save to file: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Test the loader
    loader = JEEBenchLoader()
    
    print("🧪 Testing JEE Bench Loader...")
    
    # Try loading from HuggingFace
    success = loader.load_dataset(limit=10)
    
    if success:
        stats = loader.get_stats()
        print(f"✅ Loaded: {stats}")
        
        samples = loader.export_sample(3)
        print(f"📝 Samples: {json.dumps(samples, indent=2)}")
    else:
        print("❌ Failed to load dataset")