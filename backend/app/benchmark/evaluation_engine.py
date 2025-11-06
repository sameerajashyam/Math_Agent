# app/benchmark/evaluation_engine.py
import logging
import time
import asyncio
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class JEEBenchEvaluationEngine:
    """JEE Bench evaluation engine for Math Agent"""
    
    def __init__(self, math_agent, routing_agent, rag_system):
        self.math_agent = math_agent
        self.routing_agent = routing_agent
        self.rag_system = rag_system
        self.results = []
        self.metrics = {}
        
    async def run_benchmark(self, questions: List[Dict], solutions: List[Dict]) -> Dict:
        """Run comprehensive benchmark evaluation"""
        logger.info(f"🚀 Starting JEE Bench evaluation with {len(questions)} questions")
        
        start_time = time.time()
        results = []
        
        for i, question_data in enumerate(questions):
            try:
                logger.info(f"🔍 Evaluating question {i+1}/{len(questions)}: {question_data['id']}")
                
                # Run evaluation for single question
                result = await self._evaluate_single_question(question_data, solutions[i])
                results.append(result)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Progress: {i+1}/{len(questions)} questions evaluated")
                    
            except Exception as e:
                logger.error(f"❌ Error evaluating question {question_data['id']}: {e}")
                results.append({
                    'question_id': question_data['id'],
                    'error': str(e),
                    'success': False
                })
        
        # Calculate overall metrics
        self.results = results
        self.metrics = self._calculate_metrics(results)
        self.metrics['total_time'] = time.time() - start_time
        self.metrics['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ JEE Bench evaluation completed in {self.metrics['total_time']:.2f}s")
        return self.metrics
    
    async def _evaluate_single_question(self, question_data: Dict, solution_data: Dict) -> Dict:
        """Evaluate single question"""
        question_id = question_data['id']
        question_text = question_data['question']
        expected_solution = solution_data['solution']
        expected_answer = solution_data.get('expected_answer', '')
        
        result = {
            'question_id': question_id,
            'question_text': question_text[:100] + "..." if len(question_text) > 100 else question_text,
            'topic': question_data.get('topic', 'unknown'),
            'difficulty': question_data.get('difficulty', 'unknown'),
            'success': False,
            'metrics': {}
        }
        
        try:
            # Step 1: Test routing decision
            routing_start = time.time()
            routing_result = self.routing_agent.process_question(question_text, self.rag_system)
            routing_time = time.time() - routing_start
            
            result['routing_decision'] = routing_result['route_decision']['suggested_action']
            result['routing_confidence'] = routing_result['route_decision']['confidence']
            result['routing_time'] = routing_time
            
            # Step 2: Get solution from math agent
            solution_start = time.time()
            
            # Use the main solve endpoint
            from app.models import MathQuestion
            math_question = MathQuestion(question=question_text)
            
            # Import the solve function from main
            from app.main import solve_math_problem
            agent_solution = await solve_math_problem(math_question)
            
            solution_time = time.time() - solution_start
            
            result['solution_time'] = solution_time
            result['total_time'] = routing_time + solution_time
            result['agent_solution'] = agent_solution.dict()
            
            # Step 3: Evaluate solution quality
            evaluation = self._evaluate_solution_quality(
                agent_solution, expected_solution, expected_answer
            )
            
            result['evaluation'] = evaluation
            result['success'] = evaluation['overall_score'] > 0.5
            
            # Step 4: Calculate metrics
            result['metrics'] = {
                'routing_accuracy': self._evaluate_routing_accuracy(routing_result, question_data),
                'solution_quality': evaluation['overall_score'],
                'response_time': result['total_time'],
                'kb_used': 'knowledge_base' in result['agent_solution'].get('source', ''),
                'web_search_used': 'web_search' in result['agent_solution'].get('source', '')
            }
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {question_id}: {e}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def _evaluate_solution_quality(self, agent_solution, expected_solution, expected_answer: str) -> Dict:
        """Evaluate quality of generated solution"""
        try:
            agent_answer = agent_solution.final_answer
            agent_steps = agent_solution.steps
            
            # Basic answer matching
            answer_similarity = self._calculate_answer_similarity(agent_answer, expected_answer)
            
            # Step completeness evaluation
            step_quality = self._evaluate_step_completeness(agent_steps)
            
            # Educational value assessment
            educational_value = self._assess_educational_value(agent_steps)
            
            # Overall score (weighted average)
            overall_score = (
                answer_similarity * 0.4 +
                step_quality * 0.4 +
                educational_value * 0.2
            )
            
            return {
                'answer_similarity': answer_similarity,
                'step_quality': step_quality,
                'educational_value': educational_value,
                'overall_score': overall_score,
                'agent_answer': agent_answer,
                'expected_answer': expected_answer
            }
            
        except Exception as e:
            logger.error(f"Solution quality evaluation failed: {e}")
            return {
                'answer_similarity': 0,
                'step_quality': 0,
                'educational_value': 0,
                'overall_score': 0,
                'error': str(e)
            }
    
    def _calculate_answer_similarity(self, agent_answer: str, expected_answer: str) -> float:
        """Calculate similarity between agent answer and expected answer"""
        if not agent_answer or not expected_answer:
            return 0.0
        
        # Simple numerical comparison for math answers
        try:
            # Extract numbers from answers
            import re
            agent_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", agent_answer)
            expected_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
            
            if agent_numbers and expected_numbers:
                # Compare first number found
                agent_num = float(agent_numbers[0])
                expected_num = float(expected_numbers[0])
                
                if expected_num != 0:
                    similarity = 1 - abs(agent_num - expected_num) / abs(expected_num)
                    return max(0, min(1, similarity))
                
        except:
            pass
        
        # Fallback: string similarity
        agent_lower = agent_answer.lower()
        expected_lower = expected_answer.lower()
        
        if agent_lower == expected_lower:
            return 1.0
        elif expected_lower in agent_lower:
            return 0.8
        else:
            return 0.3
    
    def _evaluate_step_completeness(self, steps: List[Dict]) -> float:
        """Evaluate completeness of solution steps"""
        if not steps:
            return 0.0
        
        # Check step structure
        valid_steps = 0
        total_steps = len(steps)
        
        for step in steps:
            if (step.get('step_number') and 
                step.get('explanation') and 
                len(step.get('explanation', '')) > 10):
                valid_steps += 1
        
        step_completeness = valid_steps / total_steps if total_steps > 0 else 0
        
        # Bonus for multiple steps
        step_count_score = min(1.0, total_steps / 5)  # Max score for 5+ steps
        
        return (step_completeness * 0.7 + step_count_score * 0.3)
    
    def _assess_educational_value(self, steps: List[Dict]) -> float:
        """Assess educational value of solution"""
        if not steps:
            return 0.0
        
        educational_indicators = 0
        total_indicators = 4  # explanation, clarity, learning, context
        
        for step in steps:
            explanation = step.get('explanation', '').lower()
            
            # Check for educational indicators
            if len(explanation) > 20:  # Substantial explanation
                educational_indicators += 0.25
            if any(word in explanation for word in ['because', 'therefore', 'since', 'thus']):  # Reasoning
                educational_indicators += 0.25
            if any(word in explanation for word in ['step', 'method', 'approach', 'technique']):  # Methodology
                educational_indicators += 0.25
            if any(word in explanation for word in ['note', 'remember', 'important', 'key']):  # Learning points
                educational_indicators += 0.25
        
        return min(1.0, educational_indicators)
    
    def _evaluate_routing_accuracy(self, routing_result: Dict, question_data: Dict) -> float:
        """Evaluate routing decision accuracy"""
        routing_decision = routing_result['route_decision']['suggested_action']
        confidence = routing_result['route_decision']['confidence']
        
        # Simple heuristic: high confidence should correlate with KB usage
        if 'kb' in routing_decision and confidence > 0.7:
            return 0.9
        elif 'web' in routing_decision and confidence < 0.5:
            return 0.8
        else:
            return 0.6
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate overall benchmark metrics"""
        successful_results = [r for r in results if r.get('success', False)]
        total_questions = len(results)
        
        if total_questions == 0:
            return {}
        
        # Basic metrics
        success_rate = len(successful_results) / total_questions
        
        # Average scores
        avg_solution_quality = np.mean([r.get('evaluation', {}).get('overall_score', 0) 
                                      for r in successful_results])
        avg_routing_accuracy = np.mean([r.get('metrics', {}).get('routing_accuracy', 0) 
                                      for r in successful_results])
        avg_response_time = np.mean([r.get('total_time', 0) for r in successful_results])
        
        # Routing analysis
        kb_usage = sum(1 for r in successful_results 
                      if r.get('metrics', {}).get('kb_used', False))
        web_usage = sum(1 for r in successful_results 
                       if r.get('metrics', {}).get('web_search_used', False))
        
        # Topic performance
        topic_performance = {}
        for result in successful_results:
            topic = result.get('topic', 'unknown')
            score = result.get('evaluation', {}).get('overall_score', 0)
            if topic not in topic_performance:
                topic_performance[topic] = []
            topic_performance[topic].append(score)
        
        # Difficulty performance
        difficulty_performance = {}
        for result in successful_results:
            difficulty = result.get('difficulty', 'unknown')
            score = result.get('evaluation', {}).get('overall_score', 0)
            if difficulty not in difficulty_performance:
                difficulty_performance[difficulty] = []
            difficulty_performance[difficulty].append(score)
        
        return {
            'success_rate': success_rate,
            'avg_solution_quality': avg_solution_quality,
            'avg_routing_accuracy': avg_routing_accuracy,
            'avg_response_time': avg_response_time,
            'total_questions': total_questions,
            'successful_questions': len(successful_results),
            'routing_analysis': {
                'kb_usage': kb_usage,
                'web_usage': web_usage,
                'kb_usage_rate': kb_usage / len(successful_results) if successful_results else 0,
                'web_usage_rate': web_usage / len(successful_results) if successful_results else 0
            },
            'topic_performance': {topic: np.mean(scores) for topic, scores in topic_performance.items()},
            'difficulty_performance': {diff: np.mean(scores) for diff, scores in difficulty_performance.items()},
            'benchmark_version': '1.0',
            'dataset': 'JEE-Bench'
        }
    
    def generate_report(self) -> Dict:
        """Generate comprehensive benchmark report"""
        return {
            'summary': self.metrics,
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        if self.metrics.get('success_rate', 0) < 0.7:
            recommendations.append("Improve solution generation for complex JEE problems")
        
        if self.metrics.get('avg_solution_quality', 0) < 0.6:
            recommendations.append("Enhance step-by-step explanation quality")
        
        if self.metrics.get('avg_routing_accuracy', 0) < 0.7:
            recommendations.append("Optimize routing decisions for mathematical questions")
        
        kb_usage_rate = self.metrics.get('routing_analysis', {}).get('kb_usage_rate', 0)
        if kb_usage_rate < 0.3:
            recommendations.append("Expand knowledge base with JEE-level mathematical concepts")
        
        # Topic-specific recommendations
        topic_performance = self.metrics.get('topic_performance', {})
        for topic, score in topic_performance.items():
            if score < 0.6:
                recommendations.append(f"Focus on improving {topic} problem-solving")
        
        if not recommendations:
            recommendations.append("System performing well. Continue monitoring and incremental improvements")
        
        return recommendations
    
    def export_results(self, filepath: str):
        """Export benchmark results to JSON file"""
        import json
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Benchmark results exported to {filepath}")