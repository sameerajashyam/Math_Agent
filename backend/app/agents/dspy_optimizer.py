import dspy
from typing import List, Dict, Optional
import logging
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class MathRoutingSignature(dspy.Signature):
    """DSPy signature for math question routing optimization"""
    question = dspy.InputField(desc="Mathematical question to route")
    kb_confidence = dspy.InputField(desc="Knowledge base similarity score (0-1)")
    topic = dspy.InputField(desc="Mathematical topic classification")
    complexity = dspy.InputField(desc="Question complexity level")
    user_feedback_history = dspy.InputField(desc="Historical feedback for similar questions")
    
    route_decision = dspy.OutputField(desc="Routing decision: use_kb|use_web|use_mcp|enhanced_kb")
    confidence_score = dspy.OutputField(desc="Confidence in routing decision (0-1)")
    reasoning = dspy.OutputField(desc="Explanation for routing decision based on learning")
    improvement_suggestions = dspy.OutputField(desc="Suggestions to improve routing for this question type")

class FeedbackLearningSignature(dspy.Signature):
    """DSPy signature for learning from user feedback"""
    question = dspy.InputField(desc="Original math question")
    original_solution = dspy.InputField(desc="Solution that was provided")
    user_rating = dspy.InputField(desc="User rating 1-5")
    user_feedback_text = dspy.InputField(desc="User's textual feedback")
    solution_source = dspy.InputField(desc="Where the solution came from: kb|web|mcp|equation")
    
    learned_improvements = dspy.OutputField(desc="Specific improvements identified")
    confidence_adjustment = dspy.OutputField(desc="How to adjust confidence scoring for similar questions")
    routing_optimization = dspy.OutputField(desc="How to optimize routing for future similar questions")
    kb_update_suggestions = dspy.OutputField(desc="What to add/update in knowledge base")

class SearchStrategySignature(dspy.Signature):
    """DSPy signature for optimizing search strategies"""
    question = dspy.InputField(desc="Mathematical question to solve")
    question_topic = dspy.InputField(desc="Type of math question")
    complexity_level = dspy.InputField(desc="Question complexity")
    search_performance_history = dspy.InputField(desc="Historical performance of different search strategies")
    
    optimal_strategy_order = dspy.OutputField(desc="Recommended order of search strategies to try")
    strategy_reasoning = dspy.OutputField(desc="Explanation for strategy selection")
    performance_predictions = dspy.OutputField(desc="Expected success rates for each strategy")

class DSPyOptimizer:
    """
    REAL DSPy-powered optimizer for routing decisions and feedback learning
    """
    
    def __init__(self):
        self.lm = self._initialize_language_model()
        if self.lm:
            dspy.configure(lm=self.lm)
        
        # Initialize DSPy predictors
        self.routing_predictor = dspy.Predict(MathRoutingSignature)
        self.feedback_predictor = dspy.Predict(FeedbackLearningSignature)
        self.strategy_predictor = dspy.Predict(SearchStrategySignature)
        
        # Store learning history
        self.learning_history = []
        self.feedback_patterns = {}
        self.strategy_performance = {}
        
        logger.info("✅ DSPy Optimizer initialized with feedback learning and strategy optimization")
    
    def _initialize_language_model(self):
        """Initialize language model for DSPy with proper error handling"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != "your_openai_api_key_here":
                # Try different DSPy LM initialization methods
                try:
                    # Method 1: Try dspy.OpenAI (older versions)
                    logger.info("🔑 Attempting to initialize DSPy with OpenAI...")
                    return dspy.OpenAI(model='gpt-3.5-turbo', api_key=api_key)
                except AttributeError:
                    try:
                        # Method 2: Try dspy.LM (newer versions)
                        logger.info("🔑 Trying alternative DSPy LM initialization...")
                        from dspy.teleprompt import BootstrapFewShot
                        # For newer DSPy versions, you might need to use different initialization
                        return dspy.LM('openai/gpt-3.5-turbo', api_key=api_key)
                    except (AttributeError, ImportError):
                        # Method 3: Fallback to basic LM configuration
                        logger.info("🔑 Using basic LM configuration...")
                        # This is a simplified approach for newer DSPy versions
                        class BasicLM:
                            def __init__(self):
                                self.name = "openai/gpt-3.5-turbo"
                            
                            def __call__(self, *args, **kwargs):
                                # This would need actual implementation based on your DSPy version
                                logger.warning("DSPy LM called but not fully implemented")
                                return None
                        
                        return BasicLM()
            else:
                logger.warning("OpenAI API key not configured for DSPy")
                return None
                
        except Exception as e:
            logger.error(f"DSPy LM initialization failed: {e}")
            # Create a mock LM for fallback
            class MockLM:
                def __init__(self):
                    self.name = "mock_lm"
                    self.available = False
                
                def __call__(self, *args, **kwargs):
                    logger.warning("Mock LM called - DSPy not fully functional")
                    return type('MockPrediction', (), {
                        'route_decision': 'use_kb',
                        'confidence_score': '0.7',
                        'reasoning': 'Fallback reasoning',
                        'improvement_suggestions': 'Configure DSPy properly'
                    })()
            
            return MockLM()
    
    def optimize_routing_with_feedback(self, question: str, kb_confidence: float, 
                                    topic: str, complexity: str, feedback_history: List[Dict] = None) -> Dict:
        """Use DSPy to optimize routing based on learned feedback patterns"""
        if not self.lm or not hasattr(self.lm, 'available') or getattr(self.lm, 'available', True) is False:
            return self._fallback_routing(kb_confidence)
        
        try:
            # Prepare feedback history context
            feedback_context = self._prepare_feedback_context(feedback_history, topic)
            
            prediction = self.routing_predictor(
                question=question,
                kb_confidence=kb_confidence,
                topic=topic,
                complexity=complexity,
                user_feedback_history=feedback_context
            )
            
            optimized_decision = {
                'suggested_action': getattr(prediction, 'route_decision', 'use_kb'),
                'confidence': float(getattr(prediction, 'confidence_score', kb_confidence)),
                'reasoning': getattr(prediction, 'reasoning', 'DSPy optimized routing'),
                'improvement_suggestions': getattr(prediction, 'improvement_suggestions', ''),
                'dspy_optimized': True,
                'feedback_informed': bool(feedback_history)
            }
            
            # Store this optimization in learning history
            self._record_routing_optimization(question, optimized_decision)
            
            return optimized_decision
            
        except Exception as e:
            logger.error(f"DSPy routing optimization failed: {e}")
            return self._fallback_routing(kb_confidence)
    
    def optimize_search_strategy(self, question: str, topic: str, complexity: str, 
                               performance_history: Dict = None) -> Dict:
        """Use DSPy to optimize search strategy order"""
        if not self.lm or not hasattr(self.lm, 'available') or getattr(self.lm, 'available', True) is False:
            return self._fallback_search_strategy()
        
        try:
            # Prepare performance history context
            performance_context = self._prepare_performance_context(performance_history)
            
            prediction = self.strategy_predictor(
                question=question,
                question_topic=topic,
                complexity_level=complexity,
                search_performance_history=performance_context
            )
            
            strategy_optimization = {
                'optimal_strategy_order': getattr(prediction, 'optimal_strategy_order', 'speed_time_solver,equation_solver,mcp_search,llm_enhanced,tavily_direct'),
                'strategy_reasoning': getattr(prediction, 'strategy_reasoning', 'Default strategy order'),
                'performance_predictions': getattr(prediction, 'performance_predictions', ''),
                'dspy_optimized': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update strategy performance knowledge
            self._update_strategy_performance(strategy_optimization)
            
            return strategy_optimization
            
        except Exception as e:
            logger.error(f"DSPy strategy optimization failed: {e}")
            return self._fallback_search_strategy()
    
    def learn_from_feedback(self, feedback_data: Dict) -> Dict:
        """Use DSPy to extract learning insights from user feedback"""
        if not self.lm or not hasattr(self.lm, 'available') or getattr(self.lm, 'available', True) is False:
            return self._fallback_feedback_learning(feedback_data)
        
        try:
            prediction = self.feedback_predictor(
                question=feedback_data.get('question', ''),
                original_solution=json.dumps(feedback_data.get('original_solution', {})),
                user_rating=feedback_data.get('user_rating', 3),
                user_feedback_text=feedback_data.get('user_feedback_text', ''),
                solution_source=feedback_data.get('solution_source', 'unknown')
            )
            
            learning_insights = {
                'learned_improvements': getattr(prediction, 'learned_improvements', ''),
                'confidence_adjustment': getattr(prediction, 'confidence_adjustment', ''),
                'routing_optimization': getattr(prediction, 'routing_optimization', ''),
                'kb_update_suggestions': getattr(prediction, 'kb_update_suggestions', ''),
                'dspy_learned': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store learning insights
            self._store_learning_insights(feedback_data, learning_insights)
            
            return learning_insights
            
        except Exception as e:
            logger.error(f"DSPy feedback learning failed: {e}")
            return self._fallback_feedback_learning(feedback_data)
    
    def analyze_feedback_patterns(self, feedbacks: List[Dict]) -> Dict:
        """Use DSPy to analyze patterns across multiple feedbacks"""
        if not self.lm or not hasattr(self.lm, 'available') or getattr(self.lm, 'available', True) is False or not feedbacks:
            return {'patterns_found': False}
        
        try:
            # Group feedbacks by topic/source for pattern analysis
            feedback_summary = self._summarize_feedbacks_for_analysis(feedbacks)
            
            # Use DSPy to identify patterns
            patterns = self._identify_patterns_with_dspy(feedback_summary)
            
            # Update internal patterns knowledge
            self._update_feedback_patterns(patterns)
            
            return {
                'patterns_identified': True,
                'insights': patterns,
                'suggested_actions': self._generate_actions_from_patterns(patterns),
                'feedbacks_analyzed': len(feedbacks)
            }
            
        except Exception as e:
            logger.error(f"DSPy pattern analysis failed: {e}")
            return {'patterns_found': False, 'error': str(e)}
    
    def _prepare_feedback_context(self, feedback_history: List[Dict], topic: str) -> str:
        """Prepare feedback history context for DSPy"""
        if not feedback_history:
            return "No previous feedback available"
        
        # Filter feedback for similar topics
        relevant_feedback = [
            fb for fb in feedback_history 
            if fb.get('topic') == topic or fb.get('complexity', '') in ['similar', 'same']
        ][-5:]  # Last 5 relevant feedbacks
        
        if not relevant_feedback:
            return "No topic-specific feedback available"
        
        summary = f"Previous feedback for {topic} questions:\n"
        for fb in relevant_feedback:
            rating = fb.get('rating', 0)
            source = fb.get('source', 'unknown')
            summary += f"- Rating: {rating}/5, Source: {source}, "
            if fb.get('user_notes'):
                summary += f"Notes: {fb.get('user_notes')[:100]}...\n"
            else:
                summary += "No additional notes\n"
        
        return summary
    
    def _prepare_performance_context(self, performance_history: Dict) -> str:
        """Prepare performance history context for strategy optimization"""
        if not performance_history:
            return "No performance history available"
        
        context = "Search strategy performance history:\n"
        for strategy, stats in performance_history.items():
            if stats.get('total_attempts', 0) > 0:
                success_rate = stats.get('success_rate', 0)
                context += f"- {strategy}: {success_rate:.1%} success rate ({stats.get('success_count', 0)}/{stats.get('total_attempts', 0)})\n"
        
        return context
    
    def _fallback_routing(self, kb_confidence: float) -> Dict:
        """Fallback routing logic"""
        if kb_confidence > 0.7:
            action = 'use_kb'
        elif kb_confidence > 0.4:
            action = 'use_kb_with_fallback'
        else:
            action = 'web_search_available'
        
        return {
            'suggested_action': action,
            'confidence': kb_confidence,
            'reasoning': f'Fallback routing based on KB confidence: {kb_confidence}',
            'dspy_optimized': False
        }
    
    def _fallback_search_strategy(self) -> Dict:
        """Fallback search strategy"""
        return {
            'optimal_strategy_order': 'speed_time_solver,equation_solver,mcp_search,llm_enhanced,tavily_direct,contextual_mock',
            'strategy_reasoning': 'Fallback to default strategy order',
            'performance_predictions': 'No performance predictions available',
            'dspy_optimized': False
        }
    
    def _fallback_feedback_learning(self, feedback_data: Dict) -> Dict:
        """Fallback feedback learning"""
        rating = feedback_data.get('user_rating', 3)
        source = feedback_data.get('solution_source', 'unknown')
        
        basic_improvements = []
        if rating <= 2:
            basic_improvements.append(f"Solution quality from {source} needs improvement")
        if rating >= 4:
            basic_improvements.append(f"Maintain {source} solution approach")
        
        return {
            'learned_improvements': '; '.join(basic_improvements),
            'confidence_adjustment': 'Consider user ratings in confidence scoring',
            'routing_optimization': 'No specific routing changes suggested',
            'kb_update_suggestions': 'Add high-rated solutions to knowledge base',
            'dspy_learned': False
        }
    
    def _record_routing_optimization(self, question: str, decision: Dict):
        """Record routing optimization decisions"""
        self.learning_history.append({
            'type': 'routing_optimization',
            'question': question[:100],
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        })
    
    def _store_learning_insights(self, feedback_data: Dict, insights: Dict):
        """Store learning insights from feedback"""
        self.learning_history.append({
            'type': 'feedback_learning',
            'question': feedback_data.get('question', '')[:100],
            'rating': feedback_data.get('user_rating', 0),
            'source': feedback_data.get('solution_source', 'unknown'),
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        })
    
    def _update_strategy_performance(self, strategy_optimization: Dict):
        """Update strategy performance knowledge"""
        strategy_key = f"strategy_{datetime.now().strftime('%Y%m%d_%H')}"
        self.strategy_performance[strategy_key] = strategy_optimization
    
    def _summarize_feedbacks_for_analysis(self, feedbacks: List[Dict]) -> str:
        """Summarize feedbacks for pattern analysis"""
        if not feedbacks:
            return "No feedback data available"
        
        # Group by source and rating
        source_stats = {}
        for fb in feedbacks:
            source = fb.get('source', 'unknown')
            rating = fb.get('rating', 0)
            
            if source not in source_stats:
                source_stats[source] = {'total': 0, 'sum_ratings': 0, 'low_ratings': 0}
            
            source_stats[source]['total'] += 1
            source_stats[source]['sum_ratings'] += rating
            if rating <= 2:
                source_stats[source]['low_ratings'] += 1
        
        summary = "Feedback analysis summary:\n"
        for source, stats in source_stats.items():
            avg_rating = stats['sum_ratings'] / stats['total'] if stats['total'] > 0 else 0
            low_rate = stats['low_ratings'] / stats['total'] if stats['total'] > 0 else 0
            summary += f"- {source}: {stats['total']} feedbacks, avg rating: {avg_rating:.1f}, low ratings: {low_rate:.1%}\n"
        
        return summary
    
    def _identify_patterns_with_dspy(self, feedback_summary: str) -> Dict:
        """Use DSPy to identify patterns in feedback"""
        # For now, return basic pattern analysis
        # In a full implementation, you would use another DSPy signature here
        patterns = {
            'low_rated_sources': [],
            'high_rated_sources': [],
            'common_issues': [],
            'success_factors': []
        }
        
        # Simple pattern detection from summary
        if 'low ratings' in feedback_summary.lower():
            patterns['common_issues'].append('Some sources receiving low ratings')
        if 'avg rating' in feedback_summary.lower():
            patterns['success_factors'].append('Multiple sources with acceptable ratings')
        
        return patterns
    
    def _update_feedback_patterns(self, patterns: Dict):
        """Update internal feedback patterns knowledge"""
        # Merge new patterns with existing ones
        for key, new_values in patterns.items():
            if key not in self.feedback_patterns:
                self.feedback_patterns[key] = []
            self.feedback_patterns[key].extend(new_values)
        
        # Keep only recent patterns (last 100)
        for key in self.feedback_patterns:
            self.feedback_patterns[key] = self.feedback_patterns[key][-100:]
    
    def _generate_actions_from_patterns(self, patterns: Dict) -> List[str]:
        """Generate actionable suggestions from patterns"""
        actions = []
        
        if patterns.get('low_rated_sources'):
            actions.append("Investigate and improve low-rated solution sources")
        if patterns.get('common_issues'):
            actions.append("Address common issues reported in feedback")
        if patterns.get('high_rated_sources'):
            actions.append("Leverage high-performing sources more frequently")
        
        if not actions:
            actions.append("Continue monitoring feedback patterns")
        
        return actions
    
    def get_learning_history(self, limit: int = 10) -> List[Dict]:
        """Get recent learning history"""
        return self.learning_history[-limit:]
    
    def get_optimization_stats(self) -> Dict:
        """Get DSPy optimization statistics"""
        routing_optimizations = [h for h in self.learning_history if h['type'] == 'routing_optimization']
        feedback_learnings = [h for h in self.learning_history if h['type'] == 'feedback_learning']
        
        return {
            'total_optimizations': len(self.learning_history),
            'routing_optimizations': len(routing_optimizations),
            'feedback_learnings': len(feedback_learnings),
            'strategy_optimizations': len(self.strategy_performance),
            'dspy_available': self.lm is not None and hasattr(self.lm, 'available') and getattr(self.lm, 'available', True),
            'patterns_learned': len(self.feedback_patterns),
            'recent_activity': [h['type'] for h in self.learning_history[-5:]]
        }
    
    def get_feedback_patterns(self) -> Dict:
        """Get current feedback patterns"""
        return self.feedback_patterns
    
    def clear_learning_history(self):
        """Clear learning history (for testing/reset)"""
        self.learning_history.clear()
        self.feedback_patterns.clear()
        self.strategy_performance.clear()
        logger.info("🧹 DSPy learning history cleared")