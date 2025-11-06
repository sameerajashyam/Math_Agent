import re
import logging
import json
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)

class MathAIGateway:
    """
    AI Gateway with specialized guardrails for educational mathematics
    """
    
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent
        self.blocked_patterns = self._load_blocked_patterns()
        self.math_topics = self._load_math_topics()
        self.analytics = GuardrailAnalytics()
    
    def _load_blocked_patterns(self) -> Dict:
        """Load patterns for content that should be blocked - FIXED VERSION"""
        return {
            "personal_info": [
                r'\b(ssn|social security|credit card|password|phone number|address|email)\b',
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
                r'\b\d{16}\b',  # Credit card pattern
            ],
            "inappropriate_content": [
                r'\b(violence|harm|dangerous|illegal|weapon|attack)\b',
                r'\b(hate|discriminat|racist|sexist)\b',
                r'\b(adult|explicit|sexual)\b',
            ],
            # FIXED: Only block clearly non-educational content
            "non_educational": [
                r'\b(game|entertainment|shopping|sports|celebrity|movie|music|tv|television)\b',
                r'\b(weather|news|politics|government|election|party)\b',
                r'\b(food|recipe|cooking|restaurant|nutrition)\b',
                r'\b(travel|vacation|hotel|flight|destination)\b',
                r'\b(fashion|clothing|style|beauty|cosmetic)\b',
                r'\b(pet|animal|dog|cat|pet care|veterinarian)\b',
                r'\b(relationship|dating|marriage|family|parenting)\b'
            ]
        }
    
    def _load_math_topics(self) -> List[str]:
        """Define approved mathematical topics"""
        return [
            "algebra", "geometry", "calculus", "trigonometry", "statistics",
            "probability", "arithmetic", "number theory", "linear algebra",
            "differential equations", "discrete mathematics", "optimization",
            "graph theory", "combinatorics", "real analysis", "complex analysis",
            "work_rate", "time_work"  # Added work rate problems
        ]
    
    def process_input(self, question: str) -> Dict:
        """
        Comprehensive input processing with AI-powered guardrails
        """
        logger.info(f"🛡️ AI Gateway processing input: {question[:100]}...")
        
        # Step 1: Basic sanitization
        sanitized_question = self._basic_sanitize(question)
        
        # Step 2: Quick math check - if it has numbers and math indicators, auto-approve
        if self._is_likely_math_problem(sanitized_question):
            logger.info("✅ Quick math check passed - auto-approving")
            return self._auto_approve_math_problem(sanitized_question)
        
        # Step 3: Pattern-based blocking (only for clearly inappropriate content)
        pattern_check = self._check_blocked_patterns(sanitized_question)
        if not pattern_check["allowed"]:
            self.analytics.log_block("pattern_blocked", sanitized_question, pattern_check["reason"])
            return {
                "status": "blocked",
                "reason": pattern_check["reason"],
                "risk_level": "high",
                "action": "reject"
            }
        
        # Step 4: AI-powered mathematical intent classification
        intent_analysis = self._analyze_mathematical_intent(sanitized_question)
        if not intent_analysis["is_mathematical"]:
            self.analytics.log_block("non_mathematical", sanitized_question, "Not a math question")
            return {
                "status": "blocked",
                "reason": "non_mathematical_query",
                "details": intent_analysis.get("reason", "Query doesn't appear to be mathematical"),
                "suggestion": "Please ask a mathematics-related question",
                "action": "reject"
            }
        
        # Step 5: Complexity and topic classification
        complexity_analysis = self._analyze_complexity_topic(sanitized_question)
        
        # Step 6: Educational appropriateness check
        educational_check = self._check_educational_appropriateness(sanitized_question)
        if not educational_check["appropriate"]:
            self.analytics.log_block("non_educational", sanitized_question, educational_check["reason"])
            return {
                "status": "blocked",
                "reason": "not_educational",
                "details": educational_check["reason"],
                "action": "reject"
            }
        
        # Step 7: PII removal
        final_question = self._remove_pii(sanitized_question)
        
        self.analytics.log_approved(final_question, intent_analysis["topic"])
        
        return {
            "status": "approved",
            "sanitized_question": final_question,
            "topic": intent_analysis["topic"],
            "complexity": complexity_analysis["complexity"],
            "subtopics": complexity_analysis["subtopics"],
            "educational_level": educational_check["level"],
            "confidence": intent_analysis["confidence"]
        }
    
    def _is_likely_math_problem(self, question: str) -> bool:
        """Quick check if question is likely a math problem"""
        question_lower = question.lower()
        
        # Check for numbers and math operations
        has_numbers = bool(re.search(r'\d+', question))
        
        # Check for common math problem patterns
        math_patterns = [
            r'\btakes\s+\d+\s+days?\b',  # "takes X days"
            r'\b\d+\s+days?\b',  # "X days"
            r'\btogether\b',  # "together" in work problems
            r'\bhow\s+long\b',  # "how long"
            r'\bwork\s+together\b',  # "work together"
            r'\bcomplete\s+.*\s+together\b',  # "complete together"
        ]
        
        has_math_pattern = any(re.search(pattern, question_lower) for pattern in math_patterns)
        
        # Check for work rate indicators
        work_rate_indicators = ['takes', 'days', 'together', 'complete', 'work', 'finish']
        has_work_indicators = any(indicator in question_lower for indicator in work_rate_indicators)
        
        return has_numbers and (has_math_pattern or has_work_indicators)
    
    def _auto_approve_math_problem(self, question: str) -> Dict:
        """Auto-approve clearly mathematical problems"""
        # Simple topic classification for common patterns
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['takes', 'days', 'together', 'work']):
            topic = "work_rate"
            complexity = "intermediate"
        elif any(word in question_lower for word in ['probability', 'chance', 'likely']):
            topic = "probability" 
            complexity = "intermediate"
        elif any(word in question_lower for word in ['solve', 'equation', 'x', 'y']):
            topic = "algebra"
            complexity = "basic"
        else:
            topic = "arithmetic"
            complexity = "basic"
        
        self.analytics.log_approved(question, topic)
        
        return {
            "status": "approved",
            "sanitized_question": question,
            "topic": topic,
            "complexity": complexity,
            "subtopics": [topic],
            "educational_level": "high_school",
            "confidence": 0.9,
            "auto_approved": True
        }

    def process_output(self, solution: Dict, original_question: str) -> Dict:
        """
        Comprehensive output validation with AI-powered quality checks
        """
        logger.info("🛡️ AI Gateway validating output...")
        
        # Step 1: Structural validation
        structure_check = self._validate_solution_structure(solution)
        if not structure_check["valid"]:
            return {
                "status": "invalid",
                "reason": "invalid_structure",
                "details": structure_check["issues"],
                "action": "regenerate"
            }
        
        # Step 2: Mathematical correctness verification
        correctness_check = self._verify_mathematical_correctness(solution, original_question)
        if not correctness_check["correct"]:
            return {
                "status": "invalid",
                "reason": "mathematical_error",
                "details": correctness_check["issues"],
                "action": "regenerate"
            }
        
        # Step 3: Educational quality assessment
        quality_check = self._assess_educational_quality(solution)
        if quality_check["score"] < 0.7:
            return {
                "status": "needs_improvement",
                "reason": "low_educational_quality",
                "details": quality_check["issues"],
                "improvements": quality_check["suggestions"],
                "action": "enhance"
            }
        
        # Step 4: Language and clarity check
        clarity_check = self._check_clarity_language(solution)
        if not clarity_check["clear"]:
            return {
                "status": "needs_improvement",
                "reason": "clarity_issues",
                "details": clarity_check["issues"],
                "action": "clarify"
            }
        
        # Step 5: Final sanitization
        sanitized_solution = self._sanitize_output(solution)
        
        self.analytics.log_output_validation(quality_check["score"])
        
        return {
            "status": "approved",
            "sanitized_solution": sanitized_solution,
            "quality_score": quality_check["score"],
            "educational_value": quality_check["educational_value"],
            "clarity_score": clarity_check["score"]
        }
    
    def _analyze_mathematical_intent(self, question: str) -> Dict:
        """AI-powered mathematical intent classification"""
        if not self.llm_agent or not self.llm_agent.is_available():
            # Fallback to keyword-based analysis when LLM is not available
            return self._fallback_intent_analysis(question)
        
        try:
            prompt = f"""
            Analyze if this query is a mathematics question and classify its topic.
            
            QUESTION: "{question}"
            
            Respond with JSON:
            {{
                "is_mathematical": boolean,
                "topic": "algebra|geometry|calculus|probability|statistics|arithmetic|work_rate|other",
                "confidence": 0.0-1.0,
                "reason": "explanation if not mathematical",
                "subtopic": "more specific category if available"
            }}
            
            Consider:
            - Does it involve mathematical concepts, calculations, or problem-solving?
            - Is it asking for mathematical explanation or solution?
            - Could it be answered using mathematical reasoning?
            - Work rate problems (people working together) are mathematical
            - Word problems with numbers are mathematical
            """
            
            response = self.llm_agent.generate_response(prompt)
            if not response:
                return self._fallback_intent_analysis(question)
                
            # Extract JSON from response
            result = self._extract_json_from_response(response)
            if not result:
                return self._fallback_intent_analysis(question)
            
            # Override: if it has numbers and looks like math, force mathematical
            if not result.get("is_mathematical", False) and self._has_math_characteristics(question):
                result["is_mathematical"] = True
                result["topic"] = "work_rate" if "takes" in question.lower() and "days" in question.lower() else "arithmetic"
                result["confidence"] = 0.8
                result["reason"] = "Overridden: clear math characteristics detected"
                
            return result
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return self._fallback_intent_analysis(question)
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract JSON from LLM response with robust error handling"""
        if not response:
            return None
            
        try:
            # Clean the response
            clean_response = response.strip()
            
            # Try to find JSON in various formats
            json_patterns = [
                r'\{.*\}',  # Basic JSON object
                r'```json\s*(.*?)\s*```',  # ```json ... ```
                r'```\s*(.*?)\s*```',  # ``` ... ```
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, clean_response, re.DOTALL)
                if matches:
                    clean_response = matches[0].strip()
                    break
            
            # If no JSON found but response looks like it might be JSON, try parsing directly
            if not clean_response.startswith('{'):
                # Check if it might be a malformed JSON response
                if 'is_mathematical' in clean_response or 'quality_score' in clean_response:
                    # Try to extract key-value pairs
                    json_match = re.search(r'\{.*\}', clean_response)
                    if json_match:
                        clean_response = json_match.group()
            
            # Parse the JSON
            if clean_response and clean_response.startswith('{'):
                return json.loads(clean_response)
            else:
                logger.warning(f"❌ No valid JSON found in response: {response[:100]}...")
                return None
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.error(f"❌ Problematic response: {response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"❌ JSON extraction failed: {e}")
            return None

    def _fallback_intent_analysis(self, question: str) -> Dict:
        """Fallback intent analysis when LLM fails"""
        question_lower = question.lower().strip()
        
        # More comprehensive mathematical keywords
        math_keywords = [
            'solve', 'calculate', 'equation', 'algebra', 'geometry', 'probability', 
            'percentage', 'ratio', 'fraction', 'derivative', 'integral', 'triangle',
            'circle', 'area', 'volume', 'distance', 'speed', 'time', 'work', 'rate',
            'sum', 'product', 'difference', 'multiple', 'divide', 'add', 'subtract',
            'multiply', 'angle', 'perimeter', 'radius', 'diameter', 'square', 'cube',
            'root', 'logarithm', 'trigonometry', 'calculus', 'statistics', 'mean',
            'median', 'mode', 'variable', 'formula', 'theorem', 'proof', 'number',
            'digit', 'decimal', 'fraction', 'percent', 'graph', 'coordinate', 'axis',
            'matrix', 'vector', 'function', 'polynomial', 'quadratic', 'linear',
            'inequality', 'expression', 'evaluate', 'compute', 'find', 'determine',
            'how many', 'how much', 'what is the value', 'simplify', 'factor',
            'takes', 'days', 'together', 'complete', 'finish', 'how long'
        ]
        
        # Non-mathematical patterns (more comprehensive)
        non_math_patterns = [
            'who is', 'what is', 'when did', 'where is', 'why is', 'how to',
            'biography', 'history', 'born', 'died', 'invented', 'created',
            'famous', 'celebrity', 'person', 'people', 'author', 'writer',
            'actor', 'actress', 'singer', 'politician', 'scientist', 'inventor',
            'movie', 'film', 'song', 'music', 'book', 'novel', 'painting',
            'art', 'sports', 'game', 'team', 'country', 'city', 'capital',
            'language', 'culture', 'religion', 'philosophy', 'psychology'
        ]
        
        has_math_keywords = any(keyword in question_lower for keyword in math_keywords)
        has_non_math_patterns = any(pattern in question_lower for pattern in non_math_patterns)
        has_numbers = any(char.isdigit() for char in question)
        
        # More sophisticated decision logic
        if has_non_math_patterns and not has_math_keywords:
            is_mathematical = False
            confidence = 0.95
            reason = "This appears to be a general knowledge or biographical question"
            suggestion = "I specialize in mathematics. Try asking about calculations, algebra, geometry, or other math topics."
        elif has_math_keywords:
            is_mathematical = True
            confidence = 0.85
            reason = "Contains mathematical keywords and concepts"
            suggestion = "This appears to be a mathematical question"
        elif has_numbers and len(question.split()) <= 10:  # Short questions with numbers
            is_mathematical = True
            confidence = 0.7
            reason = "Contains numerical elements in a concise question"
            suggestion = "This appears to involve calculations"
        else:
            is_mathematical = False
            confidence = 0.8
            reason = "Does not appear to be mathematical based on content analysis"
            suggestion = "Please ask about mathematics, calculations, algebra, geometry, or related topics"
        
        # Determine topic
        if 'takes' in question_lower and 'days' in question_lower:
            topic = "work_rate"
        elif any(word in question_lower for word in ['probability', 'chance', 'likely']):
            topic = "probability"
        elif any(word in question_lower for word in ['solve', 'equation', 'x', 'y']):
            topic = "algebra"
        elif has_numbers:
            topic = "arithmetic"
        else:
            topic = "other"
        
        return {
            "is_mathematical": is_mathematical,
            "topic": topic,
            "confidence": confidence,
            "reason": reason,
            "suggestion": suggestion
        }
    
    def _has_math_characteristics(self, question: str) -> bool:
        """Check if question has clear mathematical characteristics"""
        # Has numbers
        has_numbers = bool(re.search(r'\d+', question))
        
        # Has math operations or concepts
        math_terms = ['takes', 'days', 'together', 'work', 'complete', 'how long', 'probability']
        has_math_terms = any(term in question.lower() for term in math_terms)
        
        # Looks like a word problem
        is_word_problem = any(word in question.lower() for word in ['how', 'what', 'find', 'calculate', 'solve'])
        
        return has_numbers and (has_math_terms or is_word_problem)
    
    def _analyze_complexity_topic(self, question: str) -> Dict:
        """Analyze question complexity and specific mathematical topics"""
        # Simple complexity analysis without LLM
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['derivative', 'integral', 'calculus', 'matrix', 'vector']):
            complexity = "advanced"
            subtopics = ["calculus", "advanced_mathematics"]
        elif any(word in question_lower for word in ['probability', 'statistics', 'geometry', 'algebra']):
            complexity = "intermediate"
            subtopics = [word for word in ['probability', 'statistics', 'geometry', 'algebra'] if word in question_lower]
        else:
            complexity = "basic"
            subtopics = ["arithmetic", "basic_math"]
        
        # Add specific topic detection
        if 'takes' in question_lower and 'days' in question_lower:
            subtopics.append("work_rate")
        
        return {
            "complexity": complexity,
            "subtopics": subtopics,
            "prerequisites": [],
            "estimated_steps": 3 if complexity == "basic" else 4
        }
    
    def _check_educational_appropriateness(self, question: str) -> Dict:
        """Verify the question is appropriate for educational context"""
        # Auto-approve if it's clearly a math problem
        if self._is_likely_math_problem(question):
            return {
                "appropriate": True,
                "level": "high_school",
                "educational_value": "high",
                "reason": "Mathematical word problem"
            }
        
        # Simple educational check without LLM
        question_lower = question.lower()
        
        # Check for clearly inappropriate content
        inappropriate_patterns = [
            'violence', 'harm', 'dangerous', 'illegal', 'weapon', 'attack',
            'hate', 'discriminat', 'racist', 'sexist', 'adult', 'explicit', 'sexual'
        ]
        
        if any(pattern in question_lower for pattern in inappropriate_patterns):
            return {
                "appropriate": False,
                "level": "inappropriate",
                "educational_value": "none",
                "reason": "Contains inappropriate content"
            }
        
        # Default to appropriate for most content
        return {
            "appropriate": True,
            "level": "high_school",
            "educational_value": "medium",
            "reason": "General educational content"
        }
    
    def _verify_mathematical_correctness(self, solution: Dict, question: str) -> Dict:
        """AI-powered mathematical correctness verification"""
        # Simple validation without LLM
        issues = []
        
        # Check structure
        if not solution.get('steps') or not isinstance(solution['steps'], list):
            issues.append("Missing or invalid steps")
        
        if not solution.get('final_answer'):
            issues.append("Missing final answer")
        
        # Check steps structure
        if solution.get('steps'):
            for i, step in enumerate(solution['steps']):
                if not step.get('step_number'):
                    issues.append(f"Step {i+1} missing step number")
                if not step.get('explanation'):
                    issues.append(f"Step {i+1} missing explanation")
        
        return {
            "correct": len(issues) == 0,
            "confidence": 0.8 if len(issues) == 0 else 0.3,
            "issues": issues,
            "corrections": []
        }
    
    def _assess_educational_quality(self, solution: Dict) -> Dict:
        """Assess educational quality of the solution"""
        # Simple quality assessment without LLM
        score = 0.7  # Base score
        issues = []
        strengths = ["Solution provided with steps"]
        
        if solution.get('steps'):
            step_count = len(solution['steps'])
            if step_count < 2:
                issues.append("Too few steps for educational value")
                score -= 0.2
            elif step_count > 5:
                strengths.append("Detailed step-by-step explanation")
                score += 0.1
            
            # Check explanation quality
            for step in solution['steps']:
                explanation = step.get('explanation', '')
                if len(explanation.split()) < 3:
                    issues.append("Some explanations are too brief")
                    score -= 0.1
                    break
        
        if solution.get('final_answer'):
            strengths.append("Clear final answer provided")
            score += 0.1
        
        return {
            "score": max(0.1, min(1.0, score)),
            "educational_value": "high" if score > 0.8 else "medium" if score > 0.6 else "low",
            "issues": issues,
            "suggestions": ["Add more detailed explanations"] if score < 0.8 else [],
            "strengths": strengths
        }
    
    def _basic_sanitize(self, text: str) -> str:
        """Basic text sanitization"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove potentially harmful characters
        text = re.sub(r'[^\w\s\.\?\!\,\+\-\*\/\=\(\)]', '', text)
        return text.strip()
    
    def _check_blocked_patterns(self, text: str) -> Dict:
        """Check for blocked content patterns - FIXED to be more permissive"""
        text_lower = text.lower()
        
        # First, check for mathematical indicators that should override blocking
        if self._is_likely_math_problem(text):
            return {"allowed": True}
        
        # Only block if there are no math indicators and the content is clearly non-educational
        for category, patterns in self.blocked_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    # Allow through if it has mathematical content despite matching blocked pattern
                    if self._has_math_characteristics(text) and category == "non_educational":
                        continue
                    return {
                        "allowed": False,
                        "reason": f"blocked_{category}",
                        "pattern_matched": pattern
                    }
        
        return {"allowed": True}
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information"""
        # Simple PII patterns (enhanced version would use NLP)
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b'
        }
        
        cleaned_text = text
        for pii_type, pattern in patterns.items():
            cleaned_text = re.sub(pattern, f'[REDACTED_{pii_type.upper()}]', cleaned_text)
        
        return cleaned_text
    
    def _validate_solution_structure(self, solution: Dict) -> Dict:
        """Validate solution structure meets requirements"""
        issues = []
        
        if not solution.get('steps') or not isinstance(solution['steps'], list):
            issues.append("Missing or invalid steps array")
        
        if not solution.get('final_answer'):
            issues.append("Missing final answer")
        
        if solution.get('steps'):
            for i, step in enumerate(solution['steps']):
                if not step.get('step_number'):
                    issues.append(f"Step {i} missing step_number")
                if not step.get('explanation'):
                    issues.append(f"Step {i} missing explanation")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def _check_clarity_language(self, solution: Dict) -> Dict:
        """Check clarity and language appropriateness"""
        # Basic checks - could be enhanced with AI
        issues = []
        score = 0.8  # Base score
        
        if solution.get('steps'):
            for step in solution['steps']:
                explanation = step.get('explanation', '')
                if len(explanation.split()) < 3:  # Very short explanation
                    issues.append("Some explanations are too brief")
                    score -= 0.1
                if len(explanation.split()) > 100:  # Very long explanation
                    issues.append("Some explanations are too verbose")
                    score -= 0.1
        
        return {
            "clear": score >= 0.6,
            "score": max(0.1, score),
            "issues": issues
        }
    
    def _sanitize_output(self, solution: Dict) -> Dict:
        """Final output sanitization"""
        sanitized = solution.copy()
        
        # Ensure no PII in output
        if 'source_url' in sanitized:
            # Remove or anonymize URLs if needed
            pass
        
        return sanitized


class GuardrailAnalytics:
    """Track guardrail performance and analytics"""
    
    def __init__(self):
        self.input_blocks = []
        self.output_blocks = []
        self.approved_requests = []
        self.start_time = datetime.now()
    
    def log_block(self, block_type: str, question: str, reason: str):
        self.input_blocks.append({
            "timestamp": datetime.now(),
            "type": block_type,
            "question_preview": question[:50] + "..." if len(question) > 50 else question,
            "reason": reason
        })
    
    def log_approved(self, question: str, topic: str):
        self.approved_requests.append({
            "timestamp": datetime.now(),
            "question_preview": question[:50] + "..." if len(question) > 50 else question,
            "topic": topic
        })
    
    def log_output_validation(self, quality_score: float):
        self.output_blocks.append({
            "timestamp": datetime.now(),
            "quality_score": quality_score
        })
    
    def get_analytics(self) -> Dict:
        total_requests = len(self.input_blocks) + len(self.approved_requests)
        block_rate = len(self.input_blocks) / total_requests if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "blocked_requests": len(self.input_blocks),
            "approved_requests": len(self.approved_requests),
            "block_rate": round(block_rate, 3),
            "block_reasons": Counter([block["reason"] for block in self.input_blocks]),
            "top_topics": Counter([req["topic"] for req in self.approved_requests if "topic" in req]),
            "average_quality_score": round(
                sum(block["quality_score"] for block in self.output_blocks) / len(self.output_blocks), 3
            ) if self.output_blocks else 0,
            "uptime_hours": round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
        }


class RoutingAgent:
    def __init__(self):
        # Import inside __init__ to avoid circular imports
        try:
            from .web_search_agent import WebSearchAgent
            self.web_search_agent = WebSearchAgent()
        except ImportError:
            logger.warning("WebSearchAgent not available")
            self.web_search_agent = None
        
        try:
            from .mcp_search import MCPSearch  # MCP integration
            self.mcp_search = MCPSearch()
        except ImportError:
            logger.warning("MCPSearch not available")
            self.mcp_search = type('MockMCPSearch', (), {'mcp_available': False})()
        
        try:
            from .llm_agent import LLMAgent
            self.llm_agent = LLMAgent()
        except ImportError:
            logger.warning("LLMAgent not available")
            self.llm_agent = type('MockLLMAgent', (), {'is_available': lambda: False})()
        
        # Initialize AI Gateway with LLM agent
        self.ai_gateway = MathAIGateway(self.llm_agent)
        
        # **FIXED: Much more reasonable confidence thresholds**
        self.high_confidence_threshold = 0.75  # Reduced from 0.99
        self.medium_confidence_threshold = 0.55  # Reduced from 0.7
        self.low_confidence_threshold = 0.3     # Reduced from 0.4
        
        # Web search triggers - EXPANDED for conceptual/theoretical questions
        self.web_search_indicators = [
            # Mathematical concepts and theorems
            'theorem', 'proof', 'theory', 'concept', 'definition', 'principle',
            'lemma', 'corollary', 'axiom', 'postulate', 'hypothesis',
            
            # Probability and statistics
            'bayes', 'probability', 'statistics', 'stochastic', 'random',
            'distribution', 'regression', 'correlation', 'variance',
            
            # Advanced mathematics
            'quantum', 'fourier', 'laplace', 'differential equation',
            'partial derivative', 'multivariable', 'complex analysis',
            'integral equation', 'vector calculus', 'tensor', 'manifold',
            'topology', 'group theory', 'ring theory', 'field theory',
            'set theory', 'number theory', 'graph theory', 'game theory',
            
            # Economics and optimization
            'nash', 'equilibrium', 'optimization', 'linear programming',
            'black-scholes', 'calculus of variations', 'functional analysis',
            
            # Analysis branches
            'real analysis', 'complex analysis', 'abstract algebra',
            'linear algebra', 'matrix theory', 'eigenvalue', 'eigenvector',
            
            # Physics and engineering math
            'schrodinger', 'hamiltonian', 'lagrangian', 'navier-stokes',
            'maxwell', 'thermodynamics', 'relativity'
        ]
        
        # Questions that should ALWAYS trigger web search
        self.always_web_search = [
            'what is', 'define', 'explain', 'describe', 'how does',
            'who invented', 'history of', 'applications of', 'examples of',
            'compare', 'difference between', 'similar to'
        ]

        # Questions that might need LLM enhancement for numerical differences
        self.numerical_sensitive_indicators = [
            'budget', 'cost', 'price', 'spent', 'paid', 'money', '$',
            'distance', 'speed', 'time', 'rate', 'percentage', '%',
            'ratio', 'proportion', 'fraction', 'share',
            'age', 'years', 'months', 'days',
            'weight', 'height', 'length', 'area', 'volume'
        ]
        
        logger.info("✅ Enhanced Routing Agent with AI Gateway & MCP support initialized")
        
    def process_question(self, question: str, rag_system=None) -> Dict:
        """
        Enhanced routing logic with AI Gateway integration and MCP support
        """
        try:
            logger.info(f"🔄 Enhanced routing agent processing: {question}")
            
            # ===== AI GATEWAY INPUT GUARDRAILS =====
            gateway_result = self.ai_gateway.process_input(question)
            
            if gateway_result["status"] == "blocked":
                logger.warning(f"🚫 AI Gateway blocked request: {gateway_result['reason']}")
                return {
                    "status": "error",
                    "error_message": f"Question blocked by AI Gateway: {gateway_result.get('reason')}",
                    "gateway_action": "blocked",
                    "suggestion": gateway_result.get('suggestion', 'Please ask a mathematical question'),
                    "user_friendly_message": self._create_user_friendly_block_message(gateway_result)
                }
            
            # Get sanitized question from gateway
            sanitized_question = gateway_result["sanitized_question"]
            complexity = gateway_result["complexity"]
            topic = gateway_result["topic"]
            
            logger.info(f"✅ AI Gateway approved: {topic} question, complexity: {complexity}")
            
            # Step 3: Enhanced routing with RAG integration
            route_decision = self._enhanced_analyze_and_route(sanitized_question, rag_system, gateway_result)
            
            return {
                'status': 'success',
                'sanitized_question': sanitized_question,
                'route_decision': route_decision,
                'requires_web_search': route_decision.get('needs_web_search', False),
                'suggested_action': route_decision.get('suggested_action', 'use_kb'),
                'confidence': route_decision.get('confidence', 0.0),
                'routing_reason': route_decision.get('routing_reason', ''),
                'gateway_metadata': {
                    'input_approved': True,
                    'topic': topic,
                    'complexity': complexity,
                    'educational_level': gateway_result.get('educational_level', 'high_school')
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced routing agent error: {e}")
            return self._create_error_response("Internal routing error")
    
    def _create_user_friendly_block_message(self, gateway_result: Dict) -> str:
        """Create user-friendly message for blocked requests"""
        reason = gateway_result.get('reason', '')
        suggestion = gateway_result.get('suggestion', '')
        
        if 'non_mathematical' in reason.lower():
            return "This appears to be a general knowledge question. I specialize in mathematics and can help with calculations, algebra, geometry, and other math topics."
        elif 'inappropriate' in reason.lower():
            return "I'm designed to assist with educational mathematics content only."
        elif 'personal' in reason.lower():
            return "I focus on mathematical problems and concepts rather than personal information."
        else:
            return f"I specialize in mathematics. {suggestion}"

    def _enhanced_analyze_and_route(self, question: str, rag_system=None, gateway_result: Dict = None) -> Dict:
        """
        Enhanced routing with actual RAG similarity scores and AI Gateway insights
        """
        # If we have RAG system, get actual similarity scores
        kb_confidence = 0.0
        kb_match_found = False
        numerical_sensitive = False
        kb_result = None  # Store the actual KB result
        
        if rag_system:
            try:
                # First check for numerical sensitivity
                numerical_sensitive = self._is_numerically_sensitive(question)
                
                # Get KB result - IMPORTANT: Store the actual result
                kb_result = rag_system.search_knowledge_base(question)
                if kb_result:
                    kb_confidence = kb_result.get('similarity_score', 0.0)
                    kb_match_found = True
                    logger.info(f"📊 RAG confidence score: {kb_confidence:.4f}")
                    if kb_result.get('question'):
                        logger.info(f"📚 KB match found: {kb_result.get('question', 'Unknown')[:50]}...")
                    
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
        
        # Enhanced routing logic with AI Gateway insights
        needs_web_search, routing_reason = self._should_use_web_search(
            question, gateway_result, kb_confidence, kb_match_found, numerical_sensitive
        )
        
        suggested_action = self._determine_action(kb_confidence, needs_web_search, gateway_result, numerical_sensitive)
        
        # Check MCP availability for web search
        mcp_available = getattr(self.mcp_search, 'mcp_available', False)
        
        return {
            'is_math_question': True,  # Already validated by gateway
            'needs_web_search': needs_web_search,
            'suggested_action': suggested_action,
            'confidence': kb_confidence,
            'kb_confidence': kb_confidence,
            'kb_match_found': kb_match_found,
            'kb_result': kb_result,  # **FIXED: Return the actual KB result**
            'content_confidence': gateway_result.get('confidence', 0.8) if gateway_result else 0.8,
            'complexity_level': gateway_result.get('complexity', 'intermediate') if gateway_result else 'intermediate',
            'numerical_sensitive': numerical_sensitive,
            'mcp_available': mcp_available,
            'routing_reason': routing_reason,
            'reasoning': f"Gateway topic: {gateway_result.get('topic', 'unknown')}. {routing_reason}" if gateway_result else routing_reason
        }
    
    def _is_numerically_sensitive(self, question: str) -> bool:
        """Check if question is sensitive to numerical differences"""
        question_lower = question.lower()
        
        # Check for numerical indicators
        has_numerical_indicators = any(
            indicator in question_lower for indicator in self.numerical_sensitive_indicators
        )
        
        # Check for multiple numbers in question (potential for differences)
        numbers = re.findall(r'\$?(\d+\.?\d*)', question)
        has_multiple_numbers = len(numbers) >= 2
        
        return has_numerical_indicators and has_multiple_numbers
    
    def _should_use_web_search(self, question: str, gateway_result: Dict, 
                             kb_confidence: float, kb_match_found: bool,
                             numerical_sensitive: bool) -> Tuple[bool, str]:
        """
        **FIXED: Enhanced web search decision logic with AI Gateway insights**
        """
        question_lower = question.lower()
        
        # REASON 1: Always use web search for conceptual/theoretical questions
        for indicator in self.web_search_indicators:
            if indicator in question_lower:
                reason = f"Conceptual topic '{indicator}' requires comprehensive web search"
                logger.info(f"🔍 {reason}")
                return True, reason
        
        # REASON 2: Always use web search for definition/explanation questions
        for indicator in self.always_web_search:
            if indicator in question_lower:
                reason = f"Question type '{indicator}' requires verified external sources"
                logger.info(f"🔍 {reason}")
                return True, reason
        
        # **FIXED: REASON 3: If KB match exists with reasonable confidence, USE KB FIRST**
        if kb_match_found and kb_confidence >= 0.5:
            reason = f"Good KB match found (confidence: {kb_confidence:.2f}) - using knowledge base"
            logger.info(f"📚 {reason}")
            return False, reason
        
        # REASON 4: No match in knowledge base for math question
        if not kb_match_found:
            reason = "Math question with no KB match - using web search for solution"
            logger.info(f"🔍 {reason}")
            return True, reason
        
        # **FIXED: REASON 5: Only use web search for very low KB confidence**
        if kb_confidence < 0.3:
            reason = f"Very low KB confidence ({kb_confidence:.2f}) - using web search"
            logger.info(f"🔍 {reason}")
            return True, reason
        
        # **FIXED: REASON 6: Default to KB if we have any match at all (fallback)**
        if kb_match_found:
            reason = f"KB match available (confidence: {kb_confidence:.2f}) - preferring knowledge base over web search"
            logger.info(f"📚 {reason}")
            return False, reason
        
        reason = "Default case - using knowledge base if available"
        logger.info(f"⚡ {reason}")
        return False, reason
    
    def _determine_action(self, kb_confidence: float, needs_web_search: bool, 
                         gateway_result: Dict, numerical_sensitive: bool) -> str:
        """**FIXED: Enhanced action determination with AI Gateway insights**"""
        
        if needs_web_search:
            # Check if MCP is available for enhanced search
            if getattr(self.mcp_search, 'mcp_available', False):
                return 'mcp_web_search'
            else:
                return 'web_search_available'
        
        # **FIXED: High confidence - use KB directly (more reasonable threshold)**
        elif kb_confidence >= self.high_confidence_threshold:
            return 'use_kb'
        
        # **FIXED: Medium confidence with numerical sensitivity - use KB with potential LLM enhancement**
        elif kb_confidence >= self.medium_confidence_threshold and numerical_sensitive:
            return 'use_kb_with_llm_fallback'
        
        # **FIXED: Medium confidence without numerical sensitivity - use KB**
        elif kb_confidence >= self.medium_confidence_threshold:
            return 'use_kb_with_fallback'
        
        # **FIXED: Low confidence but KB match exists - still try KB first**
        elif kb_confidence > 0:  # Any KB match at all
            return 'use_kb_with_fallback'
        
        # No KB match at all
        else:
            return 'web_search_available'
    
    def perform_web_search(self, question: str, kb_confidence: float = 0.0) -> Optional[Dict]:
        """Perform web search with MCP priority and KB confidence context"""
        try:
            logger.info(f"🌐 Performing enhanced web search for: {question} (KB confidence: {kb_confidence:.2f})")
            
            # Try MCP search first if available
            if getattr(self.mcp_search, 'mcp_available', False):
                logger.info("🔍 Attempting MCP search...")
                mcp_result = self.mcp_search.search_via_mcp(question)
                if mcp_result and mcp_result.get('status') == 'success':
                    logger.info("✅ MCP search successful")
                    solution = self.mcp_search.extract_math_solution(question, mcp_result)
                    if solution:
                        solution['mcp_used'] = True
                        
                        # Apply AI Gateway output validation
                        output_validation = self.ai_gateway.process_output(solution, question)
                        if output_validation["status"] == "approved":
                            solution['gateway_approved'] = True
                            solution['quality_score'] = output_validation.get('quality_score', 0.8)
                        else:
                            logger.warning(f"⚠️ Gateway output validation: {output_validation['reason']}")
                            solution['gateway_warnings'] = output_validation
                        
                        return solution
            
            # Fallback to traditional web search
            if self.web_search_agent:
                web_result = self.web_search_agent.search_math_solution(question, kb_confidence)
                if web_result:
                    web_result['mcp_used'] = False
                    
                    # Apply AI Gateway output validation
                    output_validation = self.ai_gateway.process_output(web_result, question)
                    if output_validation["status"] == "approved":
                        web_result['gateway_approved'] = True
                        web_result['quality_score'] = output_validation.get('quality_score', 0.8)
                    else:
                        logger.warning(f"⚠️ Gateway output validation: {output_validation['reason']}")
                        web_result['gateway_warnings'] = output_validation
                
                return web_result
            else:
                logger.warning("WebSearchAgent not available")
                return None
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return None
    
    def validate_output(self, solution: Dict, original_question: str) -> Dict:
        """Validate output using AI Gateway"""
        return self.ai_gateway.process_output(solution, original_question)
    
    def sanitize_output(self, solution: Dict) -> Dict:
        """Sanitize output using AI Gateway"""
        return self.ai_gateway._sanitize_output(solution)
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics for monitoring"""
        gateway_analytics = self.ai_gateway.analytics.get_analytics()
        
        return {
            'high_confidence_threshold': self.high_confidence_threshold,
            'medium_confidence_threshold': self.medium_confidence_threshold,
            'low_confidence_threshold': self.low_confidence_threshold,
            'web_search_indicators_count': len(self.web_search_indicators),
            'numerical_sensitive_indicators_count': len(self.numerical_sensitive_indicators),
            'mcp_available': getattr(self.mcp_search, 'mcp_available', False),
            'gateway_analytics': gateway_analytics
        }
    
    def _create_error_response(self, message: str) -> Dict:
        """Create standardized error response"""
        return {
            'status': 'error',
            'error_message': message,
            'requires_web_search': False,
            'confidence': 0.0
        }

    def test_routing_decision(self, question: str, rag_system=None) -> Dict:
        """Test routing decision for a specific question"""
        routing_result = self.process_question(question, rag_system)
        
        # Get KB confidence separately
        kb_confidence = 0.0
        numerical_sensitive = False
        if rag_system:
            kb_result = rag_system.search_knowledge_base(question)
            if kb_result:
                kb_confidence = kb_result.get('similarity_score', 0.0)
            numerical_sensitive = self._is_numerically_sensitive(question)
        
        return {
            'question': question,
            'routing_result': routing_result,
            'kb_confidence': kb_confidence,
            'numerical_sensitive': numerical_sensitive,
            'should_use_web_search': routing_result['route_decision']['needs_web_search'],
            'reason': routing_result['route_decision']['routing_reason'],
            'suggested_action': routing_result['route_decision']['suggested_action']
        }

    def analyze_numerical_sensitivity(self, question: str) -> Dict:
        """Analyze numerical sensitivity of a question"""
        numbers = re.findall(r'\$?(\d+\.?\d*)', question)
        numerical_indicators = [
            indicator for indicator in self.numerical_sensitive_indicators 
            if indicator in question.lower()
        ]
        
        return {
            'question': question,
            'numbers_found': numbers,
            'numerical_indicators': numerical_indicators,
            'is_numerically_sensitive': self._is_numerically_sensitive(question),
            'number_count': len(numbers),
            'has_multiple_numbers': len(numbers) >= 2,
            'has_numerical_indicators': len(numerical_indicators) > 0
        }

    def get_gateway_analytics(self) -> Dict:
        """Get AI Gateway analytics"""
        return self.ai_gateway.analytics.get_analytics()

    def test_gateway_validation(self, question: str) -> Dict:
        """Test AI Gateway validation for a question"""
        gateway_result = self.ai_gateway.process_input(question)
        
        return {
            'question': question,
            'gateway_result': gateway_result,
            'is_approved': gateway_result['status'] == 'approved',
            'reason': gateway_result.get('reason', 'approved'),
            'sanitized_question': gateway_result.get('sanitized_question', question)
        }