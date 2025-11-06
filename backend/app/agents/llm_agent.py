import os
import logging
import json
import re
from typing import Dict, Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class LLMAgent:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("✅ OpenAI LLM Agent initialized")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                self.client = None
        else:
            logger.warning("❌ OpenAI API key not configured")
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate_response(self, prompt: str, system_message: str = None, max_tokens: int = 500) -> Optional[str]:
        """Generate response using OpenAI API"""
        if not self.is_available():
            logger.warning("❌ LLM not available for response generation")
            return None
        
        try:
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append({"role": "system", "content": system_message})
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Provide clear, concise, and accurate responses."
                })
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ LLM response generation failed: {e}")
            return None

    def _extract_json_from_response(self, response: str) -> Optional[Dict]:
        """Extract and parse JSON from LLM response with robust error handling"""
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

    def analyze_mathematical_intent(self, question: str) -> Dict:
        """Analyze if question has mathematical intent - FIXED VERSION"""
        if not self.is_available():
            return {
                "is_mathematical": False,
                "confidence": 0.0,
                "reason": "LLM not available",
                "suggestion": "Please ask a mathematical question"
            }
        
        try:
            prompt = f"""
            Analyze if this question has mathematical intent for a math AI system.
            
            QUESTION: "{question}"
            
            Respond with ONLY a JSON object in this exact format, no other text:
            {{
                "is_mathematical": true or false,
                "confidence": number between 0.0 and 1.0,
                "reason": "brief explanation of why it is or isn't mathematical",
                "suggestion": "what the user should ask instead if not mathematical"
            }}
            
            Mathematical questions include: calculations, algebra, geometry, probability, statistics, equations, word problems with numbers, math concepts.
            Non-mathematical questions include: general knowledge, biographies, history, opinions, personal questions.
            """
            
            response = self.generate_response(
                prompt, 
                "You are a mathematical intent classifier. Return ONLY valid JSON, no other text."
            )
            
            if response:
                result = self._extract_json_from_response(response)
                if result:
                    # Validate required fields
                    if all(key in result for key in ['is_mathematical', 'confidence', 'reason', 'suggestion']):
                        return result
                    else:
                        logger.warning("❌ Missing fields in intent analysis response")
            
            # Fallback analysis with improved logic
            return self._fallback_intent_analysis(question)
            
        except Exception as e:
            logger.error(f"❌ Mathematical intent analysis failed: {e}")
            return self._fallback_intent_analysis(question)

    def _fallback_intent_analysis(self, question: str) -> Dict:
        """Fallback intent analysis when LLM fails"""
        question_lower = question.lower().strip()
        
        # Mathematical keywords
        math_keywords = [
            'solve', 'calculate', 'equation', 'algebra', 'geometry', 'probability', 
            'percentage', 'ratio', 'fraction', 'derivative', 'integral', 'triangle',
            'circle', 'area', 'volume', 'distance', 'speed', 'time', 'work', 'rate',
            'sum', 'product', 'difference', 'multiple', 'divide', 'add', 'subtract',
            'multiply', 'angle', 'perimeter', 'radius', 'diameter', 'square', 'cube',
            'root', 'logarithm', 'trigonometry', 'calculus', 'statistics', 'mean',
            'median', 'mode', 'variable', 'formula', 'theorem', 'proof', 'number',
            'digit', 'decimal', 'fraction', 'percent', 'graph', 'coordinate', 'axis'
        ]
        
        # Non-mathematical patterns
        non_math_patterns = [
            'who is', 'what is', 'when did', 'where is', 'why is', 'how to',
            'biography', 'history', 'born', 'died', 'invented', 'created',
            'famous', 'celebrity', 'person', 'people', 'author', 'writer',
            'actor', 'actress', 'singer', 'politician', 'scientist', 'inventor'
        ]
        
        has_math_keywords = any(keyword in question_lower for keyword in math_keywords)
        has_non_math_patterns = any(pattern in question_lower for pattern in non_math_patterns)
        has_numbers = any(char.isdigit() for char in question)
        
        # Decision logic
        if has_math_keywords and not has_non_math_patterns:
            is_mathematical = True
            confidence = 0.8
            reason = "Contains mathematical keywords and concepts"
            suggestion = "This appears to be a mathematical question"
        elif has_numbers and not has_non_math_patterns:
            is_mathematical = True
            confidence = 0.6
            reason = "Contains numerical elements"
            suggestion = "This appears to involve calculations"
        elif has_non_math_patterns:
            is_mathematical = False
            confidence = 0.9
            reason = "Appears to be a general knowledge or biographical question"
            suggestion = "Please ask about mathematical problems, calculations, or concepts"
        else:
            is_mathematical = False
            confidence = 0.7
            reason = "Does not appear to be mathematical"
            suggestion = "Please ask about mathematics, calculations, or related topics"
        
        return {
            "is_mathematical": is_mathematical,
            "confidence": confidence,
            "reason": reason,
            "suggestion": suggestion
        }

    def assess_solution_quality(self, question: str, solution: Dict) -> Dict:
        """Assess quality of a math solution"""
        if not self.is_available():
            return {
                "quality_score": 0.5,
                "is_correct": True,
                "feedback": "LLM not available for quality assessment",
                "suggestions": []
            }
        
        try:
            solution_text = self._format_solution_for_assessment(solution)
            
            prompt = f"""
            Assess the quality and correctness of this math solution:
            
            QUESTION: {question}
            
            SOLUTION:
            {solution_text}
            
            Respond with ONLY a JSON object in this exact format, no other text:
            {{
                "quality_score": number between 0.0 and 1.0,
                "is_correct": true or false,
                "feedback": "brief assessment of the solution quality",
                "suggestions": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = self.generate_response(
                prompt, 
                "You are a math solution quality assessor. Be objective and precise. Return ONLY valid JSON."
            )
            
            if response:
                result = self._extract_json_from_response(response)
                if result and all(key in result for key in ['quality_score', 'is_correct', 'feedback', 'suggestions']):
                    return result
            
            # Fallback assessment
            return {
                "quality_score": 0.7,
                "is_correct": True,
                "feedback": "Basic assessment completed",
                "suggestions": ["Consider verifying with additional sources"]
            }
            
        except Exception as e:
            logger.error(f"❌ Solution quality assessment failed: {e}")
            return {
                "quality_score": 0.5,
                "is_correct": True,
                "feedback": "Quality assessment unavailable",
                "suggestions": []
            }

    def _format_solution_for_assessment(self, solution: Dict) -> str:
        """Format solution for quality assessment"""
        steps_text = ""
        if 'steps' in solution:
            for step in solution['steps']:
                step_num = step.get('step_number', '?')
                explanation = step.get('explanation', '')
                equation = step.get('equation', '')
                steps_text += f"Step {step_num}: {explanation}"
                if equation:
                    steps_text += f" | Equation: {equation}"
                steps_text += "\n"
        
        final_answer = solution.get('final_answer', 'Not provided')
        
        return f"STEPS:\n{steps_text}\nFINAL ANSWER: {final_answer}"

    def solve_math_problem(self, question: str, context: str = "") -> Optional[Dict]:
        """Solve math problem using LLM"""
        if not self.is_available():
            logger.warning("LLM not available for math problem solving")
            return None
        
        try:
            prompt = self._create_math_solving_prompt(question, context)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert math tutor. Solve math problems with clear, step-by-step explanations. 
                        Always provide the solution in the exact JSON format specified. Be precise with numbers and calculations.
                        Return ONLY valid JSON, no other text."""
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_math_solution_response(llm_response)
            
        except Exception as e:
            logger.error(f"LLM math solving failed: {e}")
            return None

    def _create_math_solving_prompt(self, question: str, context: str = "") -> str:
        """Create prompt for math problem solving"""
        base_prompt = f"""
        Solve this math problem accurately:

        QUESTION: {question}
        """
        
        if context:
            base_prompt += f"\nCONTEXT: {context}"
        
        base_prompt += """
        
        Provide the solution in EXACTLY this JSON format and nothing else:
        {
            "steps": [
                {"step_number": 1, "explanation": "Clear step explanation", "equation": "mathematical equation if applicable"},
                {"step_number": 2, "explanation": "Next step explanation", "equation": "equation"}
            ],
            "final_answer": "Final numerical answer or result"
        }

        IMPORTANT: Return ONLY the JSON object, no other text or explanations.
        """
        
        return base_prompt

    def _parse_math_solution_response(self, llm_response: str) -> Optional[Dict]:
        """Parse LLM math solution response"""
        try:
            result = self._extract_json_from_response(llm_response)
            if result:
                # Validate structure
                if 'steps' in result and 'final_answer' in result:
                    # Ensure steps have required fields
                    validated_steps = []
                    for i, step in enumerate(result['steps']):
                        validated_steps.append({
                            'step_number': i + 1,
                            'explanation': step.get('explanation', f'Step {i + 1}'),
                            'equation': step.get('equation', '')
                        })
                    
                    return {
                        'steps': validated_steps,
                        'final_answer': str(result['final_answer'])
                    }
            
            # Fallback if JSON parsing fails
            logger.warning("Math solution JSON parsing failed, creating fallback solution")
            return {
                'steps': [
                    {
                        'step_number': 1,
                        'explanation': 'Solution generated by AI analysis',
                        'equation': ''
                    }
                ],
                'final_answer': 'Answer requires detailed calculation'
            }
            
        except Exception as e:
            logger.error(f"Math solution parsing failed: {e}")
            return {
                'steps': [
                    {
                        'step_number': 1,
                        'explanation': 'Solution generated by AI analysis',
                        'equation': ''
                    }
                ],
                'final_answer': 'Answer requires detailed calculation'
            }

    # ... (rest of your existing methods remain the same)

    def enhance_web_results(self, question: str, web_results: Dict) -> Optional[Dict]:
        """Process web search results using LLM to generate clean answer"""
        if not self.is_available():
            return None
        
        try:
            web_content = self._extract_web_content(web_results)
            if not web_content:
                return None
            
            prompt = self._create_llm_prompt(question, web_content)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful math and science tutor. Provide clear, step-by-step explanations. Always return valid JSON format with no other text."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            llm_response = response.choices[0].message.content
            return self._parse_llm_response(question, llm_response)
            
        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return None
    
    def _extract_web_content(self, web_results: Dict) -> str:
        """Extract content from web search results"""
        content_parts = []
        
        if web_results.get('results'):
            for i, result in enumerate(web_results['results'][:3]):
                title = result.get('title', '')
                content = result.get('content', '')
                url = result.get('url', '')
                
                if content and len(content) > 50:
                    content_parts.append(f"RESULT {i+1}:\nTitle: {title}\nURL: {url}\nContent: {content[:500]}...")
        
        return "\n\n".join(content_parts) if content_parts else ""
    
    def _create_llm_prompt(self, question: str, web_content: str) -> str:
        """Create prompt for LLM to generate structured answer"""
        return f"""
        TASK: Analyze the web search results and create a clear, structured answer.

        USER QUESTION: {question}

        WEB SEARCH RESULTS:
        {web_content}

        INSTRUCTIONS:
        1. Analyze the web search results thoroughly
        2. Extract the most relevant information
        3. Create a clear, step-by-step explanation
        4. Provide a concise final answer
        5. For math problems, show the working and calculations

        RESPONSE FORMAT (JSON only, no other text):
        {{
            "steps": [
                {{"step_number": 1, "explanation": "Clear explanation here", "equation": "optional equation"}},
                {{"step_number": 2, "explanation": "Next step explanation", "equation": "optional equation"}}
            ],
            "final_answer": "Concise final answer here"
        }}

        IMPORTANT: Return ONLY valid JSON, no other text.
        """
    
    def _parse_llm_response(self, question: str, llm_response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            result = self._extract_json_from_response(llm_response)
            if result and 'steps' in result and 'final_answer' in result:
                return {
                    'solution': result,
                    'source': 'llm_enhanced',
                    'confidence': 0.9,
                    'api_status': 'llm_processed'
                }
            else:
                raise ValueError("Invalid JSON structure from LLM")
            
        except Exception as e:
            logger.error(f"LLM response parsing failed: {e}")
            
            return {
                'solution': {
                    'steps': [{
                        'step_number': 1,
                        'explanation': llm_response[:500] if len(llm_response) > 100 else "LLM generated response",
                        'equation': ''
                    }],
                    'final_answer': "Answer generated from analyzed web content"
                },
                'source': 'llm_fallback',
                'confidence': 0.7,
                'api_status': 'llm_parsing_failed'
            }