import React, { useState } from 'react';
import axios from 'axios';
import './MathSolver.css';

const MathSolver = () => {
  const [question, setQuestion] = useState('');
  const [solution, setSolution] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState({ rating: 0, comment: '' });
  const [submittingFeedback, setSubmittingFeedback] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [showCorrectionForm, setShowCorrectionForm] = useState(false);
  const [correctionText, setCorrectionText] = useState('');
  const [correctedAnswer, setCorrectedAnswer] = useState('');

  const API_BASE = 'http://localhost:8000';

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setSolution(null);
    setFeedback({ rating: 0, comment: '' });
    setCurrentQuestion(question.trim());
    setShowCorrectionForm(false);
    setCorrectionText('');
    setCorrectedAnswer('');

    try {
      console.log('Sending request to:', `${API_BASE}/solve`);
      const response = await axios.post(`${API_BASE}/solve`, {
        question: question.trim()
      }, {
        timeout: 10000,
        headers: {
          'Content-Type': 'application/json',
        }
      });
      console.log('Response received:', response.data);
      setSolution(response.data);
    } catch (err) {
      console.error('Request failed:', err);
      let errorMessage = 'Failed to solve the problem';
      
      if (err.code === 'ECONNREFUSED') {
        errorMessage = 'Cannot connect to the server. Make sure the backend is running on port 8000.';
      } else if (err.response) {
        errorMessage = err.response.data?.detail || `Server error: ${err.response.status}`;
      } else if (err.request) {
        errorMessage = 'No response from server. Check your network connection.';
      } else {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedbackSubmit = async () => {
    if (!solution || feedback.rating === 0) {
      alert('Please select a rating before submitting feedback.');
      return;
    }

    setSubmittingFeedback(true);

    try {
      const feedbackData = {
        question: currentQuestion,
        original_solution: solution,
        rating: feedback.rating,
        feedback: feedback.comment || `Rated ${feedback.rating} stars`
      };

      console.log('Submitting learning feedback:', feedbackData);
      
      const response = await axios.post(`${API_BASE}/feedback/simple`, feedbackData, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      console.log('Feedback response:', response.data);
      
      if (response.data.learning_triggered) {
        alert('🎓 Thank you! We\'ve learned from your feedback and added this to our knowledge base!');
      } else {
        alert('✅ Thank you for your feedback!');
      }
      
      setFeedback({ rating: 0, comment: '' });
      
    } catch (err) {
      console.error('Failed to submit feedback:', err);
      let errorMsg = 'Failed to submit feedback';
      
      if (err.response) {
        errorMsg = err.response.data?.detail || `Server error: ${err.response.status}`;
      }
      
      alert(`❌ ${errorMsg}`);
    } finally {
      setSubmittingFeedback(false);
    }
  };

  const handleRatingClick = async (rating) => {
    setFeedback(prev => ({ ...prev, rating }));
    
    if (rating >= 4 && solution && currentQuestion) {
      const confirmLearn = window.confirm(
        `Rate this solution ${rating} stars? This will help us learn and improve!`
      );
      
      if (confirmLearn) {
        await handleFeedbackSubmit();
      }
    }
  };

  const handleCorrectionSubmit = async () => {
    if (!correctionText.trim()) {
      alert('Please provide the correct solution explanation.');
      return;
    }

    try {
      const correctionData = {
        question: currentQuestion,
        wrong_solution: solution,
        correct_solution: {
          steps: [
            {
              step_number: 1,
              explanation: correctionText,
              equation: correctedAnswer ? `Answer: ${correctedAnswer}` : ""
            }
          ],
          final_answer: correctedAnswer || "See explanation above"
        },
        user_notes: "User provided correction via frontend"
      };

      console.log('Submitting correction:', correctionData);
      
      const response = await axios.post(`${API_BASE}/feedback/correction`, correctionData);
      
      if (response.data.correction_applied) {
        alert('🎯 Thank you! The correction has been applied to our knowledge base.');
        setShowCorrectionForm(false);
        setCorrectionText('');
        setCorrectedAnswer('');
      } else {
        alert('✅ Thank you for the correction! It has been recorded.');
      }
    } catch (err) {
      console.error('Correction submission failed:', err);
      alert('❌ Failed to submit correction. Please try again.');
    }
  };

  const testConnection = async () => {
    try {
      const response = await axios.get(`${API_BASE}/health`);
      alert(`✅ Backend is healthy: ${response.data.status}`);
    } catch (err) {
      alert('❌ Backend connection failed. Make sure the server is running on port 8000.');
    }
  };

  const getSourceBadge = (source, confidence) => {
    const badges = {
      'knowledge_base': { 
        emoji: '📚', 
        text: 'Knowledge Base', 
        description: 'From our verified math knowledge base',
        color: 'green'
      },
      'web_search_llm': { 
        emoji: '🌐', 
        text: 'Web Search + AI', 
        description: 'Enhanced with web search and AI reasoning',
        color: 'blue'
      },
      'web_search': { 
        emoji: '🌐', 
        text: 'Web Search', 
        description: 'Information sourced from the web',
        color: 'blue'
      },
      'mcp_search': { 
        emoji: '🔍', 
        text: 'MCP Search', 
        description: 'Using Model Context Protocol search',
        color: 'purple'
      },
      'equation_solver': { 
        emoji: '🧮', 
        text: 'Equation Solver', 
        description: 'Solved using mathematical equation solver',
        color: 'orange'
      },
      'speed_time_solver': { 
        emoji: '⚡', 
        text: 'Speed Solver', 
        description: 'Solved using speed-time calculator',
        color: 'red'
      },
      'basic_solver': { 
        emoji: '🤖', 
        text: 'AI Generated', 
        description: 'Generated using AI reasoning',
        color: 'gray'
      },
      'llm_enhanced': { 
        emoji: '🌐', 
        text: 'Web Search + AI', 
        description: 'Enhanced with web search and AI reasoning',
        color: 'blue'
      },
      'enhanced_fallback': { 
        emoji: '🤖', 
        text: 'AI Generated', 
        description: 'Generated using AI reasoning',
        color: 'gray'
      },
      'web_search_fallback': { 
        emoji: '🌐', 
        text: 'Web Search', 
        description: 'Information sourced from the web',
        color: 'blue'
      },
      'feedback_learning': { 
        emoji: '🎓', 
        text: 'Learned Solution', 
        description: 'Learned from user feedback',
        color: 'green'
      }
    };

    const badge = badges[source] || { 
      emoji: '🔍', 
      text: source, 
      description: 'AI generated solution',
      color: 'gray'
    };

    return (
      <div className={`source-badge source-${badge.color}`}>
        <span className="source-emoji">{badge.emoji}</span>
        <span className="source-text">
          {badge.text}
          {confidence > 0 && (
            <span className="confidence"> • {Math.round(confidence * 100)}% confidence</span>
          )}
        </span>
        <div className="source-tooltip">
          {badge.description}
          {solution?.source_url && (
            <div className="source-link">
              <a href={solution.source_url} target="_blank" rel="noopener noreferrer">
                📎 View Source
              </a>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="math-solver">
      <div className="solver-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
          <h2>Ask a Math Question</h2>
          <button 
            onClick={testConnection}
            className="test-connection-btn"
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Test Connection
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="question-form">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your math question here... (e.g., Solve 2x + 5 = 15, Explain probability, etc.)"
            rows="3"
            disabled={loading}
            style={{
              width: '100%',
              padding: '0.75rem',
              border: '1px solid #ddd',
              borderRadius: '6px',
              fontSize: '1rem',
              resize: 'vertical',
              fontFamily: 'inherit'
            }}
          />
          <button 
            type="submit" 
            disabled={loading || !question.trim()}
            style={{
              width: '100%',
              padding: '0.75rem',
              backgroundColor: loading ? '#6c757d' : '#007bff',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              fontSize: '1rem',
              cursor: loading ? 'not-allowed' : 'pointer',
              marginTop: '0.5rem'
            }}
          >
            {loading ? '🧠 Solving...' : 'Solve Problem'}
          </button>
        </form>

        {error && (
          <div className="error-message" style={{
            padding: '1rem',
            backgroundColor: '#f8d7da',
            color: '#721c24',
            border: '1px solid #f5c6cb',
            borderRadius: '6px',
            marginTop: '1rem'
          }}>
            ❌ {error}
            <div style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>
              <strong>Troubleshooting:</strong>
              <ul style={{ textAlign: 'left', margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                <li>Make sure the backend is running: <code>cd backend &amp;&amp; python run.py</code></li>
                <li>Check that no other application is using port 8000</li>
                <li>Verify the backend started without errors</li>
              </ul>
            </div>
          </div>
        )}

        {solution && (
          <div className="solution-container" style={{
            marginTop: '2rem',
            padding: '1.5rem',
            border: '1px solid #e9ecef',
            borderRadius: '8px',
            backgroundColor: '#f8f9fa'
          }}>
            <div className="solution-header" style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              marginBottom: '1rem',
              flexWrap: 'wrap',
              gap: '1rem'
            }}>
              <h3 style={{ margin: 0, color: '#333' }}>Solution</h3>
              {getSourceBadge(solution.display_source || solution.source, solution.confidence)}
            </div>

            <div className="question-display" style={{
              padding: '1rem',
              backgroundColor: 'white',
              borderRadius: '6px',
              marginBottom: '1rem',
              borderLeft: '4px solid #007bff'
            }}>
              <strong>Question:</strong> {solution.question}
            </div>

            <div className="steps-container" style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ color: '#333', marginBottom: '1rem' }}>Step-by-Step Solution:</h4>
              {solution.steps && solution.steps.map((step) => (
                <div key={step.step_number} className="step" style={{
                  display: 'flex',
                  marginBottom: '1rem',
                  padding: '1rem',
                  backgroundColor: 'white',
                  borderRadius: '6px',
                  border: '1px solid #e9ecef'
                }}>
                  <div className="step-number" style={{
                    width: '2rem',
                    height: '2rem',
                    backgroundColor: '#007bff',
                    color: 'white',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontWeight: 'bold',
                    marginRight: '1rem',
                    flexShrink: 0
                  }}>
                    {step.step_number}
                  </div>
                  <div className="step-content" style={{ flex: 1 }}>
                    <div className="explanation" style={{ 
                      marginBottom: step.equation ? '0.5rem' : '0',
                      lineHeight: '1.5'
                    }}>
                      {step.explanation}
                    </div>
                    {step.equation && (
                      <div className="equation" style={{
                        padding: '0.5rem',
                        backgroundColor: '#f8f9fa',
                        borderRadius: '4px',
                        fontFamily: 'monospace',
                        border: '1px solid #e9ecef'
                      }}>
                        {step.equation}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="final-answer" style={{
              padding: '1rem',
              backgroundColor: '#d4edda',
              color: '#155724',
              borderRadius: '6px',
              border: '1px solid #c3e6cb',
              fontWeight: 'bold',
              fontSize: '1.1rem'
            }}>
              <strong>Final Answer:</strong> {solution.final_answer}
            </div>

            {/* Enhanced Feedback & Correction Section */}
            <div className="feedback-section" style={{ marginTop: '2rem' }}>
              <h4 style={{ color: '#333', marginBottom: '1rem' }}>Help us improve:</h4>
              
              {/* Quick Rating */}
              <div className="rating-section" style={{ marginBottom: '1rem' }}>
                <p style={{ marginBottom: '0.5rem', fontWeight: 'bold' }}>Rate this solution:</p>
                <div className="rating-buttons" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      onClick={() => handleRatingClick(star)}
                      className={`rating-btn ${feedback.rating === star ? 'active' : ''}`}
                      disabled={submittingFeedback}
                      style={{
                        padding: '0.5rem 1rem',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        backgroundColor: feedback.rating === star ? '#e3f2fd' : 'white',
                        cursor: submittingFeedback ? 'not-allowed' : 'pointer',
                        opacity: submittingFeedback ? 0.6 : 1
                      }}
                    >
                      ⭐ {star}
                    </button>
                  ))}
                </div>
              </div>

              {/* Action Buttons */}
              <div className="feedback-actions" style={{ 
                marginBottom: '1rem', 
                display: 'flex', 
                gap: '0.5rem', 
                flexWrap: 'wrap' 
              }}>
                <button 
                  onClick={() => setShowCorrectionForm(!showCorrectionForm)}
                  className="correction-btn"
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#ff6b6b',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  🚩 Report Wrong Answer
                </button>
                
                <button 
                  onClick={() => {
                    setFeedback({ rating: 5, comment: 'Great solution!' });
                    handleFeedbackSubmit();
                  }}
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.9rem'
                  }}
                >
                  👍 Helpful Solution
                </button>
              </div>

              {/* Correction Form */}
              {showCorrectionForm && (
                <div className="correction-form" style={{
                  marginTop: '1rem',
                  padding: '1rem',
                  border: '1px solid #ff6b6b',
                  borderRadius: '6px',
                  backgroundColor: '#fff9f9'
                }}>
                  <h4 style={{ margin: '0 0 1rem 0', color: '#d63031' }}>Submit Correction</h4>
                  <p style={{ fontSize: '0.9rem', marginBottom: '1rem', color: '#666' }}>
                    Found an error? Help us improve by providing the correct solution.
                  </p>
                  
                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                      Correct Answer (optional):
                    </label>
                    <input
                      type="text"
                      placeholder="Enter the correct final answer..."
                      value={correctedAnswer}
                      onChange={(e) => setCorrectedAnswer(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.5rem',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        fontSize: '1rem'
                      }}
                    />
                  </div>
                  
                  <div style={{ marginBottom: '1rem' }}>
                    <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 'bold' }}>
                      Step-by-Step Correction: *
                    </label>
                    <textarea
                      placeholder="Please provide the correct step-by-step solution..."
                      rows="4"
                      value={correctionText}
                      onChange={(e) => setCorrectionText(e.target.value)}
                      style={{
                        width: '100%',
                        padding: '0.5rem',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        resize: 'vertical',
                        fontSize: '1rem',
                        fontFamily: 'inherit'
                      }}
                    />
                  </div>
                  
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <button 
                      onClick={handleCorrectionSubmit}
                      disabled={!correctionText.trim()}
                      style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: '#4CAF50',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: correctionText.trim() ? 'pointer' : 'not-allowed',
                        opacity: correctionText.trim() ? 1 : 0.6
                      }}
                    >
                      Submit Correction
                    </button>
                    <button 
                      onClick={() => {
                        setShowCorrectionForm(false);
                        setCorrectionText('');
                        setCorrectedAnswer('');
                      }}
                      style={{
                        padding: '0.5rem 1rem',
                        backgroundColor: '#6c757d',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Detailed Feedback */}
              <div className="detailed-feedback" style={{ marginTop: '1rem' }}>
                <textarea
                  value={feedback.comment}
                  onChange={(e) => setFeedback({ ...feedback, comment: e.target.value })}
                  placeholder="Detailed feedback (optional)... What was good? What can be improved?"
                  rows="3"
                  disabled={submittingFeedback}
                  style={{
                    width: '100%',
                    padding: '0.75rem',
                    border: '1px solid #ddd',
                    borderRadius: '6px',
                    fontSize: '1rem',
                    resize: 'vertical',
                    fontFamily: 'inherit'
                  }}
                />
                
                <div className="feedback-submit" style={{ 
                  marginTop: '0.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  flexWrap: 'wrap'
                }}>
                  <button
                    onClick={handleFeedbackSubmit}
                    disabled={feedback.rating === 0 || submittingFeedback}
                    className="submit-feedback-btn"
                    style={{
                      padding: '0.75rem 1.5rem',
                      backgroundColor: '#2196F3',
                      color: 'white',
                      border: 'none',
                      borderRadius: '6px',
                      cursor: (feedback.rating === 0 || submittingFeedback) ? 'not-allowed' : 'pointer',
                      opacity: (feedback.rating === 0 || submittingFeedback) ? 0.6 : 1,
                      fontSize: '1rem'
                    }}
                  >
                    {submittingFeedback ? 'Submitting...' : 'Submit Detailed Feedback'}
                  </button>
                  
                  {feedback.rating > 0 && (
                    <span className="rating-preview" style={{ color: '#666', fontSize: '0.9rem' }}>
                      You rated: {feedback.rating} ⭐
                      {feedback.rating >= 4 && " (Will be added to knowledge base!)"}
                    </span>
                  )}
                </div>
              </div>

              {/* Learning Info */}
              <div style={{ 
                marginTop: '1rem', 
                padding: '1rem', 
                background: '#e8f5e8', 
                borderRadius: '6px',
                border: '1px solid #4CAF50',
                fontSize: '0.9rem'
              }}>
                <strong>💡 Learning System Active:</strong>
                <ul style={{ margin: '0.5rem 0', paddingLeft: '1.5rem' }}>
                  <li>⭐ 4-5 stars: Solution added to knowledge base</li>
                  <li>🚩 Report Wrong: Corrects errors in the system</li>
                  <li>📊 All feedback: Helps optimize AI routing</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="sample-questions" style={{
        marginTop: '2rem',
        padding: '1.5rem',
        backgroundColor: '#f8f9fa',
        borderRadius: '8px',
        border: '1px solid #e9ecef'
      }}>
        <h3 style={{ marginBottom: '1rem', color: '#333' }}>Try these sample questions:</h3>
        <div className="sample-buttons" style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
          <button 
            onClick={() => setQuestion("Solve the equation: 2x + 5 = 15")}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'white',
              border: '1px solid #007bff',
              color: '#007bff',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Linear Equation
          </button>
          <button 
            onClick={() => setQuestion("Find the roots of the quadratic equation: x² - 5x + 6 = 0")}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'white',
              border: '1px solid #007bff',
              color: '#007bff',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Quadratic Equation
          </button>
          <button 
            onClick={() => setQuestion("Tickets numbered 1 to 20 are mixed. What is probability of multiple of 3 or 5?")}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'white',
              border: '1px solid #007bff',
              color: '#007bff',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Probability Problem
          </button>
          <button 
            onClick={() => setQuestion("Thomas takes 7 days, Raj takes 9 days. How long together?")}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'white',
              border: '1px solid #007bff',
              color: '#007bff',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Work Rate Problem
          </button>
          <button 
            onClick={() => setQuestion("Explain the concept of machine learning mathematics")}
            style={{
              padding: '0.5rem 1rem',
              backgroundColor: 'white',
              border: '1px solid #007bff',
              color: '#007bff',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.9rem'
            }}
          >
            Advanced Concept
          </button>
        </div>
      </div>
    </div>
  );
};

export default MathSolver;