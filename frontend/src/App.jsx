import React from 'react';
import MathSolver from './components/MathSolver';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="app-header">
        <h1>🧮 Math Agent - Phase 1</h1>
        <p>Core RAG System for Mathematical Problem Solving</p>
      </header>
      <main>
        <MathSolver />
      </main>
    </div>
  );
}

export default App;