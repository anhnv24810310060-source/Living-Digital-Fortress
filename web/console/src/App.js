import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import AttackMap from './components/AttackMap';
import PolicyDesigner from './components/PolicyDesigner';
import PluginManager from './components/PluginManager';
import ShadowEvaluator from './components/ShadowEvaluator';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/attack-map" replace />} />
            <Route path="/attack-map" element={<AttackMap />} />
            <Route path="/policy-designer" element={<PolicyDesigner />} />
            <Route path="/plugin-manager" element={<PluginManager />} />
            <Route path="/shadow-evaluator" element={<ShadowEvaluator />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;