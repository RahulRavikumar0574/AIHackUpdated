import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage.jsx';
import CreditForm from './pages/CreditForm.jsx';


const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/predict" element={<CreditForm />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
