import React from 'react';
import { useNavigate } from 'react-router-dom';
import bgImage from '../assets/5.jpg';

import Orb from '../components/Orb';

const HomePage = () => {
  const navigate = useNavigate();
  return (
    <div
      className="min-h-screen w-full flex flex-col items-center justify-center p-4 relative overflow-hidden bg-cover bg-center"
      style={{
        backgroundImage: `url(${bgImage})`
      }}
    >
      {/* Background Overlay */}
      <div className="absolute inset-0 bg-black opacity-20"></div>

      {/* Animated Orb background */}
      <div className="absolute inset-0 flex items-start justify-center" style={{ zIndex: 0, pointerEvents: 'none' }}>
        <div style={{ width: '80vw', maxWidth: '1000px', height: '500px', marginTop: '5vh' }}>
          <Orb hue={220} hoverIntensity={0.3} rotateOnHover={true} />
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-4xl w-full text-center p-6 flex flex-col items-center flex-grow justify-center">
        <h1 className="text-5xl md:text-6xl font-extrabold text-white mb-4 drop-shadow-md">
          Credit Card Approval Prediction
        </h1>
        <p className="text-xl text-white mb-8 drop-shadow">
          Full financial history and the best customer service you've ever had.
        </p>
        <div className="w-full flex justify-center mb-4">
          <button
            onClick={() => navigate('/predict')}
            className="w-full sm:w-auto py-3 px-6 mt-2 sm:mt-0 font-semibold rounded-lg transition-colors bg-white hover:bg-gray-200 text-black shadow-lg"
          >
            Check your score
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="relative z-10 w-full text-center py-4 px-6 mt-auto">
        <p className="text-sm text-white opacity-75">
          Hackathon Presentation By: Rahul, Tarun, Bharath, Harie, John
        </p>
      </div>
    </div>
  );
};

export default HomePage;