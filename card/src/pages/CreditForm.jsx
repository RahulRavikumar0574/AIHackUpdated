import React, { useState } from 'react';
import axios from 'axios';
import bgImage from '../assets/5.jpg';

const CreditForm = () => {
  const [formData, setFormData] = useState({
    gender: 'Male',
    age: '',
    maritalStatus: 'Married',
    children: '',
    familyMembers: '',
    income: '',
    incomeSource: 'Working',
    education: 'Higher education',
    employmentStatus: 'Employed',
    yearsEmployed: '',
    occupation: 'Unknown',
    ownCar: 'Yes',
    ownRealty: 'Yes',
    housingSituation: 'House / apartment',
    creditHistoryLength: '',
    countLatePayments: '',
    percentageOnTimePayments: '',
    monthsSinceLastDelinquency: '',
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showImages, setShowImages] = useState(false);
  
  // New state for full-screen image view
  const [fullScreenImage, setFullScreenImage] = useState(null);
  const [showFullScreen, setShowFullScreen] = useState(false);

  const occupationOptions = [
    'Unknown', 'Laborers', 'Managers', 'Sales staff', 'Drivers', 'Core staff', 'Accountants',
    'High skill tech staff', 'Medicine staff', 'Security staff', 'Cooking staff',
    'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Waiters/barmen staff',
    'Secretaries', 'Realty agents', 'HR staff', 'IT staff'
  ];

  const handleChange = (e) => {
    const { name, value, type } = e.target;

    if (type === 'radio') {
      setFormData((prevData) => ({
        ...prevData,
        [name]: value,
      }));
    } else {
      setFormData((prevData) => ({
        ...prevData,
        [name]: value,
      }));
    }
  };

  // dynamic start
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setPrediction(null);
    setShowImages(false);

    // Sanitize the payload by converting empty strings to numbers
    const payload = {
      ...formData,
      age: formData.age ? Number(formData.age) : 0,
      children: formData.children ? Number(formData.children) : 0,
      familyMembers: formData.familyMembers ? Number(formData.familyMembers) : 1,
      income: formData.income ? Number(formData.income) : 0,
      yearsEmployed: formData.yearsEmployed ? Number(formData.yearsEmployed) : 0,
      creditHistoryLength: formData.creditHistoryLength ? Number(formData.creditHistoryLength) : 0,
      countLatePayments: formData.countLatePayments ? Number(formData.countLatePayments) : 0,
      percentageOnTimePayments: formData.percentageOnTimePayments ? Number(formData.percentageOnTimePayments) : 100,
      monthsSinceLastDelinquency: formData.monthsSinceLastDelinquency ? Number(formData.monthsSinceLastDelinquency) : 999,
    };
    
    // The `contactMethods` field is not used in the backend, but we'll remove it from the payload
    // if it exists, to prevent it from causing a validation error on the backend.
    delete payload.contactMethods;

    try {
      const response = await axios.post('http://localhost:8000/predict', payload);
      setPrediction(response.data);
      setShowImages(true); 
    } catch (error) {
      console.error('Prediction failed:', error);
      if (error.response) {
        // Log the specific error from the backend for debugging
        console.error('Backend validation error:', error.response.data);
      }
      setPrediction({
        isApproved: false,
        message: 'An error occurred during prediction. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  };

  // Click handlers for full-screen view
  const handleImageClick = (imageUrl) => {
    setFullScreenImage(imageUrl);
    setShowFullScreen(true);
  };

  const handleCloseFullScreen = () => {
    setFullScreenImage(null);
    setShowFullScreen(false);
  };

  return (
    <div
      className="min-h-screen w-full flex items-center justify-center p-4 relative overflow-hidden bg-cover bg-center"
      style={{
        backgroundImage: `url(${bgImage})`
      }}
    >
      {/* Background Overlay */}
      <div className="absolute inset-0 bg-black opacity-20"></div>

      {/* Main Content - Full Tab Form */}
      <div className="relative z-10 w-full p-8 text-white">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-extrabold drop-shadow-md">
            Credit Card Approval Prediction
          </h1>
          <p className="mt-2 text-lg drop-shadow">
            Enter applicant details to predict credit approval.
          </p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-6 max-w-2xl mx-auto">
          
          {/* Section 1: Personal Details */}
          <div className="bg-white bg-opacity-80 p-6 rounded-lg shadow-lg text-gray-800">
            <h2 className="text-2xl font-bold mb-4">Personal Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Gender */}
              <div>
                <label className="block font-semibold mb-2">Gender</label>
                <div className="flex space-x-4">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="gender"
                      value="Male"
                      checked={formData.gender === 'Male'}
                      onChange={handleChange}
                      className="form-radio text-blue-600"
                    />
                    <span className="ml-2">Male</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="gender"
                      value="Female"
                      checked={formData.gender === 'Female'}
                      onChange={handleChange}
                      className="form-radio text-blue-600"
                    />
                    <span className="ml-2">Female</span>
                  </label>
                </div>
              </div>

              {/* Age */}
              <div>
                <label className="block font-semibold mb-2">Age (in years)</label>
                <input
                  type="number"
                  name="age"
                  min={1}
                  value={formData.age}
                  onChange={handleChange}
                  placeholder="e.g., 35"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Marital Status */}
              <div className="col-span-1 md:col-span-2">
                <label className="block font-semibold mb-2">Marital Status</label>
                <select
                  name="maritalStatus"
                  value={formData.maritalStatus}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="Married">Married</option>
                  <option value="Single / not married">Single / not married</option>
                  <option value="Civil marriage">Civil marriage</option>
                  <option value="Separated">Separated</option>
                  <option value="Widow">Widow</option>
                </select>
              </div>
              
              {/* Number of Children & Family Members */}
              <div className="col-span-1 md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block font-semibold mb-2">Number of Children</label>
                  <input
                    type="number"
                    name="children"
                    min={0}
                    value={formData.children}
                    onChange={handleChange}
                    placeholder="e.g., 2"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block font-semibold mb-2">Number of Family Members</label>
                  <input
                    type="number"
                    name="familyMembers"
                    min={1}
                    value={formData.familyMembers}
                    onChange={handleChange}
                    placeholder="e.g., 4"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>
          
          {/* Section 2: Financial & Employment Details */}
          <div className="bg-white bg-opacity-80 p-6 rounded-lg shadow-lg text-gray-800">
            <h2 className="text-2xl font-bold mb-4">Financial & Employment Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Annual Income & Income Source */}
              <div className="col-span-1 md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block font-semibold mb-2">Total Annual Income ($)</label>
                  <input
                    type="number"
                    name="income"
                    value={formData.income}
                    onChange={handleChange}
                    placeholder="e.g., 90000"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block font-semibold mb-2">Income Source</label>
                  <select
                    name="incomeSource"
                    value={formData.incomeSource}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Working">Working</option>
                    <option value="Commercial associate">Commercial associate</option>
                    <option value="Pensioner">Pensioner</option>
                    <option value="State servant">State servant</option>
                    <option value="Student">Student</option>
                  </select>
                </div>
              </div>

              {/* Education Level */}
              <div className="col-span-1 md:col-span-2">
                <label className="block font-semibold mb-2">Education Level</label>
                <select
                  name="education"
                  value={formData.education}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="Higher education">Higher education</option>
                  <option value="Secondary / secondary special">Secondary / secondary special</option>
                  <option value="Incomplete higher">Incomplete higher</option>
                  <option value="Lower secondary">Lower secondary</option>
                  <option value="Academic degree">Academic degree</option>
                </select>
              </div>
              
              {/* Employment Status */}
              <div className="col-span-1 md:col-span-2">
                <label className="block font-semibold mb-2">Employment Status</label>
                <div className="flex space-x-4">
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="employmentStatus"
                      value="Employed"
                      checked={formData.employmentStatus === 'Employed'}
                      onChange={handleChange}
                      className="form-radio text-blue-600"
                    />
                    <span className="ml-2">Employed</span>
                  </label>
                  <label className="inline-flex items-center">
                    <input
                      type="radio"
                      name="employmentStatus"
                      value="Unemployed"
                      checked={formData.employmentStatus === 'Unemployed'}
                      onChange={handleChange}
                      className="form-radio text-blue-600"
                    />
                    <span className="ml-2">Unemployed</span>
                  </label>
                </div>
              </div>

              {/* Years Employed (Conditional) */}
              {formData.employmentStatus === 'Employed' && (
                <div className="col-span-1 md:col-span-2">
                  <label className="block font-semibold mb-2">Years Employed</label>
                  <input
                    type="number"
                    name="yearsEmployed"
                    value={formData.yearsEmployed}
                    onChange={handleChange}
                    placeholder="e.g., 5"
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              )}
              
              {/* Occupation */}
              <div className="col-span-1 md:col-span-2">
                <label className="block font-semibold mb-2">Occupation</label>
                <select
                  name="occupation"
                  value={formData.occupation}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {occupationOptions.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Section 3: Assets & Contact Information */}
          <div className="bg-white bg-opacity-80 p-6 rounded-lg shadow-lg text-gray-800">
            <h2 className="text-2xl font-bold mb-4">Assets & Contact Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Own a Car? & Own Realty? */}
              <div className="col-span-1 md:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block font-semibold mb-2">Do you own a car?</label>
                  <div className="flex space-x-4">
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name="ownCar"
                        value="Yes"
                        checked={formData.ownCar === 'Yes'}
                        onChange={handleChange}
                        className="form-radio text-blue-600"
                      />
                      <span className="ml-2">Yes</span>
                    </label>
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name="ownCar"
                        value="No"
                        checked={formData.ownCar === 'No'}
                        onChange={handleChange}
                        className="form-radio text-blue-600"
                      />
                      <span className="ml-2">No</span>
                    </label>
                  </div>
                </div>
                <div>
                  <label className="block font-semibold mb-2">Do you own real estate?</label>
                  <div className="flex space-x-4">
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name="ownRealty"
                        value="Yes"
                        checked={formData.ownRealty === 'Yes'}
                        onChange={handleChange}
                        className="form-radio text-blue-600"
                      />
                      <span className="ml-2">Yes</span>
                    </label>
                    <label className="inline-flex items-center">
                      <input
                        type="radio"
                        name="ownRealty"
                        value="No"
                        checked={formData.ownRealty === 'No'}
                        onChange={handleChange}
                        className="form-radio text-blue-600"
                      />
                      <span className="ml-2">No</span>
                    </label>
                  </div>
                </div>
              </div>
              
              {/* Current Housing Situation */}
              <div className="col-span-1 md:col-span-2">
                <label className="block font-semibold mb-2">Current Housing Situation</label>
                <select
                  name="housingSituation"
                  value={formData.housingSituation}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="House / apartment">House / apartment</option>
                  <option value="With parents">With parents</option>
                  <option value="Rented apartment">Rented apartment</option>
                  <option value="Municipal apartment">Municipal apartment</option>
                  <option value="Co-op apartment">Co-op apartment</option>
                </select>
              </div>
            </div>
          </div>

          {/* New Section: Credit History Details */}
          <div className="bg-white bg-opacity-80 p-6 rounded-lg shadow-lg text-gray-800">
            <h2 className="text-2xl font-bold mb-4">Credit History Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Credit History Length */}
              <div>
                <label className="block font-semibold mb-2">Credit History Length (in months)</label>
                <input
                  type="number"
                  name="creditHistoryLength"
                  value={formData.creditHistoryLength}
                  onChange={handleChange}
                  placeholder="e.g., 60"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Count of Late Payments */}
              <div>
                <label className="block font-semibold mb-2">Count of Late Payments</label>
                <input
                  type="number"
                  name="countLatePayments"
                  value={formData.countLatePayments}
                  onChange={handleChange}
                  placeholder="e.g., 2"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Percentage of On-Time Payments */}
              <div>
                <label className="block font-semibold mb-2">Percentage of On-Time Payments</label>
                <input
                  type="number"
                  name="percentageOnTimePayments"
                  value={formData.percentageOnTimePayments}
                  onChange={handleChange}
                  placeholder="e.g., 95"
                  min="0"
                  max="100"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              {/* Months Since Last Delinquency */}
              <div>
                <label className="block font-semibold mb-2">Months Since Last Delinquency</label>
                <input
                  type="number"
                  name="monthsSinceLastDelinquency"
                  value={formData.monthsSinceLastDelinquency}
                  onChange={handleChange}
                  placeholder="e.g., 12"
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>
          </div>
          
          {/* Submit Button */}
          <div className="flex justify-center mt-8">
            <button
              type="submit"
              className={`w-full py-3 font-semibold rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-600 
                ${loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white'}`}
              disabled={loading}
            >
              {loading ? 'Predicting...' : 'Predict Approval'}
            </button>
          </div>
        </form>

        {/* Prediction Result Section */}
        {prediction && (
          <div
            className={`mt-8 p-6 rounded-xl w-full transition-all duration-300 max-w-2xl mx-auto ${
              prediction.isApproved
                ? 'bg-green-100 border border-green-400'
                : 'bg-red-100 border border-red-400'
            }`}
          >
            <p
              className={`text-lg font-semibold ${
                prediction.isApproved ? 'text-green-800' : 'text-red-800'
              }`}
            >
              {prediction.message}
            </p>

            {(prediction.risk_probability_default !== undefined || prediction.decision_threshold !== undefined) && (
              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-800">
                {prediction.risk_probability_default !== undefined && (
                  <div className="bg-white bg-opacity-70 p-3 rounded">
                    <div className="font-semibold text-gray-700">Estimated Default Risk</div>
                    <div className="text-gray-900 text-lg">
                      {`${(prediction.risk_probability_default * 100).toFixed(1)}%`}
                    </div>
                  </div>
                )}
                {prediction.decision_threshold !== undefined && (
                  <div className="bg-white bg-opacity-70 p-3 rounded">
                    <div className="font-semibold text-gray-700">Decision Threshold</div>
                    <div className="text-gray-900 text-lg">
                      {`${(prediction.decision_threshold * 100).toFixed(1)}%`}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* New Section for Displaying Images */}
        {showImages && (
          <div className="mt-8 p-6 rounded-lg bg-white bg-opacity-80 text-gray-800 max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 text-center">Model Visualizations</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="flex flex-col items-center">
                <h3 className="text-xl font-semibold mb-2">Model Performance</h3>
                <img
                  src="http://localhost:8000/static/credit_analysis_model.png"
                  alt="Model Performance"
                  className="w-full h-auto rounded-lg shadow-md cursor-pointer"
                  onClick={() => handleImageClick("http://localhost:8000/static/credit_analysis_model.png")}
                />
              </div>
              <div className="flex flex-col items-center">
                <h3 className="text-xl font-semibold mb-2">Demographic Insights</h3>
                <img
                  src="http://localhost:8000/static/credit_analysis_demographics.png"
                  alt="Demographic Insights"
                  className="w-full h-auto rounded-lg shadow-md cursor-pointer"
                  onClick={() => handleImageClick("http://localhost:8000/static/credit_analysis_demographics.png")}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Full-screen Modal Component */}
      {showFullScreen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50 p-4"
            onClick={handleCloseFullScreen}
          >
            <div className="relative max-w-5xl max-h-full">
              <img
                src={fullScreenImage}
                alt="Full Screen View"
                className="max-w-full max-h-full"
                onClick={e => e.stopPropagation()}
              />
              <button
                className="absolute top-4 right-4 text-white text-3xl font-bold bg-gray-800 rounded-full w-10 h-10 flex items-center justify-center opacity-75 hover:opacity-100 transition-opacity"
                onClick={handleCloseFullScreen}
              >
                &times;
              </button>
            </div>
          </div>
      )}
    </div>
  );
};

export default CreditForm;