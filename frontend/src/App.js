import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import OfficeNote from './pages/OfficeNote';
import Home from './pages/Home';
import ScheduleVI from './pages/ScheduleVI';
import WhatsMissing from './pages/WhatsMissing';
import News from './pages/News';
import DatabaseCheck from './pages/DBCheck'
import Sidebar from './components/Sidebar';
import Observations from './pages/Observations'
import Footer from './components/Footer';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  return (
    <Router>
      <div className='app-container'>
        <Sidebar />
      <main className='main-content'>
        <Routes>
          <Route path ="/" element={<Home />} />
          <Route path="/office-note" element={<OfficeNote />} />
          <Route path ="/sch-vii" element={<ScheduleVI />} />
          <Route path ="/news" element={<News />} />
          <Route path ="/whats-missing" element={<WhatsMissing />} />
          <Route path ="/db-check" element={<DatabaseCheck />} />
          <Route path ="/observations" element={<Observations />} />
        </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
