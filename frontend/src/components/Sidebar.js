import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaHome, FaFileAlt, FaClipboardList, FaNewspaper, FaSearch, FaBars, FaDatabase, FaListUl } from 'react-icons/fa';

const Sidebar = () => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <div className={`sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <button className="collapse-btn" onClick={() => setIsCollapsed(!isCollapsed)}>
        <FaBars />
      </button>
      <nav className="sidebar-menu">
        
        <Link to="/" className={`menu-item mt-5 ${isActive('/') ? 'active' : ''}`} title="Home">
          <FaHome />
          <span>Home</span>
        </Link>
        
        <Link to="/observations" className={`menu-item ${isActive('/observations') ? 'active' : ''}`} title="Observations">
          <FaListUl />
          <span>Observations</span>
        </Link>
        
        <Link to="/news" className={`menu-item ${isActive('/news') ? 'active' : ''}`} title="News Analysis">
          <FaNewspaper />
          <span>News</span>
        </Link>

        <Link to="/whats-missing" className={`menu-item ${isActive('/whats-missing') ? 'active' : ''}`} title="What's Missing?">
          <FaSearch />
          <span>What&apos;s Missing?</span>
        </Link>

        <Link to="/db-check" className={`menu-item ${isActive('/db-check') ? 'active' : ''}`} title="Database Check">
          <FaDatabase />
          <span>Database Check</span>
        </Link>

        <Link to="/office-note" className={`menu-item ${isActive('/office-note') ? 'active' : ''}`} title="Office Note">
          <FaClipboardList />
          <span>Office Note</span>
        </Link>

        <hr className="divider mb-5" />

      </nav>
    </div>
  );
};

export default Sidebar;
