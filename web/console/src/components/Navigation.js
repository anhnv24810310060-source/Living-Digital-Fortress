import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { path: '/attack-map', label: 'Attack Map', icon: '🗺️' },
    { path: '/policy-designer', label: 'Policy Designer', icon: '⚙️' },
    { path: '/plugin-manager', label: 'Plugin Manager', icon: '🔌' },
    { path: '/shadow-evaluator', label: 'Shadow Evaluator', icon: '🧪' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-header">
        <h1>ShieldX Console</h1>
        <div className="nav-status">
          <span className="status-indicator active"></span>
          <span>System Online</span>
        </div>
      </div>
      
      <ul className="nav-menu">
        {navItems.map((item) => (
          <li key={item.path} className={location.pathname === item.path ? 'active' : ''}>
            <Link to={item.path}>
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </Link>
          </li>
        ))}
      </ul>

      <div className="nav-footer">
        <div className="user-info">
          <span>Admin User</span>
          <span className="tenant-id">Tenant: tenant_001</span>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;