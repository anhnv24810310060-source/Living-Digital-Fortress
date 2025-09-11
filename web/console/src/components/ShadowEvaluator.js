import React, { useState, useEffect } from 'react';
import axios from 'axios';

const ShadowEvaluator = () => {
  const [evaluations, setEvaluations] = useState([]);
  const [newEval, setNewEval] = useState({
    rule_name: '',
    rule_type: 'ip_blacklist',
    rule_config: {},
    sample_size: 1000,
    tenant_id: 'tenant_001'
  });
  const [loading, setLoading] = useState(false);
  const [selectedEval, setSelectedEval] = useState(null);

  const ruleTypes = [
    { value: 'ip_blacklist', label: 'IP Blacklist' },
    { value: 'signature_detection', label: 'Signature Detection' },
    { value: 'anomaly_detection', label: 'Anomaly Detection' },
    { value: 'rate_limiting', label: 'Rate Limiting' }
  ];

  const createEvaluation = async () => {
    if (!newEval.rule_name) {
      alert('Please enter a rule name');
      return;
    }

    setLoading(true);
    try {
      const evalRequest = {
        ...newEval,
        rule_id: `rule_${Date.now()}`,
        rule_config: getRuleConfig()
      };

      const response = await axios.post('/api/shadow/eval', evalRequest);
      
      if (response.data.success) {
        alert('Shadow evaluation started successfully!');
        setNewEval({
          rule_name: '',
          rule_type: 'ip_blacklist',
          rule_config: {},
          sample_size: 1000,
          tenant_id: 'tenant_001'
        });
        fetchEvaluations();
      }
    } catch (error) {
      console.error('Failed to create evaluation:', error);
      alert('Failed to create evaluation');
    }
    setLoading(false);
  };

  const getRuleConfig = () => {
    switch (newEval.rule_type) {
      case 'ip_blacklist':
        return {
          blacklisted_ips: ['185.220.101.182', '103.224.182.251', '91.240.118.172']
        };
      case 'signature_detection':
        return {
          signatures: ['SELECT * FROM', 'UNION SELECT', '<script>']
        };
      case 'anomaly_detection':
        return {
          anomaly_threshold: 0.8
        };
      case 'rate_limiting':
        return {
          max_requests_per_minute: 100
        };
      default:
        return {};
    }
  };

  const fetchEvaluations = async () => {
    try {
      const response = await axios.get('/api/shadow/evaluations');
      setEvaluations(response.data.evaluations || []);
    } catch (error) {
      console.error('Failed to fetch evaluations:', error);
      // Mock data for demo
      setEvaluations([
        {
          eval_id: 'eval_001',
          rule_name: 'IP Blacklist Test',
          rule_type: 'ip_blacklist',
          status: 'completed',
          sample_size: 1000,
          true_positives: 85,
          false_positives: 12,
          true_negatives: 890,
          false_negatives: 13,
          precision: 0.876,
          recall: 0.867,
          f1_score: 0.871,
          estimated_fp_rate: 0.012,
          estimated_tp_rate: 0.085,
          recommendations: ['Rule performance looks good - ready for production'],
          execution_time_ms: 2340,
          created_at: new Date().toISOString()
        },
        {
          eval_id: 'eval_002',
          rule_name: 'SQL Injection Detection',
          rule_type: 'signature_detection',
          status: 'running',
          sample_size: 1500,
          created_at: new Date().toISOString()
        }
      ]);
    }
  };

  const fetchEvaluationDetails = async (evalId) => {
    try {
      const response = await axios.get(`/api/shadow/result?eval_id=${evalId}`);
      setSelectedEval(response.data);
    } catch (error) {
      console.error('Failed to fetch evaluation details:', error);
    }
  };

  useEffect(() => {
    fetchEvaluations();
    const interval = setInterval(fetchEvaluations, 10000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed': return '#00aa00';
      case 'running': return '#ffaa00';
      case 'failed': return '#ff0000';
      default: return '#666666';
    }
  };

  const getPerformanceRating = (f1Score) => {
    if (f1Score >= 0.9) return { rating: 'Excellent', color: '#00aa00' };
    if (f1Score >= 0.8) return { rating: 'Good', color: '#66aa00' };
    if (f1Score >= 0.7) return { rating: 'Fair', color: '#ffaa00' };
    if (f1Score >= 0.6) return { rating: 'Poor', color: '#ff6600' };
    return { rating: 'Very Poor', color: '#ff0000' };
  };

  return (
    <div className="shadow-evaluator">
      <div className="page-header">
        <h1>Shadow Evaluation</h1>
        <p>Test security rules against historical traffic without affecting production</p>
      </div>

      <div className="eval-content">
        <div className="create-eval-section">
          <h2>Create New Evaluation</h2>
          <div className="eval-form">
            <div className="form-group">
              <label>Rule Name:</label>
              <input
                type="text"
                value={newEval.rule_name}
                onChange={(e) => setNewEval({...newEval, rule_name: e.target.value})}
                placeholder="Enter rule name"
              />
            </div>

            <div className="form-group">
              <label>Rule Type:</label>
              <select
                value={newEval.rule_type}
                onChange={(e) => setNewEval({...newEval, rule_type: e.target.value})}
              >
                {ruleTypes.map(type => (
                  <option key={type.value} value={type.value}>{type.label}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label>Sample Size:</label>
              <input
                type="number"
                value={newEval.sample_size}
                onChange={(e) => setNewEval({...newEval, sample_size: parseInt(e.target.value)})}
                min="100"
                max="10000"
              />
            </div>

            <div className="rule-config">
              <h4>Rule Configuration Preview:</h4>
              <pre>{JSON.stringify(getRuleConfig(), null, 2)}</pre>
            </div>

            <button 
              onClick={createEvaluation} 
              disabled={loading}
              className="create-eval-btn"
            >
              {loading ? 'Creating...' : 'Start Shadow Evaluation'}
            </button>
          </div>
        </div>

        <div className="evaluations-list">
          <h2>Recent Evaluations</h2>
          <div className="eval-grid">
            {evaluations.map((eval) => (
              <div key={eval.eval_id} className="eval-card">
                <div className="eval-header">
                  <h3>{eval.rule_name}</h3>
                  <span 
                    className="eval-status"
                    style={{ color: getStatusColor(eval.status) }}
                  >
                    {eval.status.toUpperCase()}
                  </span>
                </div>

                <div className="eval-details">
                  <div className="eval-meta">
                    <span>Type: {eval.rule_type}</span>
                    <span>Samples: {eval.sample_size.toLocaleString()}</span>
                  </div>

                  {eval.status === 'completed' && (
                    <div className="eval-metrics">
                      <div className="metric-row">
                        <span>Precision:</span>
                        <span className="metric-value">{(eval.precision * 100).toFixed(1)}%</span>
                      </div>
                      <div className="metric-row">
                        <span>Recall:</span>
                        <span className="metric-value">{(eval.recall * 100).toFixed(1)}%</span>
                      </div>
                      <div className="metric-row">
                        <span>F1 Score:</span>
                        <span className="metric-value">{(eval.f1_score * 100).toFixed(1)}%</span>
                      </div>
                      <div className="metric-row">
                        <span>FP Rate:</span>
                        <span className="metric-value">{(eval.estimated_fp_rate * 100).toFixed(2)}%</span>
                      </div>

                      <div className="performance-rating">
                        <span 
                          style={{ color: getPerformanceRating(eval.f1_score).color }}
                        >
                          {getPerformanceRating(eval.f1_score).rating}
                        </span>
                      </div>
                    </div>
                  )}

                  {eval.status === 'running' && (
                    <div className="eval-progress">
                      <div className="progress-bar">
                        <div className="progress-fill"></div>
                      </div>
                      <span>Analyzing traffic samples...</span>
                    </div>
                  )}

                  <div className="eval-actions">
                    <button 
                      onClick={() => fetchEvaluationDetails(eval.eval_id)}
                      className="view-details-btn"
                    >
                      View Details
                    </button>
                    <span className="eval-time">
                      {new Date(eval.created_at).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {selectedEval && (
          <div className="eval-details-modal">
            <div className="modal-content">
              <div className="modal-header">
                <h2>Evaluation Details</h2>
                <button onClick={() => setSelectedEval(null)}>×</button>
              </div>

              <div className="modal-body">
                <div className="confusion-matrix">
                  <h3>Confusion Matrix</h3>
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th></th>
                        <th>Predicted Attack</th>
                        <th>Predicted Benign</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <th>Actual Attack</th>
                        <td className="tp">{selectedEval.true_positives}</td>
                        <td className="fn">{selectedEval.false_negatives}</td>
                      </tr>
                      <tr>
                        <th>Actual Benign</th>
                        <td className="fp">{selectedEval.false_positives}</td>
                        <td className="tn">{selectedEval.true_negatives}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="recommendations">
                  <h3>Recommendations</h3>
                  <ul>
                    {selectedEval.recommendations?.map((rec, index) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>

                <div className="execution-info">
                  <p><strong>Execution Time:</strong> {selectedEval.execution_time_ms}ms</p>
                  <p><strong>Sample Size:</strong> {selectedEval.sample_size.toLocaleString()}</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ShadowEvaluator;