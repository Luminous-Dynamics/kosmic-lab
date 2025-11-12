import React, { useState, useEffect, useMemo } from 'react';
import { LineChart, Line, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Play, Pause, RotateCcw, Download, Info } from 'lucide-react';

// Simulate real-time K-index data (in production, connect to simulation WebSocket)
const generateSimulationData = (timestep) => {
  const params = {
    temperature: 0.3 + Math.sin(timestep / 50) * 0.2,
    flow_rate: 0.5 + Math.cos(timestep / 40) * 0.3,
    noise_level: 0.2 + Math.random() * 0.1
  };
  
  // Simulate corridor: K > 1.0 when parameters are balanced
  const in_corridor = (
    params.temperature > 0.3 && params.temperature < 0.5 &&
    params.flow_rate > 0.4 && params.flow_rate < 0.7 &&
    params.noise_level < 0.3
  );
  
  const base_k = in_corridor ? 1.2 : 0.6;
  
  return {
    timestep,
    k_index: base_k + (Math.random() - 0.5) * 0.1,
    ...params,
    harmonies: {
      coherence: in_corridor ? 1.3 : 0.5,
      flourishing: in_corridor ? 1.2 : 0.6,
      wisdom: in_corridor ? 1.1 : 0.7,
      play: in_corridor ? 1.4 : 0.5,
      interconnection: in_corridor ? 1.2 : 0.6,
      reciprocity: in_corridor ? 1.3 : 0.5,
      evolution: in_corridor ? 1.1 : 0.7
    },
    phi: in_corridor ? 2.5 + Math.random() : 0.8 + Math.random() * 0.5,
    free_energy: in_corridor ? 3.0 - Math.random() : 8.0 + Math.random() * 2,
    survival_rate: in_corridor ? 0.85 + Math.random() * 0.1 : 0.3 + Math.random() * 0.2
  };
};

const KosmicDashboard = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [timestep, setTimestep] = useState(0);
  const [history, setHistory] = useState([]);
  const [currentData, setCurrentData] = useState(generateSimulationData(0));
  const [selectedView, setSelectedView] = useState('overview');

  // Simulation loop
  useEffect(() => {
    if (!isRunning) return;
    
    const interval = setInterval(() => {
      setTimestep(t => {
        const newT = t + 1;
        const data = generateSimulationData(newT);
        setCurrentData(data);
        setHistory(prev => [...prev.slice(-200), data]); // Keep last 200 points
        return newT;
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleReset = () => {
    setIsRunning(false);
    setTimestep(0);
    setHistory([]);
    setCurrentData(generateSimulationData(0));
  };

  // Parameter space heatmap data
  const parameterSpaceData = useMemo(() => {
    const grid = [];
    for (let temp = 0; temp <= 1; temp += 0.05) {
      for (let flow = 0; flow <= 1; flow += 0.05) {
        const in_corridor = (temp > 0.3 && temp < 0.5 && flow > 0.4 && flow < 0.7);
        grid.push({
          temperature: temp,
          flow_rate: flow,
          k_index: in_corridor ? 1.2 + Math.random() * 0.2 : 0.5 + Math.random() * 0.3,
          in_corridor
        });
      }
    }
    return grid;
  }, []);

  // Seven Harmonies radar data
  const radarData = currentData.harmonies ? [
    { harmony: 'Coherence', value: currentData.harmonies.coherence, baseline: 1.0 },
    { harmony: 'Flourishing', value: currentData.harmonies.flourishing, baseline: 1.0 },
    { harmony: 'Wisdom', value: currentData.harmonies.wisdom, baseline: 1.0 },
    { harmony: 'Play', value: currentData.harmonies.play, baseline: 1.0 },
    { harmony: 'Interconnection', value: currentData.harmonies.interconnection, baseline: 1.0 },
    { harmony: 'Reciprocity', value: currentData.harmonies.reciprocity, baseline: 1.0 },
    { harmony: 'Evolution', value: currentData.harmonies.evolution, baseline: 1.0 }
  ] : [];

  const inCorridor = currentData.k_index > 1.0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-950 via-purple-900 to-pink-950 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-pink-400">
          Kosmic Corridor Explorer
        </h1>
        <p className="text-gray-300 flex items-center gap-2">
          <Info size={16} />
          Real-time visualization of the Goldilocks Corridor in parameter space
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 mb-6 flex items-center justify-between">
        <div className="flex gap-4">
          <button
            onClick={() => setIsRunning(!isRunning)}
            className="flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-600 hover:to-blue-600 px-6 py-2 rounded-lg font-semibold transition-all"
          >
            {isRunning ? <Pause size={20} /> : <Play size={20} />}
            {isRunning ? 'Pause' : 'Run'}
          </button>
          <button
            onClick={handleReset}
            className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 px-6 py-2 rounded-lg font-semibold transition-all"
          >
            <RotateCcw size={20} />
            Reset
          </button>
        </div>
        
        <div className="flex items-center gap-6">
          <div className="text-right">
            <div className="text-sm text-gray-400">Timestep</div>
            <div className="text-2xl font-bold">{timestep}</div>
          </div>
          <div className={`text-right px-4 py-2 rounded-lg ${inCorridor ? 'bg-green-500/20 border border-green-500' : 'bg-red-500/20 border border-red-500'}`}>
            <div className="text-sm text-gray-400">K-Index</div>
            <div className="text-2xl font-bold">{currentData.k_index.toFixed(3)}</div>
            <div className="text-xs">{inCorridor ? '✓ IN CORRIDOR' : '✗ Outside'}</div>
          </div>
        </div>
      </div>

      {/* View Selector */}
      <div className="flex gap-2 mb-6">
        {['overview', 'harmonies', 'parameters'].map(view => (
          <button
            key={view}
            onClick={() => setSelectedView(view)}
            className={`px-4 py-2 rounded-lg font-semibold transition-all capitalize ${
              selectedView === view 
                ? 'bg-gradient-to-r from-pink-500 to-purple-500' 
                : 'bg-white/10 hover:bg-white/20'
            }`}
          >
            {view}
          </button>
        ))}
      </div>

      {/* Main Content */}
      {selectedView === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* K-Index Timeline */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">K-Index Evolution</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="timestep" stroke="#fff" />
                <YAxis stroke="#fff" domain={[0, 2]} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }} />
                <Legend />
                <Line type="monotone" dataKey="k_index" stroke="#22d3ee" strokeWidth={2} dot={false} name="K-Index" />
                <Line type="monotone" dataKey={1.0} stroke="#ef4444" strokeWidth={1} strokeDasharray="5 5" name="Corridor Threshold" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Current Metrics */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">System Metrics</h2>
            <div className="grid grid-cols-2 gap-4">
              <MetricCard label="Φ (Integration)" value={currentData.phi.toFixed(2)} unit="" color="cyan" />
              <MetricCard label="Free Energy" value={currentData.free_energy.toFixed(2)} unit="" color="pink" />
              <MetricCard label="Survival Rate" value={(currentData.survival_rate * 100).toFixed(1)} unit="%" color="green" />
              <MetricCard label="Temperature" value={currentData.temperature.toFixed(3)} unit="" color="orange" />
            </div>
          </div>

          {/* Φ vs Free Energy */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Integration vs Prediction (IIT × FEP)</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="timestep" stroke="#fff" />
                <YAxis yAxisId="left" stroke="#22d3ee" />
                <YAxis yAxisId="right" orientation="right" stroke="#ec4899" />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }} />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="phi" stroke="#22d3ee" strokeWidth={2} dot={false} name="Φ (Integration)" />
                <Line yAxisId="right" type="monotone" dataKey="free_energy" stroke="#ec4899" strokeWidth={2} dot={false} name="Free Energy" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Agent Survival */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Agent Survival Rate</h2>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="timestep" stroke="#fff" />
                <YAxis stroke="#fff" domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }} />
                <Legend />
                <Line type="monotone" dataKey="survival_rate" stroke="#10b981" strokeWidth={2} dot={false} name="Survival Rate" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {selectedView === 'harmonies' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Radar Chart */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Seven Primary Harmonies</h2>
            <ResponsiveContainer width="100%" height={400}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#ffffff40" />
                <PolarAngleAxis dataKey="harmony" stroke="#fff" />
                <PolarRadiusAxis domain={[0, 2]} stroke="#fff" />
                <Radar name="Current" dataKey="value" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} />
                <Radar name="Baseline" dataKey="baseline" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
            <div className={`mt-4 p-4 rounded-lg ${inCorridor ? 'bg-green-500/20 border border-green-500' : 'bg-yellow-500/20 border border-yellow-500'}`}>
              <p className="text-sm">
                {inCorridor 
                  ? '✓ All harmonies elevated above baseline (K > 1.0)' 
                  : '⚠ Some harmonies below threshold'}
              </p>
            </div>
          </div>

          {/* Harmony Timeline */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Harmony Evolution</h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="timestep" stroke="#fff" />
                <YAxis stroke="#fff" domain={[0, 2]} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }} />
                <Legend />
                <Line type="monotone" dataKey="harmonies.coherence" stroke="#22d3ee" strokeWidth={1} dot={false} name="H1" />
                <Line type="monotone" dataKey="harmonies.flourishing" stroke="#10b981" strokeWidth={1} dot={false} name="H2" />
                <Line type="monotone" dataKey="harmonies.wisdom" stroke="#f59e0b" strokeWidth={1} dot={false} name="H3" />
                <Line type="monotone" dataKey="harmonies.play" stroke="#ec4899" strokeWidth={1} dot={false} name="H4" />
                <Line type="monotone" dataKey="harmonies.interconnection" stroke="#8b5cf6" strokeWidth={1} dot={false} name="H5" />
                <Line type="monotone" dataKey="harmonies.reciprocity" stroke="#ef4444" strokeWidth={1} dot={false} name="H6" />
                <Line type="monotone" dataKey="harmonies.evolution" stroke="#06b6d4" strokeWidth={1} dot={false} name="H7" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Harmony Details */}
          <div className="lg:col-span-2 bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Current Harmony Scores</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              {radarData.map((h, i) => (
                <div key={i} className="text-center">
                  <div className="text-3xl font-bold" style={{ color: h.value > 1.0 ? '#10b981' : '#ef4444' }}>
                    {h.value.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">{h.harmony}</div>
                  <div className="text-xs mt-1">
                    {h.value > 1.0 ? '✓' : '✗'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {selectedView === 'parameters' && (
        <div className="grid grid-cols-1 gap-6">
          {/* Parameter Space Map */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Parameter Space: Corridor Map</h2>
            <ResponsiveContainer width="100%" height={500}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis type="number" dataKey="temperature" name="Temperature" domain={[0, 1]} stroke="#fff" />
                <YAxis type="number" dataKey="flow_rate" name="Flow Rate" domain={[0, 1]} stroke="#fff" />
                <ZAxis type="number" dataKey="k_index" range={[20, 200]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }}
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value, name) => [value.toFixed(3), name]}
                />
                <Legend />
                <Scatter 
                  name="Outside Corridor" 
                  data={parameterSpaceData.filter(d => !d.in_corridor)} 
                  fill="#ef4444" 
                  fillOpacity={0.3}
                />
                <Scatter 
                  name="In Corridor" 
                  data={parameterSpaceData.filter(d => d.in_corridor)} 
                  fill="#10b981" 
                  fillOpacity={0.6}
                />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="mt-4 p-4 bg-blue-500/20 border border-blue-500 rounded-lg">
              <p className="text-sm">
                💡 <strong>Goldilocks Corridor:</strong> Green region shows parameter combinations where K {'>'} 1.0 (all harmonies elevated).
                Current position: Temperature = {currentData.temperature.toFixed(3)}, Flow = {currentData.flow_rate.toFixed(3)}
              </p>
            </div>
          </div>

          {/* Parameter Trajectories */}
          <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6">
            <h2 className="text-xl font-bold mb-4">Parameter Evolution</h2>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff20" />
                <XAxis dataKey="timestep" stroke="#fff" />
                <YAxis stroke="#fff" domain={[0, 1]} />
                <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: 'none', borderRadius: '8px' }} />
                <Legend />
                <Line type="monotone" dataKey="temperature" stroke="#f59e0b" strokeWidth={2} dot={false} name="Temperature" />
                <Line type="monotone" dataKey="flow_rate" stroke="#06b6d4" strokeWidth={2} dot={false} name="Flow Rate" />
                <Line type="monotone" dataKey="noise_level" stroke="#ec4899" strokeWidth={2} dot={false} name="Noise" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="mt-8 text-center text-gray-400 text-sm">
        <p>Kosmic Simulation Suite v0.1 | Testing Recursive Meta-Intelligence</p>
        <p className="mt-1">Integration of Autopoiesis → IIT → FEP → Bioelectric → Multiscale TE</p>
      </div>
    </div>
  );
};

const MetricCard = ({ label, value, unit, color }) => {
  const colorMap = {
    cyan: 'from-cyan-500 to-blue-500',
    pink: 'from-pink-500 to-purple-500',
    green: 'from-green-500 to-emerald-500',
    orange: 'from-orange-500 to-red-500'
  };

  return (
    <div className={`p-4 rounded-lg bg-gradient-to-br ${colorMap[color]} bg-opacity-20`}>
      <div className="text-sm text-gray-300 mb-1">{label}</div>
      <div className="text-2xl font-bold">
        {value}
        <span className="text-sm ml-1">{unit}</span>
      </div>
    </div>
  );
};

export default KosmicDashboard;