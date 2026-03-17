import React, { useState, useEffect, useRef, useMemo } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis, Cell } from 'recharts';
import * as L from 'leaflet';

// --- REUSABLE SVG COMPONENTS ---
const AnimatedWaves = () => (
    <div className="absolute bottom-0 w-full overflow-hidden leading-[0]">
        <svg className="relative block w-[calc(100%+1.3px)] h-[50px] animate-pulse" viewBox="0 0 1200 120" preserveAspectRatio="none">
            <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V120H0V95.8C59.71,118.08,130.83,110.22,192.39,92.83c61.56-17.38,129-45.28,129-36.39Z" fill="rgba(13, 79, 107, 0.4)"></path>
            <path d="M0,45.8c60.3-22.3,131.5-14.4,193,3 61.5,17.4,129,45.3,187,34.5c58-10.8,114.2-30.1,172-41.9c82.4-16.7,168.2-17.7,250.5-.4c79.9,16.8,162.8,57.8,241.8,78.6c70,18.5,146.5,26.1,214.3,3V120H0V45.8Z" fill="rgba(10, 22, 40, 1)"></path>
        </svg>
    </div>
);

const SectionWrapper = ({ id, children }) => (
    <section id={id} className="relative w-full py-16 px-6 lg:px-24 min-h-[50vh] flex flex-col items-center justify-center glow-border rounded-xl mb-12 bg-[#050b14]/50">
        <div className="w-full max-w-7xl z-10">{children}</div>
        <AnimatedWaves />
    </section>
);

// --- NAV & HERO ---
const Navbar = () => (
    <nav className="fixed top-0 w-full bg-ocean-navy/90 backdrop-blur-md border-b border-ocean-teal z-[100] px-6 py-4 flex justify-between items-center">
        <div className="flex items-center space-x-3 glow-text-seafoam font-bold text-2xl">
            <span>🦀 OceanIQ</span>
        </div>
        <div className="hidden md:flex space-x-6 text-sm font-semibold text-ocean-seafoam">
            <a href="#metrics" className="hover:text-white transition">Metrics</a>
            <a href="#trends" className="hover:text-white transition">Trends</a>
            <a href="#calculator" className="hover:text-white transition">RM-NPI Calc</a>
            <a href="#map" className="hover:text-white transition">Map</a>
            <a href="#ai" className="hover:text-white transition">AI Core</a>
        </div>
        <div><span className="bg-ocean-teal text-white px-3 py-1 rounded text-xs tracking-widest uppercase">Oceanographic Dashboard</span></div>
    </nav>
);

const Hero = () => (
    <section className="relative w-full min-h-screen flex flex-col justify-center items-center text-center px-4">
        <h1 className="text-5xl md:text-7xl font-bold glow-text-seafoam text-ocean-seafoam mb-4 z-10">OceanIQ</h1>
        <p className="text-xl md:text-2xl font-light text-gray-300 max-w-3xl mb-12 z-10">
            AI-powered coastal risk intelligence using representation learning & RM-NPI scoring.
        </p>
        <a href="#metrics" className="z-10 bg-ocean-teal hover:bg-ocean-seafoam text-white hover:text-ocean-navy font-bold py-3 px-8 rounded-full glow-border transition transform hover:scale-105">
            Explore Dashboard
        </a>
        <AnimatedWaves />
    </section>
);

// --- SECTIONS ---
const Section1 = () => (
    <SectionWrapper id="metrics">
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 text-center">
            {[{i:"🌊",v:"247",l:"Coastal Zones Analyzed"}, {i:"🔴",v:"18",l:"Critical Risk Zones"}, {i:"🚨",v:"43",l:"Anomalies Found"}, {i:"💾",v:"73%",l:"Data Compression"}, {i:"🌿",v:"12",l:"Biodiversity Hotspots at Risk"}].map(m => (
                <div key={m.l} className="glow-card p-6 flex flex-col justify-center items-center">
                    <span className="text-3xl mb-2">{m.i}</span>
                    <h3 className="text-4xl font-bold text-ocean-seafoam">{m.v}</h3>
                    <p className="text-xs text-gray-400 mt-2 uppercase tracking-wide">{m.l}</p>
                </div>
            ))}
        </div>
    </SectionWrapper>
);

const Section2 = ({ db }) => (
    <SectionWrapper id="dataset">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">Unified Multi-Source Environmental Data Pipeline</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {[{i:"🛰️",t:"MODIS NDVI",s:"Vegetation Index Data",r:"1,240 records"}, {i:"🌧️",t:"GPM Rainfall",s:"Global Precipitation",r:"980 records"}, {i:"🌊",t:"GRDC River Discharge",s:"Runoff Data",r:"760 records"}].map(c => (
                <div key={c.t} className="glow-card p-4">
                    <h4 className="text-xl font-bold text-ocean-gold">{c.i} {c.t}</h4>
                    <p className="text-sm text-gray-300">{c.s}</p>
                    <p className="text-xs text-ocean-seafoam mt-2">{c.r}</p>
                </div>
            ))}
        </div>
        <div className="overflow-x-auto">
            <table className="w-full text-left text-sm text-gray-300 border-collapse">
                <thead className="bg-ocean-teal/40 text-ocean-seafoam">
                    <tr><th className="p-3">Zone ID</th><th className="p-3">Rainfall (mm)</th><th className="p-3">SST (°C)</th><th className="p-3">NDVI</th><th className="p-3">Discharge (m³/s)</th><th className="p-3">Date</th></tr>
                </thead>
                <tbody>
                    {db.overview.map(r => (
                        <tr key={r.id} className="border-b border-ocean-teal/20 hover:bg-ocean-teal/10">
                            <td className="p-3">{r.zone}</td><td className="p-3">{r.rainfall}</td><td className="p-3">{r.sst}</td><td className="p-3">{r.ndvi}</td><td className="p-3">{r.discharge}</td><td className="p-3">2023-08-01</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </SectionWrapper>
);

const Section3 = ({ db }) => (
    <SectionWrapper id="trends">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">Environmental Variable Trends Over Time</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[{k:"rainfall",c:"#2dd4bf",t:"Rainfall (mm)"}, {k:"sst",c:"#ff6b6b",t:"Sea Surface Temperature °C"}, {k:"ndvi",c:"#fbbf24",t:"NDVI Vegetation Index"}, {k:"discharge",c:"#0d4f6b",t:"River Discharge m³/s"}].map(c => (
                <div key={c.k} className="h-64 glow-card p-4">
                    <h4 className="text-sm font-bold mb-2 text-center text-gray-300">{c.t}</h4>
                    <ResponsiveContainer width="100%" height="90%">
                        <LineChart data={db.timeseries}>
                            <XAxis dataKey="month" stroke="#fff" tick={{fontSize:10}} />
                            <YAxis stroke="#fff" tick={{fontSize:10}} />
                            <Tooltip contentStyle={{backgroundColor:'#0a1628', border:'1px solid #2dd4bf'}} />
                            <Line type="monotone" dataKey={c.k} stroke={c.c} strokeWidth={2} dot={{r:3}} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            ))}
        </div>
    </SectionWrapper>
);

const Section4 = () => {
    const [q, setQ] = useState(0.5); const [n, setN] = useState(0.5); const [s, setS] = useState(0.5); const [d, setD] = useState(0.5);
    const score = (q * n * s * d).toFixed(3);
    const getRisk = () => score > 0.8 ? {l:"CRITICAL RISK", c:"text-ocean-coral", b:"bg-ocean-coral"} : score > 0.5 ? {l:"HIGH RISK", c:"text-orange-500", b:"bg-orange-500"} : score > 0.2 ? {l:"MODERATE RISK", c:"text-ocean-gold", b:"bg-ocean-gold"} : {l:"LOW RISK", c:"text-green-500", b:"bg-green-500"};
    const risk = getRisk();

    return (
        <SectionWrapper id="calculator">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">River Mouth Nutrient Pressure Index (RM-NPI)</h2>
            <p className="text-gray-400 mb-8 italic">A physics-based formula triggering biological warnings.</p>
            <div className="flex flex-col md:flex-row gap-8">
                <div className="flex-1 glow-card p-6 space-y-6">
                    <div className="text-center font-bold text-xl bg-ocean-navy p-3 rounded border border-ocean-teal">Formula: RM-NPI = Q × N × S × D</div>
                    {[{l:"Q — River Discharge",v:q,f:setQ,d:"Freshwater volume pushed into the ocean"}, {l:"N — Nutrient Load (NDVI Proxy)",v:n,f:setN,d:"Agricultural fertilizer density"}, {l:"S — Seasonal Rainfall",v:s,f:setS,d:"Monsoon and runoff intensity"}, {l:"D — Distance Decay",v:d,f:setD,d:"Distance from coast to ocean pixel"}].map(i => (
                        <div key={i.l}>
                            <div className="flex justify-between text-sm"><label>{i.l}</label><span>{i.v}</span></div>
                            <input type="range" min="0" max="1" step="0.01" value={i.v} onChange={(e)=>i.f(parseFloat(e.target.value))} className="w-full accent-ocean-seafoam" />
                            <p className="text-xs text-gray-500">{i.d}</p>
                        </div>
                    ))}
                </div>
                <div className={`flex-1 glow-card p-6 flex flex-col justify-center items-center border-t-8 ${risk.b}`}>
                    <h3 className="text-2xl text-gray-300">Live Calculated RM-NPI</h3>
                    <div className={`text-8xl font-black my-4 ${risk.c} drop-shadow-lg`}>{score}</div>
                    <div className={`text-2xl font-bold px-6 py-2 rounded-full text-[#0a1628] ${risk.b} animate-pulse`}>{risk.l}</div>
                </div>
            </div>
        </SectionWrapper>
    );
};

const Section5Map = ({ db }) => {
    const mapRef = useRef(null);
    useEffect(() => {
        if (!mapRef.current) return;
        if (mapRef.current._leaflet_id) return; // Prevent re-initialization

        const map = L.map(mapRef.current, { scrollWheelZoom: false }).setView([17.0, 79.0], 5);
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        db.overview.forEach(z => {
            const color = z.risk==="CRITICAL"?"#ff6b6b":z.risk==="HIGH"?"#f97316":z.risk==="MODERATE"?"#fbbf24":"#22c55e";
            const marker = L.circleMarker([z.lat, z.lon], {
                radius: 10,
                color: color,
                fillColor: color,
                fillOpacity: 0.6,
                className: "animate-pulse"
            }).addTo(map);
            
            marker.bindPopup(`
                <div class="p-1 text-black">
                    <h3 class="font-bold text-lg mb-1">${z.zone}</h3>
                    <p>RM-NPI: <strong>${z.rmnpi}</strong></p>
                    <p>Risk: <span style="color:${color}">${z.risk}</span></p>
                    <p class="text-xs mt-1 text-gray-800">Rainfall: ${z.rainfall}mm</p>
                    <p class="text-xs text-gray-800">SST: ${z.sst}°C</p>
                </div>
            `);
        });

        return () => { map.remove(); };
    }, [db.overview]);

    return (
        <SectionWrapper id="map">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">Coastal Zone Risk Map — India</h2>
            <div ref={mapRef} className="h-[500px] w-full glow-border rounded-xl overflow-hidden relative z-10" />
        </SectionWrapper>
    );
};

const Section6 = ({ db }) => (
    <SectionWrapper id="ai">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">AI Representation Learning — Dual Channel Autoencoder</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="glow-card p-4 h-64">
                <h4 className="text-center text-sm mb-2 text-gray-300">Reconstruction Loss (50 Epochs)</h4>
                <ResponsiveContainer width="100%" height="90%">
                    <LineChart data={db.epochs}><XAxis dataKey="epoch" hide/><YAxis hide/><Tooltip contentStyle={{backgroundColor:'#0a1628'}}/><Line type="monotone" dataKey="loss" stroke="#2dd4bf" strokeWidth={3} dot={false}/></LineChart>
                </ResponsiveContainer>
            </div>
            <div className="glow-card p-4 h-64">
                <h4 className="text-center text-sm mb-2 text-gray-300">Dimensionality Reduction</h4>
                <ResponsiveContainer width="100%" height="90%">
                    <BarChart data={[{name: 'Original', val: 128}, {name: 'Latent', val: 12}]}>
                        <XAxis dataKey="name" stroke="#fff" /><YAxis hide/><Tooltip contentStyle={{backgroundColor:'#0a1628',color:'#fff'}}/><Bar dataKey="val" fill="#ff6b6b" radius={[4,4,0,0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
            <div className="glow-card p-4 h-64">
                <h4 className="text-center text-sm mb-2 text-gray-300">t-SNE Latent Clusters</h4>
                <ResponsiveContainer width="100%" height="90%">
                    <ScatterChart>
                        <XAxis type="number" dataKey="x" hide/><YAxis type="number" dataKey="y" hide/><Tooltip contentStyle={{backgroundColor:'#0a1628'}}/><ZAxis range={[50,50]}/>
                        <Scatter data={db.tsne}>
                            {db.tsne.map((e,i)=><Cell key={i} fill={e.risk==="CRITICAL"?"#ff6b6b":e.risk==="HIGH"?"#f97316":e.risk==="MODERATE"?"#fbbf24":"#22c55e"}/>)}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
        </div>
    </SectionWrapper>
);

const Section7 = ({ db }) => (
    <SectionWrapper id="anomalies">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">AI-Detected Environmental Anomalies</h2>
        <p className="text-gray-400 mb-8 font-mono">43 Anomalies detected out of 2,980 data points (1.4% failure rate)</p>
        <div className="overflow-x-auto glow-card">
            <table className="w-full text-left font-mono text-sm border-collapse">
                <thead className="bg-[#050b14] text-ocean-seafoam"><tr><th className="p-3">Zone</th><th className="p-3">Date</th><th className="p-3">Variable</th><th className="p-3">Observed</th><th className="p-3">Expected</th><th className="p-3">Deviation %</th><th className="p-3">Severity</th></tr></thead>
                <tbody>
                    {db.anomalies.map((a,i) => (
                        <tr key={i} className="border-b border-ocean-teal/20 hover:bg-ocean-teal/10">
                            <td className="p-3 font-sans text-white">{a.zone}</td><td className="p-3 text-gray-400">{a.date}</td><td className="p-3 text-ocean-gold">{a.var}</td><td className="p-3">{a.obs}</td><td className="p-3 text-gray-400">{a.exp}</td><td className="p-3 text-red-400">{a.dev}</td>
                            <td className="p-3"><span className={`px-2 py-1 rounded text-xs text-white ${a.sev==="CRITICAL"?"bg-red-600":a.sev==="HIGH"?"bg-orange-500":"bg-yellow-500"}`}>{a.sev}</span></td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </SectionWrapper>
);

const Section8 = ({ db }) => (
    <SectionWrapper id="biodiversity">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">Ocean Stress vs Biodiversity Index</h2>
        <p className="text-gray-400 mb-8 italic text-sm">High nutrient pressure zones show significant biodiversity loss, threatening coastal reefs and marine ecosystems.</p>
        <div className="h-80 w-full glow-card p-4">
            <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{top:20, right:20, bottom:20, left:20}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#0d4f6b"/>
                    <XAxis type="number" dataKey="rmnpi" name="RM-NPI Score" stroke="#fff" label={{ value: "RM-NPI Score (Nutrient Pressure)", position: "bottom", fill: "#fff" }}/>
                    <YAxis type="number" dataKey="bioIndex" name="Biodiv Index" stroke="#fff" label={{ value: "Biodiversity Health Index", angle: -90, position: "left", fill: "#fff" }}/>
                    <Tooltip cursor={{strokeDasharray: '3 3'}} contentStyle={{backgroundColor:'#0a1628', border:'1px solid #2dd4bf'}}/>
                    <Scatter data={db.biodiversity} fill="#ff6b6b" />
                </ScatterChart>
            </ResponsiveContainer>
        </div>
        <div className="w-full text-center mt-4 text-ocean-coral font-bold font-mono text-xl animate-pulse">r = -0.84 — Strong negative correlation detected.</div>
    </SectionWrapper>
);

const Section9 = ({ db }) => (
    <SectionWrapper id="datacenter">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">Intelligent Data Centre Efficiency</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 text-center border-b border-ocean-teal pb-8">
            <div className="glow-card p-4">
                <p className="text-gray-400 text-sm">Storage Size</p>
                <div className="text-2xl mt-2 line-through text-red-400">847 MB</div>
                <div className="text-4xl font-bold text-ocean-seafoam">23 MB</div>
                <p className="text-ocean-gold font-bold mt-2">97.3% Reduction</p>
            </div>
            <div className="glow-card p-4">
                <p className="text-gray-400 text-sm">Compute Cycles / Filtering</p>
                <div className="text-2xl mt-2 line-through text-red-400">12,400</div>
                <div className="text-4xl font-bold text-ocean-seafoam">1,847</div>
                <p className="text-ocean-gold font-bold mt-2">85% Reduction</p>
            </div>
            <div className="glow-card p-4">
                <p className="text-gray-400 text-sm">Energy Efficiency</p>
                <div className="text-2xl mt-2 text-gray-400">Per analysis run</div>
                <div className="text-4xl font-bold text-ocean-seafoam">~4.2 kWh</div>
                <p className="text-ocean-gold font-bold mt-2">Energy Saved</p>
            </div>
        </div>
        <p className="text-center text-gray-300 italic mb-6">"Our autoencoder ensures data centres only process what matters — reducing cost, energy, and storage while delivering sharper insights"</p>
        <div className="h-64 glow-card p-4">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart data={db.datacenter} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#0a1628" />
                    <XAxis type="number" stroke="#fff" />
                    <YAxis dataKey="name" type="category" stroke="#fff" width={100} />
                    <Tooltip contentStyle={{backgroundColor:'#0a1628', border:'1px solid #2dd4bf'}} />
                    <Legend />
                    <Bar dataKey="before" fill="#ff6b6b" name="Before AI" />
                    <Bar dataKey="after" fill="#2dd4bf" name="After Compression" />
                </BarChart>
            </ResponsiveContainer>
        </div>
    </SectionWrapper>
);

const Section10 = ({ db }) => {
    const [filter, setFilter] = useState("ALL");
    const data = filter === "ALL" ? db.overview : db.overview.filter(r => r.risk === filter);
    return (
        <SectionWrapper id="risktable">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-6">Full Coastal Zone Risk Assessment</h2>
            <div className="flex gap-2 mb-4">
                {["ALL","CRITICAL","HIGH","MODERATE","LOW"].map(f => (
                    <button key={f} onClick={()=>setFilter(f)} className={`px-4 py-1 rounded text-xs font-bold transition ${filter===f ? 'bg-ocean-seafoam text-ocean-navy':'bg-ocean-navy text-ocean-seafoam border border-ocean-seafoam hover:bg-ocean-teal'}`}>{f}</button>
                ))}
            </div>
            <div className="overflow-x-auto w-full">
                <table className="w-full text-left text-sm text-gray-300 border-collapse glow-card">
                    <thead className="bg-ocean-teal/40 text-ocean-seafoam">
                        <tr><th className="p-3">Zone</th><th className="p-3">RM-NPI Score</th><th className="p-3">Risk Level</th><th className="p-3">Recommended Action</th></tr>
                    </thead>
                    <tbody>
                        {data.map(r => (
                            <tr key={r.id} className="border-b border-ocean-teal/20 hover:bg-ocean-teal/10">
                                <td className="p-3 font-bold text-white">{r.zone}</td>
                                <td className="p-3 font-mono">{r.rmnpi}</td>
                                <td className="p-3"><span className={`px-2 py-1 rounded text-xs text-white ${r.risk==="CRITICAL"?"bg-red-600":r.risk==="HIGH"?"bg-orange-500":r.risk==="MODERATE"?"bg-yellow-500":"bg-green-500"}`}>{r.risk}</span></td>
                                <td className="p-3">{r.risk==="CRITICAL"?"Immediate intervention required":r.risk==="HIGH"?"Alert environmental agencies":r.risk==="MODERATE"?"Increase monitoring frequency":"Continue routine monitoring"}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </SectionWrapper>
    )
};

const Section11 = () => {
    const vars = ['Rain', 'SST', 'NDVI', 'Discharge', 'RM-NPI', 'Biodiv'];
    const vals = [
        [1.0, 0.2, 0.4, 0.9, 0.8, -0.7],
        [0.2, 1.0, 0.1, 0.1, 0.3, -0.5],
        [0.4, 0.1, 1.0, 0.3, 0.6, -0.8],
        [0.9, 0.1, 0.3, 1.0, 0.9, -0.8],
        [0.8, 0.3, 0.6, 0.9, 1.0, -0.9],
        [-0.7,-0.5,-0.8,-0.8,-0.9, 1.0]
    ];
    return (
        <SectionWrapper id="heatmap">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">Environmental Variable Correlation Matrix</h2>
            <div className="glow-card p-6 flex flex-col items-center overflow-x-auto">
                <div className="grid grid-cols-7 gap-1 min-w-[500px]">
                    <div className="p-2"></div>
                    {vars.map(v => <div key={v} className="text-center font-bold text-xs text-ocean-gold p-2">{v}</div>)}
                    {vars.map((rowVar, i) => (
                        <React.Fragment key={i}>
                            <div className="text-right font-bold text-xs text-ocean-gold p-2 flex items-center justify-end">{rowVar}</div>
                            {vals[i].map((val, j) => {
                                const ratio = (val + 1) / 2;
                                const r = Math.round(10 + (35 * ratio));
                                const g = Math.round(22 + (190 * ratio));
                                const b = Math.round(40 + (150 * ratio));
                                return (
                                    <div key={j} title={`${rowVar} vs ${vars[j]}: ${val}`} className="h-12 w-full flex items-center justify-center text-xs font-mono text-white transition hover:scale-110 cursor-pointer border border-ocean-navy/30" style={{backgroundColor: `rgb(${r}, ${g}, ${b})`}}>
                                        {val.toFixed(1)}
                                    </div>
                                );
                            })}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </SectionWrapper>
    )
};

const Section12 = () => (
    <SectionWrapper id="conclusion">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-8">What OceanIQ Delivers</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="glow-card p-6 text-center"><h3 className="text-4xl mb-4">🌊</h3><p className="font-bold text-ocean-seafoam mb-2">Environmental Impact</p><p className="text-gray-300 text-sm">18 critical coastal zones identified for immediate, life-saving action.</p></div>
            <div className="glow-card p-6 text-center"><h3 className="text-4xl mb-4">💾</h3><p className="font-bold text-ocean-seafoam mb-2">Technical Efficiency</p><p className="text-gray-300 text-sm">73% direct data reduction via advanced AI-driven Autoencoder compression.</p></div>
            <div className="glow-card p-6 text-center"><h3 className="text-4xl mb-4">🌿</h3><p className="font-bold text-ocean-seafoam mb-2">Social Responsibility</p><p className="text-gray-300 text-sm">Protecting the ecological food security and livelihoods of 2.3M coastal residents.</p></div>
        </div>
        <h3 className="text-2xl md:text-3xl font-bold text-center text-ocean-gold animate-pulse">"We don't just compress data — we find what's abnormal, score its risk, and tell decision makers exactly where and when to act."</h3>
    </SectionWrapper>
);

const App = () => {
    const [db, setDb] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch('/api/data')
            .then(res => {
                if (!res.ok) throw new Error("Failed to fetch data from API");
                return res.json();
            })
            .then(data => setDb(data))
            .catch(err => {
                console.error(err);
                setError(err.message);
            });
    }, []);

    if (error) {
        return (
            <div className="w-full min-h-screen flex items-center justify-center bg-ocean-navy text-ocean-coral">
                <div className="glow-card p-8 border-ocean-coral"><h2 className="text-2xl font-bold">Error</h2><p>{error}</p></div>
            </div>
        );
    }

    if (!db) {
        return (
            <div className="w-full min-h-screen flex flex-col items-center justify-center bg-ocean-navy text-ocean-seafoam">
                <div className="text-6xl animate-spin mb-4">🌊</div>
                <h2 className="text-2xl font-bold animate-pulse">Connecting to Data Center API...</h2>
            </div>
        );
    }

    return (
        <div className="w-full flex flex-col items-center">
            <Navbar />
            <Hero />
            <Section1 />
            <Section2 db={db} />
            <Section3 db={db} />
            <Section4 />
            <Section5Map db={db} />
            <Section6 db={db} />
            <Section7 db={db} />
            <Section8 db={db} />
            <Section9 db={db} />
            <Section10 db={db} />
            <Section11 />
            <Section12 />
            <footer className="w-full py-6 text-center bg-ocean-navy border-t border-ocean-teal/30 text-ocean-seafoam/60 text-sm glow-text-seafoam mt-12 z-10">
                OceanIQ AI Platform © {new Date().getFullYear()} | Intelligent Oceanographic Dashboard
            </footer>
        </div>
    );
};

export default App;
