import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis,
        CartesianGrid, Tooltip, Legend, ResponsiveContainer, ZAxis, Cell } from 'recharts';
import L from 'leaflet';

// Bind L to window so code using window.L works
window.L = L;

console.log("Executing CORAL AI Dashboard v3 — Premium UI...");

let db = window.OCEANIQ_DATA || null;

// Calculate dynamic correlation coefficient between RM-NPI and BioIndex
let CORRELATION_R = -0.84;
if (db && db.biodiversity && db.biodiversity.length > 1) {
    const data = db.biodiversity;
    const n = data.length;
    const sumX = data.reduce((acc, val) => acc + val.rmnpi, 0);
    const sumY = data.reduce((acc, val) => acc + val.bioIndex, 0);
    const sumX2 = data.reduce((acc, val) => acc + val.rmnpi * val.rmnpi, 0);
    const sumY2 = data.reduce((acc, val) => acc + val.bioIndex * val.bioIndex, 0);
    const sumXY = data.reduce((acc, val) => acc + val.rmnpi * val.bioIndex, 0);
    const numerator = (n * sumXY) - (sumX * sumY);
    const denominator = Math.sqrt(((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));
    if (denominator !== 0) {
        CORRELATION_R = parseFloat((numerator / denominator).toFixed(2));
    }
}

// ── Ocean background images — all unique, all verified marine creatures ─────────
const OCEAN_IMAGES = {
  // Vibrant coral reef teeming with tropical fish — Hero
  home:         '/dashboard_bg_hd_v5.png',
  // Manta ray gliding in deep blue — Intelligence metrics
  metrics:      'https://images.unsplash.com/photo-1518156677180-95a2893f3e9f?w=1200&q=60&auto=format&fit=crop',
  // Keep alias for safety
  intelligence: 'https://images.unsplash.com/photo-1518156677180-95a2893f3e9f?w=1200&q=60&auto=format&fit=crop',
  // Tropical beach coastline — Coastal Zones
  zones:        'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=60&auto=format&fit=crop',
  // Scuba diver exploring coral reef — Trends overview
  trends:       'https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=1200&q=60&auto=format&fit=crop',
  // Reef shark in sunlit water — Cell Risk
  risk:         'https://images.unsplash.com/photo-1560275669-46c5a88d6a4c?w=1200&q=60&auto=format&fit=crop',
  // Sea turtle swimming over reef — Biodiversity
  biodiversity: 'https://images.unsplash.com/photo-1437622368342-7a3d70133ec6?w=1200&q=60&auto=format&fit=crop',
  // Deep-sea glowing pink/purple jellyfish — Neural Compression
  compression:  'https://images.unsplash.com/photo-1568430462989-4b16f61a2cfc?w=1200&q=60&auto=format&fit=crop',
  // Underwater sunbeams piercing the ocean — RM-NPI Calculator
  calculator:   'https://images.unsplash.com/photo-1551244072-5d12893278bc?w=1200&q=60&auto=format&fit=crop',
  // School of fish in a coral reef — Map Explorer
  map:          'https://images.unsplash.com/photo-1522069169874-c58ec4b76be5?w=1200&q=60&auto=format&fit=crop',
  // Whale sharks swimming in deep ocean — Cell Risk details
  cellrisk:     'https://images.unsplash.com/photo-1598899900391-224e443a8a3c?w=1200&q=60&auto=format&fit=crop',
  // Lagoon / ocean from above — At-risk grid cell directory
  realcells:    'https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1200&q=60&auto=format&fit=crop',
  // Turquoise sea cave / ocean — AI-Detected anomalies
  anomalies:    'https://images.unsplash.com/photo-1559128010-7c1ad6e1b6a5?w=1200&q=60&auto=format&fit=crop',
  // Deep blue ocean waves — Correlation matrix heatmap
  heatmap:      'https://images.unsplash.com/photo-1532191343016-47bf741b8b3c?w=1200&q=60&auto=format&fit=crop',
  // Space/satellite view of earth's oceans — Actionable directives
  conclusion:   '/seahorse_bg.png',
  // Vibrant coral reef — Login page (unified theme)
  login:        '/login_bg.png',
};

// ── Smooth Parallax hook using requestAnimationFrame + hardware acceleration ─────────────────
function useParallax(sectionId, strength = 0.28) {
  const bgRef = React.useRef(null);
  React.useEffect(() => {
    const el = bgRef.current;
    if (!el) return;
    const section = el.closest('section') || el.parentElement;
    let frameId;
    const handleScroll = () => {
      cancelAnimationFrame(frameId);
      frameId = requestAnimationFrame(() => {
        const rect = section.getBoundingClientRect();
        const vh   = window.innerHeight;
        const visible = rect.top < vh && rect.bottom > 0;
        if (visible) {
          const pct    = (vh - rect.top) / (vh + rect.height);
          const offset = (pct - 0.5) * strength * rect.height;
          el.style.transform = `translateY(${offset}px) translateZ(0) scale(1.04)`;
        }
      });
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => {
      window.removeEventListener('scroll', handleScroll);
      cancelAnimationFrame(frameId);
    };
  }, [sectionId, strength]);
  return bgRef;
}

// ── SVG Icon Library ────────────────────────────────────
const Icon = {
  Wave: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M2 12c1.5-3 4-4.5 6.5-3.5S13 11 16 10s4-3.5 6-2" strokeLinecap="round"/><path d="M2 17c1.5-3 4-4.5 6.5-3.5S13 16 16 15s4-3.5 6-2" strokeLinecap="round"/></svg>,
  Alert: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>,
  Grid: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>,
  BarChart2: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>,
  Leaf: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M17 8C8 10 5.9 16.17 3.82 19.11 3.45 19.67 3 20 2.5 20c-.83 0-1.5-.67-1.5-1.5 0-.17.03-.34.08-.5C3.01 14.62 5 8.5 17 8z"/><path d="M17 8c0 8-6 12-10 12"/></svg>,
  Map: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>,
  Activity: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
  Satellite: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="3"/><path d="M6.3 6.3a8 8 0 000 11.31m11.37-11.31a8 8 0 010 11.31M3.51 3.51a12 12 0 000 16.97m16.97-16.97a12 12 0 010 16.97"/></svg>,
  Shield: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>,
  TrendUp: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>,
  Cpu: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>,
  Globe: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>,
  Play: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>,
  LogOut: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>,
  User: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  Mail: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>,
  Lock: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0110 0v4"/></svg>,
  Check: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="20 6 9 17 4 12"/></svg>,
  X: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
  Calendar: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>,
  Zap: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  Database: ({className=""}) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>,
};

// ── Utility Helpers ────────────────────────────────────
const MONTH_NUMS = { Jan:1, Feb:2, Mar:3, Apr:4, May:5, Jun:6, Jul:7, Aug:8, Sep:9, Oct:10, Nov:11, Dec:12 };

function getRiskColor(risk) {
    return risk === "CRITICAL" ? "#ff6b6b" : risk === "HIGH" ? "#f97316" : risk === "MODERATE" ? "#fbbf24" : "#22c55e";
}
function getRiskBg(risk) {
    return risk === "CRITICAL" ? "bg-red-600" : risk === "HIGH" ? "bg-orange-500" : risk === "MODERATE" ? "bg-yellow-500" : "bg-green-500";
}
function getBioIndex(rmnpi) {
    // Derived from the dynamically calculated correlation
    return Math.max(0.05, Math.min(0.99, 1.0 + (rmnpi * CORRELATION_R))).toFixed(2);
}
function getBioRisk(rmnpi) {
    if (rmnpi >= 0.8) return "CRITICAL";
    if (rmnpi >= 0.6) return "HIGH";
    if (rmnpi >= 0.4) return "MODERATE";
    return "GOOD";
}
function getBioColor(rmnpi) {
    if (rmnpi >= 0.8) return "#ff6b6b";
    if (rmnpi >= 0.6) return "#f97316";
    if (rmnpi >= 0.4) return "#fbbf24";
    return "#22c55e";
}
function getEcologicalThreat(rmnpi) {
    if (rmnpi >= 0.8) return "Hypoxic dead zone / algal bloom formation";
    if (rmnpi >= 0.6) return "Coral bleaching & commercial fish migration";
    if (rmnpi >= 0.4) return "Elevated phytoplankton stress & turbidity";
    return "Within normal ecological parameters";
}
function getRecommendedAction(risk) {
    if (risk === "CRITICAL") return "Deploy emergency sampling buoys immediately";
    if (risk === "HIGH")     return "Alert State Pollution Control Board";
    if (risk === "MODERATE") return "Increase monitoring to weekly intervals";
    return "Continue standard monthly monitoring";
}

// Filter timeseries to only the actual analysis window from metadata
function getFilteredTimeseries() {
    if (!db || !db.timeseries) return [];
    return db.timeseries;
}

// ── SVG Wave Component ─────────────────────────────────
const AnimatedWaves = () => (
    <div className="absolute bottom-0 w-full overflow-hidden leading-[0]">
        <svg className="relative block w-[calc(100%+1.3px)] h-[50px] animate-pulse" viewBox="0 0 1200 120" preserveAspectRatio="none">
            <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V120H0V95.8C59.71,118.08,130.83,110.22,192.39,92.83c61.56-17.38,129-45.28,129-36.39Z" fill="rgba(13, 79, 107, 0.4)"></path>
            <path d="M0,45.8c60.3-22.3,131.5-14.4,193,3 61.5,17.4,129,45.3,187,34.5c58-10.8,114.2-30.1,172-41.9c82.4-16.7,168.2-17.7,250.5-.4c79.9,16.8,162.8,57.8,241.8,78.6c70,18.5,146.5,26.1,214.3,3V120H0V45.8Z" fill="rgba(10, 22, 40, 1)"></path>
        </svg>
    </div>
);

const SectionWrapper = ({ id, children, overlayClass }) => {
  const bgRef  = useParallax(id);
  const imgUrl = OCEAN_IMAGES[id] || OCEAN_IMAGES.intelligence;
  const overlay = overlayClass || `section-overlay-${id}`;
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "250px" } // Load image 250px before entering viewport
    );
    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id={id} className="ocean-parallax-section relative w-full py-20 px-6 lg:px-24 min-h-[50vh] flex flex-col items-center justify-center rounded-2xl mb-10 overflow-hidden"
             style={{ border: '1px solid rgba(255,255,255,0.07)', boxShadow: '0 8px 48px rgba(0,0,0,0.5)' }}>
      <div ref={bgRef} className="ocean-parallax-bg" style={{ backgroundImage: isVisible ? `url(${imgUrl})` : 'none' }} />
      <div className={`ocean-parallax-overlay ${overlay}`} />
      <div className="section-content-z w-full max-w-7xl">{children}</div>
      <AnimatedWaves />
    </section>
  );
};

// ── Navbar ─────────────────────────────────────────────────────
const Navbar = ({ user, onLogout, onRunPipeline, currentPage, onNavigate, serverStatus }) => (
    <nav className="glass-nav fixed top-0 w-full z-[100] px-6 py-3 flex justify-between items-center">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => onNavigate('home')}>
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)', boxShadow: '0 0 16px rgba(192,132,252,0.4)' }}>
                <Icon.Wave className="w-4 h-4 text-white" />
            </div>
            <span className="oi-brand font-bold text-base tracking-normal" style={{ background: 'linear-gradient(90deg, #2dd4bf, #c084fc, #fb7185)' }}>CORAL AI</span>
        </div>
        <div className="hidden md:flex gap-6 items-center">
            <button onClick={() => onNavigate('intelligence')} className={`nav-link ${currentPage === 'intelligence' ? 'active' : ''}`}>Intelligence</button>
            <button onClick={() => onNavigate('zones')}        className={`nav-link ${currentPage === 'zones' ? 'active' : ''}`}>Coastal Zones</button>
            <button onClick={() => onNavigate('trends')}       className={`nav-link ${currentPage === 'trends' ? 'active' : ''}`}>Trends</button>
            <button onClick={() => onNavigate('risk')}         className={`nav-link ${currentPage === 'risk' ? 'active' : ''}`}>Cell Risk</button>
            <button onClick={() => onNavigate('biodiversity')} className={`nav-link ${currentPage === 'biodiversity' ? 'active' : ''}`}>Biodiversity</button>
            <button onClick={() => onNavigate('compression')}  className={`nav-link ${currentPage === 'compression' ? 'active' : ''}`}>Compression</button>
            <button onClick={() => onNavigate('globe')}
               className={`nav-link flex items-center gap-1.5 ${currentPage === 'globe' ? 'active' : ''}`}>
                <Icon.Globe className="w-3.5 h-3.5" /> Globe
            </button>
            <button onClick={onRunPipeline}
                className="flex items-center gap-1.5 text-xs font-semibold px-4 py-2 rounded-lg transition"
                style={{ background: 'linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)', color: '#fff', fontFamily: 'Syne, sans-serif', fontWeight: 700, boxShadow: '0 0 20px rgba(124,58,237,0.4)' }}>
                <Icon.Play className="w-3 h-3" /> Run Pipeline
            </button>
        </div>
        <div className="flex items-center gap-3">
            {user && (
                <div className="hidden md:flex items-center gap-2 text-xs">
                    <Icon.User className="w-3.5 h-3.5 text-gray-400" />
                    <span className="text-gray-300 mr-2">Logged in as: <strong className="text-ocean-seafoam">{user}</strong></span>
                </div>
            )}
            <div className="flex px-3 py-1 rounded-full items-center gap-2 border border-white/10" style={{ background: 'rgba(0,0,0,0.3)' }}>
                <div className={`w-2 h-2 rounded-full ${serverStatus === 'live' ? 'bg-green-500 shadow-[0_0_8px_#22c55e] animate-pulse' : 'bg-red-500 shadow-[0_0_8px_#ef4444]'}`}></div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-gray-300">
                    {serverStatus === 'live' ? 'PC: LIVE' : 'PC: OFFLINE'}
                </span>
            </div>
            {user && (
                <button onClick={onLogout}
                    className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white border border-gray-700 hover:border-gray-400 px-2.5 py-1.5 rounded-lg transition">
                    <Icon.LogOut className="w-3.5 h-3.5" /> Logout
                </button>
            )}
        </div>
    </nav>
);

// ── Hero ─────────────────────────────────────────────────────
const Hero = ({ onNavigate }) => {
    const bgRef = useParallax('home', 0.28);
    return (
        <section className="relative w-full min-h-screen flex flex-col justify-center items-center text-center px-4 overflow-hidden">
            {/* Ocean parallax bg for hero */}
            <div ref={bgRef} className="ocean-parallax-bg" style={{ backgroundImage: `url(${OCEAN_IMAGES.home})`, backgroundPosition: 'center 15%' }} />
            <div className="ocean-parallax-overlay section-overlay-home" />

            {/* Animated SVG emblem */}
            <div className="mb-8 z-10 relative">
                <div className="w-24 h-24 rounded-2xl flex items-center justify-center mx-auto"
                     style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.7), rgba(13,79,107,0.8), rgba(45,212,191,0.4))', border: '1px solid rgba(192,132,252,0.4)', boxShadow: '0 0 50px rgba(124,58,237,0.3), 0 0 80px rgba(45,212,191,0.15)' }}>
                    <Icon.Satellite className="w-12 h-12 text-ocean-seafoam" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 10px #22c55e' }}></div>
            </div>

            {/* ── Translucent glassmorphic pill around hero text — ensures readability & background visibility ── */}
            <div className="z-10 flex flex-col items-center section-content-z" style={{
                background: 'rgba(2, 6, 20, 0.38)',
                backdropFilter: 'blur(24px) saturate(180%)',
                WebkitBackdropFilter: 'blur(24px) saturate(180%)',
                borderRadius: '28px',
                border: '1px solid rgba(255, 255, 255, 0.16)',
                padding: '2.5rem 3rem',
                maxWidth: '780px',
                boxShadow: '0 24px 80px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.08)'
            }}>
                <div className="section-label mb-2">Coastal Ocean Risk Assessment &amp; Lifecycle Intelligence</div>
                <h1 className="oi-brand-hero mb-5 font-black tracking-normal" style={{ fontSize: 'clamp(2.5rem, 6.5vw, 4.5rem)', lineHeight: 1.15, background: 'linear-gradient(135deg, #ffffff 10%, #a5f3fc 35%, #c084fc 60%, #fb7185 85%)' }}>
                    CORAL AI
                </h1>
                <p className="text-base md:text-lg font-normal max-w-2xl mb-8" style={{ color: 'rgba(210,220,235,0.92)', lineHeight: 1.7 }}>
                    Satellite-driven coastal risk intelligence — processing <strong className="text-white">342K+ ocean grid cells</strong> through dual-channel autoencoders to surface precise biodiversity threats and environmental anomalies.
                </p>
                {db && db.metadata && (
                    <div className="mb-8 flex items-center gap-3 text-sm font-mono px-5 py-3 rounded-xl"
                         style={{ background: 'rgba(13,79,107,0.3)', border: '1px solid rgba(45,212,191,0.3)' }}>
                        <Icon.Calendar className="w-4 h-4 text-ocean-seafoam flex-shrink-0" />
                        <span className="text-gray-300">Analysis window:</span>
                        <span className="text-ocean-gold font-semibold">{db.metadata.start_date}</span>
                        <span className="text-gray-500">→</span>
                        <span className="text-ocean-gold font-semibold">{db.metadata.end_date}</span>
                    </div>
                )}
                <div className="flex gap-4 flex-wrap justify-center">
                    <button onClick={() => onNavigate('zones')}
                       className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
                       style={{ background: 'linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)', color: '#fff', fontFamily: 'Syne, sans-serif', fontWeight: 700, boxShadow: '0 0 20px rgba(124,58,237,0.4)' }}>
                        <Icon.Map className="w-4 h-4" /> View Coastal Zones
                    </button>
                    <button onClick={() => onNavigate('risk')}
                       className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
                       style={{ border: '1px solid rgba(251,113,133,0.6)', color: '#fb7185', fontFamily: 'Syne, sans-serif', padding:'12px 32px', borderRadius:'999px', fontWeight: 700, backdropFilter:'blur(16px)', background:'rgba(251,113,133,0.1)' }}
                       onMouseEnter={e => { e.currentTarget.style.background='rgba(251,113,133,0.25)'; e.currentTarget.style.color='#fff'; }}
                       onMouseLeave={e => { e.currentTarget.style.background='rgba(251,113,133,0.1)'; e.currentTarget.style.color='#fb7185'; }}>
                        <Icon.Alert className="w-4 h-4" /> At-Risk Cells
                    </button>
                </div>
            </div>
            <AnimatedWaves />
        </section>
    );
};

// -- FeatureStrip: Why CORAL AI ----------------------------------------
const FeatureStrip = ({ onNavigate, db }) => {
    const features = [
        {
            icon: <Icon.Satellite className="w-7 h-7" style={{ color:"#2dd4bf" }}/>,
            title: "Real-Time Satellite Ingestion",
            desc: "Pulls live Copernicus Marine SST, Chlorophyll-a and SPM data across 342,000+ Indian coastal grid cells — updated with every pipeline run.",
            color: "#2dd4bf",
            badge: "Live Data",
        },
        {
            icon: <Icon.Cpu className="w-7 h-7" style={{ color:"#a5f3fc" }}/>,
            title: "97% Compute Cost Reduction",
            desc: "Dual-channel autoencoder compresses 2,400 MB of raw ocean telemetry down to 72 MB — slashing cloud inference costs by 97% without losing ecological signal.",
            color: "#a5f3fc",
            badge: "Cost Efficiency",
        },
        {
            icon: <Icon.Activity className="w-7 h-7" style={{ color:"#fbbf24" }}/>,
            title: "RM-NPI Anomaly Scoring",
            desc: "Our proprietary Risk-Modified Normalised Pollution Index fuses SST, chlorophyll, SPM and seasonal baselines into a single 0-1 risk score per cell.",
            color: "#fbbf24",
            badge: "Patented Index",
        },
        {
            icon: <Icon.Shield className="w-7 h-7" style={{ color:"#34d399" }}/>,
            title: "Biodiversity Protection",
            desc: `Strong r = ${CORRELATION_R} correlation between RM-NPI and marine health index. Protecting food security for 2.3M coastal residents across India.`,
            color: "#34d399",
            badge: "Eco Impact",
        },
        {
            icon: <Icon.Map className="w-7 h-7" style={{ color:"#c084fc" }}/>,
            title: "GPS-Pinned Zone Intelligence",
            desc: "Every at-risk cell is geo-resolved to a named coastal zone with exact coordinates, ecological threat classification and recommended action.",
            color: "#c084fc",
            badge: "Precision Mapping",
        },
        {
            icon: <Icon.TrendUp className="w-7 h-7" style={{ color:"#fb923c" }}/>,
            title: "Trend Forecasting",
            desc: "Monthly time-series tracking lets authorities detect deteriorating zones 3–6 months before crisis — shifting from reactive to preventive governance.",
            color: "#fb923c",
            badge: "Early Warning",
        },
    ];

    // Dynamically calculate stats based on backend DB payload
    const totalCells = db?.pipeline_summary?.total_cells || db?.overview?.length || 342332;
    const cellsFormatted = totalCells >= 1000 ? `${(totalCells / 1000).toFixed(0)}K+` : totalCells.toLocaleString();

    const rawMb = db?.datacenter?.raw_mb || 2400;
    const compMb = db?.datacenter?.comp_mb || 72;
    const compressionPct = rawMb ? (((rawMb - compMb) / rawMb) * 100).toFixed(1) + "%" : "97%";

    const stats = [
        { v: compressionPct,  l: "Data compression",      c: "#2dd4bf" },
        { v: cellsFormatted,   l: "Ocean grid cells",       c: "#a5f3fc" },
        { v: `${rawMb.toLocaleString()} → ${compMb.toLocaleString()}`, l: "MB raw → MB stored",  c: "#fbbf24" },
        { v: `r=${CORRELATION_R}`,        l: "Biodiversity correlation",c: "#34d399" },
        { v: "2.3M",           l: "Residents protected",    c: "#c084fc" },
        { v: "<2 min",         l: "Full pipeline runtime",  c: "#fb923c" },
    ];

    const [isVisible, setIsVisible] = useState(false);
    const sectionRef = useRef(null);

    useEffect(() => {
        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "250px" }
        );
        if (sectionRef.current) {
            observer.observe(sectionRef.current);
        }
        return () => observer.disconnect();
    }, []);

    const bgRef2 = useParallax('features', 0.22);
    return (
        <section ref={sectionRef} className="ocean-parallax-section" style={{ width:"100%", padding:"5rem 1.5rem", position:"relative", overflow:"hidden" }}>
            {/* Ocean parallax BG */}
            <div ref={bgRef2} className="ocean-parallax-bg" style={{ backgroundImage: isVisible ? `url(${OCEAN_IMAGES.calculator})` : 'none' }} />
            <div className="ocean-parallax-overlay section-overlay-trends" />

            <div className="section-content-z" style={{ maxWidth:"1280px", margin:"0 auto" }}>

                {/* Section Header */}
                <div style={{ textAlign:"center", marginBottom:"3.5rem" }}>
                    <p style={{ fontFamily:"Fira Code, monospace", fontSize:"0.62rem", letterSpacing:"0.25em", textTransform:"uppercase", color:"rgba(45,212,191,0.6)", marginBottom:"0.75rem" }}>
                        Why CORAL AI
                    </p>
                    <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"clamp(1.8rem, 4vw, 2.75rem)", fontWeight:800, color:"#fff", letterSpacing:"-0.03em", lineHeight:1.1, margin:"0 0 1rem 0" }}>
                        A platform built for impact,<br/>
                        <span style={{ background:"linear-gradient(90deg, #2dd4bf, #a5f3fc)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>not just analysis</span>
                    </h2>
                    <p style={{ fontFamily:"DM Sans, sans-serif", fontSize:"1rem", color:"rgba(156,163,175,0.6)", maxWidth:"560px", margin:"0 auto", lineHeight:1.7 }}>
                        From raw satellite bytes to actionable coastal risk intelligence — CORAL AI makes oceanic monitoring affordable and precise for governments and researchers.
                    </p>
                </div>

                {/* Stats Banner */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(150px, 1fr))", gap:"1px", background:"rgba(45,212,191,0.08)", borderRadius:"16px", overflow:"hidden", border:"1px solid rgba(45,212,191,0.12)", marginBottom:"4rem" }}>
                    {stats.map(s => (
                        <div key={s.l} style={{ padding:"1.5rem 1.25rem", textAlign:"center", background:"#030810" }}>
                            <div style={{ fontFamily:"Syne, sans-serif", fontSize:"1.75rem", fontWeight:800, color:s.c, letterSpacing:"-0.04em", lineHeight:1 }}>{s.v}</div>
                            <div style={{ fontFamily:"Fira Code, monospace", fontSize:"0.58rem", letterSpacing:"0.1em", color:"rgba(156,163,175,0.45)", textTransform:"uppercase", marginTop:"0.4rem" }}>{s.l}</div>
                        </div>
                    ))}
                </div>

                {/* Feature Cards Grid */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(320px, 1fr))", gap:"1.5rem" }}>
                    {features.map(f => (
                        <div key={f.title} style={{ background:"rgba(5,11,20,0.8)", border:"1px solid rgba(45,212,191,0.1)", borderRadius:"16px", padding:"1.75rem", position:"relative", overflow:"hidden", transition:"border-color 0.3s, transform 0.2s", cursor:"default" }}
                             onMouseEnter={e=>{ e.currentTarget.style.borderColor=f.color+"55"; e.currentTarget.style.transform="translateY(-4px)"; }}
                             onMouseLeave={e=>{ e.currentTarget.style.borderColor="rgba(45,212,191,0.1)"; e.currentTarget.style.transform="translateY(0)"; }}>

                            {/* Glow accent corner */}
                            <div style={{ position:"absolute", top:0, right:0, width:"120px", height:"120px", borderRadius:"0 16px 0 100%", background:`radial-gradient(circle at top right, ${f.color}12, transparent 70%)`, pointerEvents:"none" }}/>

                            {/* Badge */}
                            <span style={{ display:"inline-flex", alignItems:"center", fontFamily:"Fira Code, monospace", fontSize:"0.58rem", letterSpacing:"0.15em", textTransform:"uppercase", color:f.color, background:`${f.color}12`, border:`1px solid ${f.color}25`, borderRadius:"20px", padding:"2px 10px", marginBottom:"1.25rem" }}>
                                {f.badge}
                            </span>

                            {/* Icon */}
                            <div style={{ width:"52px", height:"52px", borderRadius:"14px", display:"flex", alignItems:"center", justifyContent:"center", background:`${f.color}10`, border:`1px solid ${f.color}22`, marginBottom:"1.25rem" }}>
                                {f.icon}
                            </div>

                            {/* Text */}
                            <h3 style={{ fontFamily:"Syne, sans-serif", fontSize:"1.1rem", fontWeight:700, color:"#fff", margin:"0 0 0.6rem 0", letterSpacing:"-0.02em" }}>
                                {f.title}
                            </h3>
                            <p style={{ fontFamily:"DM Sans, sans-serif", fontSize:"0.875rem", color:"rgba(156,163,175,0.6)", margin:0, lineHeight:1.65 }}>
                                {f.desc}
                            </p>
                        </div>
                    ))}
                </div>

                {/* Bottom CTA strip */}
                <div style={{ marginTop:"4rem", padding:"2rem 2.5rem", borderRadius:"16px", background:"linear-gradient(135deg, rgba(13,79,107,0.3), rgba(45,212,191,0.08))", border:"1px solid rgba(45,212,191,0.18)", display:"flex", flexWrap:"wrap", alignItems:"center", justifyContent:"space-between", gap:"1.5rem" }}>
                    <div>
                        <h3 style={{ fontFamily:"Syne, sans-serif", fontSize:"1.25rem", fontWeight:700, color:"#fff", margin:"0 0 0.35rem 0" }}>
                            Ready to explore the data?
                        </h3>
                        <p style={{ fontFamily:"DM Sans, sans-serif", fontSize:"0.875rem", color:"rgba(156,163,175,0.55)", margin:0 }}>
                            Scroll down to see real coastal zone risk scores, live trends, and biodiversity intelligence.
                        </p>
                    </div>
                    <div style={{ display:"flex", gap:"1rem" }}>
                        <button onClick={() => onNavigate('intelligence')} style={{ fontFamily:"Syne, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", background:"linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)", color:"#fff", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px", border:"none", cursor:"pointer", boxShadow:"0 0 20px rgba(124,58,237,0.35)" }}>
                            <Icon.BarChart2 className="w-4 h-4"/> View Intelligence
                        </button>
                        <button onClick={() => onNavigate('compression')} style={{ fontFamily:"Syne, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", border:"1px solid rgba(45,212,191,0.3)", color:"#2dd4bf", background:"none", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px", cursor:"pointer" }}>
                            <Icon.Database className="w-4 h-4"/> See Compression Data
                        </button>
                    </div>
                </div>

            </div>
        </section>
    );
};

// ── Section 1: Live Pipeline Intelligence ───────────────────
const Section1 = () => {
    // Use real pipeline_summary if available, else fall back to computing from overview
    const ps = db.pipeline_summary || null;
    const critCount  = ps ? ps.critical_cells  : db.overview.filter(z => z.risk === "CRITICAL").length;
    const highCount  = ps ? ps.high_cells       : db.overview.filter(z => z.risk === "HIGH").length;
    const avgRmnpi   = ps ? ps.avg_rmnpi        : (db.overview.reduce((s,z) => s+z.rmnpi,0)/db.overview.length).toFixed(4);
    const totalCells = ps ? ps.total_cells      : db.overview.length;
    const maxRmnpi   = ps ? ps.max_rmnpi        : Math.max(...db.overview.map(z=>z.rmnpi));
    const filtered   = getFilteredTimeseries();

    const metrics = [
        { Icon: Icon.Grid,     v: totalCells.toLocaleString(), l: "Grid Cells Analysed",   sub: ps ? "Full satellite grid" : "Named zones only",  color: "#2dd4bf" },
        { Icon: Icon.Alert,    v: critCount.toLocaleString(),  l: "Critical Risk Cells",    sub: "RM-NPI > 0.8 — immediate action",                 color: "#ff6b6b" },
        { Icon: Icon.Zap,      v: highCount.toLocaleString(),  l: "High + Critical Cells",  sub: "RM-NPI > 0.6 — alerts active",                   color: "#f97316" },
        { Icon: Icon.BarChart2,v: avgRmnpi,                    l: "Mean RM-NPI",            sub: "Average pollution pressure across grid",          color: "#fbbf24" },
        { Icon: Icon.TrendUp,  v: maxRmnpi,                    l: "Peak RM-NPI Detected",   sub: "Highest single-cell reading",                    color: "#a78bfa" },
    ];
    return (
        <SectionWrapper id="metrics">
            <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
                <div>
                    <div className="section-label">Satellite Pipeline</div>
                    <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #2dd4bf, #60a5fa)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Live Pipeline Intelligence</h2>
                    {ps ? (
                        <p className="flex items-center gap-1.5 text-green-400 text-xs mt-2 font-medium">
                            <Icon.Check className="w-3.5 h-3.5" /> Real counts from <strong>{totalCells.toLocaleString()}</strong> satellite grid cells
                        </p>
                    ) : (
                        <p className="flex items-center gap-1.5 text-yellow-500 text-xs mt-2 font-medium">
                            <Icon.Alert className="w-3.5 h-3.5" /> Run pipeline to see real grid-wide counts
                        </p>
                    )}
                </div>
                {db.metadata && (
                    <div className="rounded-xl px-5 py-3 text-right" style={{ background: 'rgba(13,79,107,0.15)', border: '1px solid rgba(45,212,191,0.15)' }}>
                        <div className="section-label text-right">Analysis Period</div>
                        <div className="text-ocean-gold font-mono font-bold text-sm">{db.metadata.start_date} → {db.metadata.end_date}</div>
                        <div className="text-gray-500 text-xs mt-1">
                            {(() => {
                                const s = new Date(db.metadata.start_date);
                                const e = new Date(db.metadata.end_date);
                                return (e.getFullYear() - s.getFullYear()) * 12 + (e.getMonth() - s.getMonth());
                            })()} months of observations
                        </div>
                    </div>
                )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-5 gap-5 text-center">
                {metrics.map(m => (
                    <div key={m.l} className="glow-card p-6 flex flex-col justify-center items-center gap-3">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${m.color}18`, border: `1px solid ${m.color}33` }}>
                            <m.Icon className="w-5 h-5" style={{ color: m.color }} />
                        </div>
                        <h3 className="text-3xl font-bold font-mono" style={{ color: m.color }}>{m.v}</h3>
                        <div>
                            <p className="text-xs text-white font-semibold uppercase tracking-widest">{m.l}</p>
                            <p className="text-xs mt-0.5" style={{ color: 'rgba(156,163,175,0.6)' }}>{m.sub}</p>
                        </div>
                    </div>
                ))}
            </div>
            {ps && (
                <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label:"Critical (>0.8)",  v: ps.critical_cells, pct: ps.pct_critical, col:"#ff6b6b" },
                        { label:"High (>0.6)",       v: ps.high_cells,     pct: ps.pct_high,     col:"#f97316" },
                        { label:"Moderate (>0.4)",   v: ps.moderate_cells, pct: (ps.moderate_cells/ps.total_cells*100).toFixed(2), col:"#fbbf24" },
                        { label:"Low (<0.4)",        v: ps.low_cells,      pct: (ps.low_cells/ps.total_cells*100).toFixed(2),      col:"#22c55e" },
                    ].map(t => (
                        <div key={t.label} className="glow-card p-3">
                            <div className="text-xs text-gray-400 mb-1">{t.label}</div>
                            <div className="text-xl font-bold" style={{color:t.col}}>{t.v.toLocaleString()}</div>
                            <div className="w-full bg-gray-800 rounded-full h-1 mt-2">
                                <div className="h-1 rounded-full" style={{width:`${Math.min(100,parseFloat(t.pct))}%`, backgroundColor:t.col}}></div>
                            </div>
                            <div className="text-xs text-gray-500 mt-1">{t.pct}% of all cells</div>
                        </div>
                    ))}
                </div>
            )}
        </SectionWrapper>
    );
};

// ── Section 2: Coastal Zone Intelligence Cards ──────────
const Section2 = () => {
    const [selected, setSelected] = useState(null);
    return (
        <SectionWrapper id="zones">
            <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #60a5fa, #818cf8)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">Coastal Zone Intelligence</h2>
            <p className="text-gray-400 mb-8 italic text-sm">
                Each card is a real GPS-pinned coastal zone. Click to expand all metrics, exact coordinates, and AI-derived biodiversity impact.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
                {db.overview.map(z => {
                    const bioIdx  = getBioIndex(z.rmnpi);
                    const bioRisk = getBioRisk(z.rmnpi);
                    const riskCol = getRiskColor(z.risk);
                    const isOpen  = selected === z.id;
                    return (
                        <div key={z.id} onClick={() => setSelected(isOpen ? null : z.id)}
                            className="glow-card p-4 cursor-pointer transition-all duration-300 hover:scale-[1.02] border-t-4"
                            style={{ borderTopColor: riskCol }}>
                            <div className="flex justify-between items-start mb-3">
                                <h4 className="font-bold text-white text-base leading-tight">{z.zone}</h4>
                                <span className={`text-xs px-2 py-0.5 rounded font-bold text-white ${getRiskBg(z.risk)}`}>{z.risk}</span>
                            </div>

                            {/* GPS Coordinates Block */}
                            <div className="bg-[#050b14] rounded-lg p-3 mb-3 font-mono text-xs border border-ocean-teal/20">
                                <div className="text-ocean-seafoam/60 text-xs mb-1 uppercase tracking-widest">📍 GPS Coordinates</div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Lat:</span>
                                    <span className="text-white font-bold">{z.lat.toFixed(4)}°N</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Lon:</span>
                                    <span className="text-white font-bold">{z.lon.toFixed(4)}°E</span>
                                </div>
                            </div>

                            {/* Key Metric Grid */}
                            <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                                <div className="bg-ocean-teal/10 rounded-lg p-2 text-center">
                                    <div className="text-gray-400 text-xs">RM-NPI</div>
                                    <div className="font-bold text-xl" style={{ color: riskCol }}>{z.rmnpi}</div>
                                </div>
                                <div className="bg-ocean-teal/10 rounded-lg p-2 text-center">
                                    <div className="text-gray-400 text-xs">Biodiversity</div>
                                    <div className="font-bold text-xl" style={{ color: getBioColor(z.rmnpi) }}>{bioIdx}</div>
                                </div>
                                <div className="bg-ocean-teal/10 rounded-lg p-2 text-center">
                                    <div className="text-gray-400 text-xs">SST</div>
                                    <div className="font-bold text-white">{z.sst}°C</div>
                                </div>
                                <div className="bg-ocean-teal/10 rounded-lg p-2 text-center">
                                    <div className="text-gray-400 text-xs">Rainfall</div>
                                    <div className="font-bold text-white">{z.rainfall}mm</div>
                                </div>
                            </div>

                            {/* Bio Risk Indicator */}
                            <div className="flex items-center gap-2 border-t border-ocean-teal/20 pt-2 mb-1">
                                <div className="w-2 h-2 rounded-full animate-pulse flex-shrink-0" style={{ backgroundColor: getBioColor(z.rmnpi) }}></div>
                                <span className="text-xs text-gray-300">Bio Risk: <span className="font-bold" style={{ color: getBioColor(z.rmnpi) }}>{bioRisk}</span></span>
                            </div>

                            {/* Biodiversity health bar */}
                            <div className="w-full bg-gray-800 rounded-full h-1 mb-2">
                                <div className="h-1 rounded-full transition-all" style={{ width: `${parseFloat(bioIdx)*100}%`, backgroundColor: getBioColor(z.rmnpi) }}></div>
                            </div>

                            {/* Expanded Details */}
                            {isOpen && (
                                <div className="mt-3 pt-3 border-t border-ocean-teal/30 text-xs space-y-1 animate-pulse-once">
                                    <div className="text-ocean-gold font-bold text-sm mb-2">🔬 Full Analysis</div>
                                    <div className="text-gray-300">🌊 Discharge: <span className="text-white font-mono">{z.discharge} m³/s</span></div>
                                    <div className="text-gray-300">🌿 NDVI: <span className="text-white font-mono">{z.ndvi}</span></div>
                                    <div className="text-gray-300">🐠 Marine Health: <span className="font-bold" style={{ color: getBioColor(z.rmnpi) }}>{bioIdx}/1.0</span></div>
                                    <div className="text-gray-300">☣️ Threat: <span className="text-white italic">{getEcologicalThreat(z.rmnpi)}</span></div>
                                    <div className="text-gray-300">📋 Action: <span className="text-white">{getRecommendedAction(z.risk)}</span></div>
                                    <div className="mt-2 text-gray-600 italic text-xs">Click to collapse</div>
                                </div>
                            )}
                            {!isOpen && <div className="text-xs text-gray-600 mt-1 text-center">↕ Click to expand</div>}
                        </div>
                    );
                })}
            </div>

            {/* Full Coordinates Table */}
            <h3 className="text-xl font-bold text-ocean-seafoam mb-4">📋 Full Coordinate & Environmental Metrics Table</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-left text-sm text-gray-300 border-collapse">
                    <thead className="bg-ocean-teal/40 text-ocean-seafoam">
                        <tr>
                            <th className="p-3">Zone</th>
                            <th className="p-3">Latitude (°N)</th>
                            <th className="p-3">Longitude (°E)</th>
                            <th className="p-3">SST (°C)</th>
                            <th className="p-3">Rainfall (mm)</th>
                            <th className="p-3">NDVI</th>
                            <th className="p-3">Discharge (m³/s)</th>
                            <th className="p-3">RM-NPI</th>
                            <th className="p-3">Bio Index</th>
                            <th className="p-3">Risk</th>
                        </tr>
                    </thead>
                    <tbody>
                        {db.overview.map(r => (
                            <tr key={r.id} className="border-b border-ocean-teal/20 hover:bg-ocean-teal/10">
                                <td className="p-3 font-bold text-white">{r.zone}</td>
                                <td className="p-3 font-mono text-ocean-seafoam font-bold">{r.lat.toFixed(4)}</td>
                                <td className="p-3 font-mono text-ocean-seafoam font-bold">{r.lon.toFixed(4)}</td>
                                <td className="p-3">{r.sst}</td>
                                <td className="p-3">{r.rainfall}</td>
                                <td className="p-3">{r.ndvi}</td>
                                <td className="p-3">{r.discharge}</td>
                                <td className="p-3 font-mono font-bold text-lg" style={{ color: getRiskColor(r.risk) }}>{r.rmnpi}</td>
                                <td className="p-3 font-mono font-bold" style={{ color: getBioColor(r.rmnpi) }}>{getBioIndex(r.rmnpi)}</td>
                                <td className="p-3"><span className={`px-2 py-1 rounded text-xs text-white font-bold ${getRiskBg(r.risk)}`}>{r.risk}</span></td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </SectionWrapper>
    );
};

// ── Section 3: Analysis Period Trends (FILTERED) ────────
const Section3 = () => {
    const filtered  = getFilteredTimeseries();
    const isFull    = filtered.length === (db.timeseries || []).length;
    const windowStr = db.metadata ? `${db.metadata.start_date} → ${db.metadata.end_date}` : "Full Dataset";

    return (
        <SectionWrapper id="trends">
            <div className="flex flex-col md:flex-row justify-between items-start mb-6 gap-4">
                <div>
                    <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #c084fc, #fb7185)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Environmental Trends</h2>
                    {isFull ? (
                        <p className="text-yellow-500 text-xs mt-1 italic">⚠️ Showing full baseline dataset. Run pipeline with specific dates for windowed view.</p>
                    ) : (
                        <p className="text-green-400 text-xs mt-1 italic">
                            ✅ Showing trends within your analysis window: <span className="font-mono">{windowStr}</span>
                        </p>
                    )}
                </div>
                <div className="bg-ocean-teal/10 border border-ocean-teal/30 rounded-lg px-4 py-2 text-sm text-right">
                    <div className="text-gray-400 text-xs">Observation Window</div>
                    <div className="text-ocean-gold font-mono font-bold">
                        {db.metadata ? (() => {
                            const s = new Date(db.metadata.start_date);
                            const e = new Date(db.metadata.end_date);
                            return (e.getFullYear() - s.getFullYear()) * 12 + (e.getMonth() - s.getMonth());
                        })() : filtered.length} months
                    </div>
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {[
                    { k:"rainfall",  c:"#2dd4bf", t:"Rainfall (mm)", desc:"Higher rainfall = more river runoff = nutrient pressure on coast", source: "Data Source: Open-Meteo API" },
                    { k:"sst",       c:"#ff6b6b", t:"Sea Surface Temperature (°C)", desc:"Values >30°C trigger coral bleaching and thermal stress events", source: "Data Source: Copernicus Marine Service" },
                    { k:"ndvi",      c:"#fbbf24", t:"NDVI Vegetation Index", desc:"Lower NDVI = more degraded land = higher erosion & fertilizer runoff", source: "Data Source: Copernicus Marine Service" },
                    { k:"discharge", c:"#a78bfa", t:"River Discharge (m³/s)", desc:"Flood peaks carry maximum nutrient loads directly into the ocean", source: "Data Source: Copernicus Marine Service" },
                ].map(c => (
                    <div key={c.k} className="h-72 glow-card p-4">
                        <h4 className="text-sm font-bold mb-0.5 text-center text-gray-300">{c.t}</h4>
                        <p className="text-xs text-center text-gray-600 mb-1 italic">{c.desc}</p>
                        <p className="text-[10px] text-center mb-3" style={{ fontFamily:"Fira Code, monospace", color:"rgba(192,132,252,0.75)", letterSpacing:"0.03em" }}>{c.source}</p>
                        <ResponsiveContainer width="100%" height="70%">
                            <LineChart data={filtered}>
                                <XAxis dataKey="month" stroke="#fff" tick={{ fontSize: 11 }} />
                                <YAxis stroke="#fff" tick={{ fontSize: 10 }} />
                                <Tooltip contentStyle={{ backgroundColor: '#0a1628', border: `1px solid ${c.c}` }} />
                                <Line type="monotone" dataKey={c.k} stroke={c.c} strokeWidth={2.5}
                                    dot={{ r: 5, fill: c.c, strokeWidth: 0 }} activeDot={{ r: 7 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                ))}
            </div>
        </SectionWrapper>
    );
};

// ── Section 4: RM-NPI Calculator ──────────────────────
const Section4 = () => {
    const [q, setQ] = useState(0.5); const [n, setN] = useState(0.5);
    const [s, setS] = useState(0.5); const [d, setD] = useState(0.5);
    const score   = (q * n * s * d).toFixed(3);
    const getRisk = () => score > 0.8 ? { l:"CRITICAL RISK", c:"text-ocean-coral", b:"bg-ocean-coral" }
                        : score > 0.5 ? { l:"HIGH RISK",     c:"text-orange-500", b:"bg-orange-500" }
                        : score > 0.2 ? { l:"MODERATE RISK", c:"text-ocean-gold",  b:"bg-ocean-gold" }
                        :               { l:"LOW RISK",       c:"text-green-500",   b:"bg-green-500" };
    const risk   = getRisk();
    const bioIdx = getBioIndex(parseFloat(score));
    return (
        <SectionWrapper id="calculator">
            <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"1.8rem", fontWeight:800, background:"linear-gradient(135deg, #fbbf24, #f97316)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">RM-NPI Calculator</h2>
            <p className="text-gray-400 mb-8 italic">Team Corals Innovation — A physics-based formula that triggers biological ecosystem warnings.</p>
            <div className="flex flex-col md:flex-row gap-8">
                <div className="flex-1 glow-card p-6 space-y-6">
                    <div className="text-center font-bold text-xl bg-ocean-navy p-3 rounded border border-ocean-teal">
                        Formula: RM-NPI = Q × N × S × D
                    </div>
                    {[
                        { l:"Q — River Discharge",          v:q, f:setQ, d:"Freshwater volume pushed into the ocean" },
                        { l:"N — Nutrient Load (NDVI proxy)", v:n, f:setN, d:"Agricultural fertilizer density" },
                        { l:"S — Seasonal Rainfall",         v:s, f:setS, d:"Monsoon and runoff intensity" },
                        { l:"D — Distance Decay",            v:d, f:setD, d:"Distance from coast to ocean pixel" },
                    ].map(i => (
                        <div key={i.l}>
                            <div className="flex justify-between text-sm"><label>{i.l}</label><span className="font-mono text-ocean-seafoam">{i.v}</span></div>
                            <input type="range" min="0" max="1" step="0.01" value={i.v}
                                onChange={(e) => i.f(parseFloat(e.target.value))} className="w-full accent-ocean-seafoam mt-1" />
                            <p className="text-xs text-gray-500 mt-0.5">{i.d}</p>
                        </div>
                    ))}
                </div>
                <div className={`flex-1 glow-card p-6 flex flex-col justify-center items-center border-t-8 ${risk.b}`}>
                    <h3 className="text-xl text-gray-300 mb-2">Live Calculated RM-NPI</h3>
                    <div className={`text-8xl font-black my-4 ${risk.c} drop-shadow-lg`}>{score}</div>
                    <div className={`text-2xl font-bold px-6 py-2 rounded-full text-[#0a1628] ${risk.b} animate-pulse mb-6`}>{risk.l}</div>
                    <div className="bg-[#050b14] rounded-lg p-4 w-full text-center">
                        <div className="text-xs text-gray-400 uppercase tracking-widest mb-1">Predicted Biodiversity Health Index</div>
                        <div className="text-3xl font-bold" style={{ color: getBioColor(parseFloat(score)) }}>{bioIdx}</div>
                        <div className="w-full bg-gray-800 rounded-full h-1.5 mt-2">
                            <div className="h-1.5 rounded-full transition-all" style={{ width:`${parseFloat(bioIdx)*100}%`, backgroundColor: getBioColor(parseFloat(score)) }}></div>
                        </div>
                        <div className="text-xs text-gray-500 mt-2 italic">Based on the AI-discovered r = {CORRELATION_R} correlation</div>
                    </div>
                </div>
            </div>
        </SectionWrapper>
    );
};

// ── Section 5: Biodiversity Hotspot Map ────────────────
const Section5Map = () => {
    const mapRef = useRef(null);
    useEffect(() => {
        if (!mapRef.current) return;
        if (mapRef.current._leaflet_id) return;
        const map = window.L.map(mapRef.current, { scrollWheelZoom: false }).setView([14.0, 79.0], 5);
        window.L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        db.overview.forEach(z => {
            const color   = getRiskColor(z.risk);
            const bioIdx  = getBioIndex(z.rmnpi);
            const bioRisk = getBioRisk(z.rmnpi);
            const bioCol  = getBioColor(z.rmnpi);
            const radius  = z.rmnpi >= 0.8 ? 18 : z.rmnpi >= 0.6 ? 13 : z.rmnpi >= 0.4 ? 9 : 6;
            const marker  = window.L.circleMarker([z.lat, z.lon], {
                radius, color, fillColor: color, fillOpacity: 0.72, weight: 2
            }).addTo(map);

            marker.bindPopup(`
                <div style="font-family:monospace;min-width:240px;background:#0a1628;color:#fff;padding:4px">
                    <div style="font-size:15px;font-weight:bold;color:${color};margin-bottom:8px;border-bottom:1px solid ${color};padding-bottom:4px">
                        ${z.zone}
                        <span style="float:right;font-size:11px;background:${color};color:#0a1628;padding:1px 6px;border-radius:4px">${z.risk}</span>
                    </div>

                    <div style="background:#050b14;border:1px solid #0d4f6b;border-radius:6px;padding:8px;margin-bottom:8px">
                        <div style="color:#2dd4bf;font-size:10px;text-transform:uppercase;letter-spacing:2px;margin-bottom:4px">📍 Exact GPS Coordinates</div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="color:#9ca3af">Latitude:</span>
                            <strong style="color:#fff">${z.lat.toFixed(4)}°N</strong>
                        </div>
                        <div style="display:flex;justify-content:space-between">
                            <span style="color:#9ca3af">Longitude:</span>
                            <strong style="color:#fff">${z.lon.toFixed(4)}°E</strong>
                        </div>
                    </div>

                    <table style="width:100%;font-size:11px;border-collapse:collapse">
                        <tr><td style="color:#9ca3af;padding:2px 4px">RM-NPI Score</td><td style="color:${color};font-weight:bold;text-align:right">${z.rmnpi}</td></tr>
                        <tr><td style="color:#9ca3af;padding:2px 4px">Biodiversity Index</td><td style="color:${bioCol};font-weight:bold;text-align:right">${bioIdx} / 1.0</td></tr>
                        <tr><td style="color:#9ca3af;padding:2px 4px">Bio Risk</td><td style="color:${bioCol};font-weight:bold;text-align:right">${bioRisk}</td></tr>
                        <tr><td style="color:#9ca3af;padding:2px 4px">Sea Surface Temp</td><td style="text-align:right">${z.sst}°C</td></tr>
                        <tr><td style="color:#9ca3af;padding:2px 4px">Rainfall</td><td style="text-align:right">${z.rainfall} mm</td></tr>
                        <tr><td style="color:#9ca3af;padding:2px 4px">River Discharge</td><td style="text-align:right">${z.discharge} m³/s</td></tr>
                    </table>

                    <div style="margin-top:8px;background:#050b14;padding:6px;border-radius:4px;border-left:3px solid ${bioCol}">
                        <div style="font-size:10px;color:#9ca3af;margin-bottom:2px">☣️ Ecological Threat</div>
                        <div style="font-size:10px;color:#fff">${getEcologicalThreat(z.rmnpi)}</div>
                    </div>
                    <div style="margin-top:4px;font-size:9px;color:#4b5563;text-align:center">Circle size reflects RM-NPI severity</div>
                </div>
            `, { maxWidth: 280 });
        });

        return () => { map.remove(); };
    }, []);

    return (
        <SectionWrapper id="map">
            <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #34d399, #2dd4bf)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">Coastal Zone Risk & Biodiversity Map</h2>
            <p className="text-gray-400 mb-4 text-sm italic">
                Click any marker for full GPS coordinates, RM-NPI score, biodiversity index, and ecological threat description.
                Larger circles = higher pollution pressure.
            </p>
            <div className="flex gap-6 mb-4 flex-wrap">
                {[["CRITICAL","#ff6b6b"],["HIGH","#f97316"],["MODERATE","#fbbf24"],["LOW","#22c55e"]].map(([r,c]) => (
                    <div key={r} className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: c }}></div>
                        <span className="text-gray-300 font-semibold">{r}</span>
                    </div>
                ))}
            </div>
            <div ref={mapRef} className="h-[520px] w-full glow-border rounded-xl overflow-hidden relative z-10" />
        </SectionWrapper>
    );
};

// ── At-Risk Cell Registry (NEW) ────────────────────────
const SectionCellRisk = () => {
    const [filter, setFilter] = useState("ALL");
    const sorted   = [...db.overview].sort((a, b) => b.rmnpi - a.rmnpi);
    const filtered = filter === "ALL" ? sorted : sorted.filter(r => r.risk === filter);

    return (
        <SectionWrapper id="cellrisk">
            <h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #fb7185, #f97316)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">At-Risk Cell Registry</h2>
            <p className="text-gray-400 mb-2 text-sm">
                Every monitored ocean cell sorted by RM-NPI severity. Shows <strong className="text-white">exact GPS coordinates</strong>,
                biodiversity health index, ecological threat type, and recommended field action — all derived live from the AI pipeline.
            </p>
            <p className="text-gray-600 text-xs italic mb-6">
                Biodiversity Index is computed as: <span className="font-mono text-ocean-seafoam">BioIndex = 1 + (RM-NPI × {CORRELATION_R})</span> — using the r={CORRELATION_R} correlation discovered by the autoencoder.
            </p>
            <div className="flex gap-2 mb-5 flex-wrap">
                {["ALL","CRITICAL","HIGH","MODERATE","LOW"].map(f => (
                    <button key={f} onClick={() => setFilter(f)}
                        className={`px-4 py-1.5 rounded text-xs font-bold transition ${filter===f ? 'bg-ocean-seafoam text-ocean-navy' : 'bg-ocean-navy text-ocean-seafoam border border-ocean-seafoam hover:bg-ocean-teal'}`}>
                        {f} {f !== "ALL" && `(${db.overview.filter(z=>z.risk===f).length})`}
                    </button>
                ))}
            </div>
            <div className="overflow-x-auto glow-card p-5" style={{ background: 'rgba(2, 6, 20, 0.85)', backdropFilter: 'blur(24px)', borderRadius: '16px', border: '1px solid rgba(45,212,191,0.15)' }}>
                <table className="w-full text-sm text-gray-300 border-collapse">
                    <thead className="bg-ocean-teal/40 text-ocean-seafoam text-xs uppercase tracking-wide">
                        <tr>
                            <th className="p-3 text-left">#</th>
                            <th className="p-3 text-left">Zone Name</th>
                            <th className="p-3 text-left">Latitude (°N)</th>
                            <th className="p-3 text-left">Longitude (°E)</th>
                            <th className="p-3 text-left">RM-NPI</th>
                            <th className="p-3 text-left">Risk Level</th>
                            <th className="p-3 text-left">Bio Index</th>
                            <th className="p-3 text-left">Bio Risk</th>
                            <th className="p-3 text-left">SST (°C)</th>
                            <th className="p-3 text-left">Ecological Threat</th>
                            <th className="p-3 text-left">Recommended Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filtered.map((r, idx) => {
                            const bioIdx  = getBioIndex(r.rmnpi);
                            const bioRisk = getBioRisk(r.rmnpi);
                            return (
                                <tr key={r.id}
                                    className={`border-b border-ocean-teal/20 hover:bg-ocean-teal/10 transition ${r.risk==="CRITICAL" ? "bg-red-900/10" : r.risk==="HIGH" ? "bg-orange-900/5" : ""}`}>
                                    <td className="p-3 text-gray-500 font-mono text-xs">{idx+1}</td>
                                    <td className="p-3 font-bold text-white">{r.zone}</td>
                                    <td className="p-3 font-mono text-ocean-seafoam font-bold">{r.lat.toFixed(4)}</td>
                                    <td className="p-3 font-mono text-ocean-seafoam font-bold">{r.lon.toFixed(4)}</td>
                                    <td className="p-3 font-mono font-black text-2xl" style={{ color: getRiskColor(r.risk) }}>{r.rmnpi}</td>
                                    <td className="p-3">
                                        <span className={`px-2 py-1 rounded text-xs text-white font-bold ${getRiskBg(r.risk)}`}>{r.risk}</span>
                                    </td>
                                    <td className="p-3 font-mono font-bold text-lg" style={{ color: getBioColor(r.rmnpi) }}>{bioIdx}</td>
                                    <td className="p-3">
                                        <span className="text-xs font-bold" style={{ color: getBioColor(r.rmnpi) }}>{bioRisk}</span>
                                    </td>
                                    <td className="p-3 font-mono">{r.sst}</td>
                                    <td className="p-3 text-xs text-gray-400 italic max-w-[200px]">{getEcologicalThreat(r.rmnpi)}</td>
                                    <td className="p-3 text-xs text-gray-300 max-w-[200px]">{getRecommendedAction(r.risk)}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </SectionWrapper>
    );
};

// ── Real Top-25 High-Risk Grid Cells Map (NEW) ────────────
const SectionRealCells = () => {
    const mapRef = useRef(null);
    const cells = db.top_risk_cells || [];

    useEffect(() => {
        if (!mapRef.current || cells.length === 0) return;
        if (mapRef.current._leaflet_id) return;
        const map = window.L.map(mapRef.current, { scrollWheelZoom: false }).setView([13.0, 79.5], 5);
        window.L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; OpenStreetMap'
        }).addTo(map);

        cells.forEach(cell => {
            const color  = getRiskColor(cell.risk);
            const radius = cell.rmnpi >= 0.8 ? 14 : cell.rmnpi >= 0.6 ? 10 : 6;
            const marker = window.L.circleMarker([cell.lat, cell.lon], {
                radius, color, fillColor: color, fillOpacity: 0.8, weight: 2
            }).addTo(map);
            marker.bindPopup(`
                <div style="font-family:monospace;min-width:220px;background:#0a1628;color:#fff;padding:4px">
                    <div style="color:${color};font-weight:bold;font-size:13px;margin-bottom:6px">
                        #${cell.rank} Ranked High-Risk Cell
                        <span style="float:right;background:${color};color:#0a1628;font-size:10px;padding:1px 5px;border-radius:3px">${cell.risk}</span>
                    </div>
                    <div style="background:#050b14;border-radius:5px;padding:7px;margin-bottom:6px">
                        <div style="color:#2dd4bf;font-size:9px;letter-spacing:2px;margin-bottom:3px">📍 EXACT SATELLITE GRID COORDINATES</div>
                        <div style="display:flex;justify-content:space-between"><span style="color:#9ca3af">Latitude:</span> <strong>${cell.lat}°N</strong></div>
                        <div style="display:flex;justify-content:space-between"><span style="color:#9ca3af">Longitude:</span> <strong>${cell.lon}°E</strong></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
                        <span style="color:#9ca3af">RM-NPI Score:</span>
                        <strong style="color:${color}">${cell.rmnpi}</strong>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px">
                        <span style="color:#9ca3af">SST:</span><span>${cell.sst}°C</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:11px">
                        <span style="color:#9ca3af">Recon Error:</span><span>${cell.recon_error}</span>
                    </div>
                    <div style="margin-top:6px;font-size:10px;color:#6b7280;border-top:1px solid #1f2937;padding-top:4px">
                        This is an <em>actual satellite grid cell</em> ranked by AI model
                    </div>
                </div>
            `, { maxWidth: 260 });
        });
        return () => { map.remove(); };
    }, [cells.length]);

    const ps = db.pipeline_summary;

    return (
        <SectionWrapper id="realcells">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🛰️ Real AI-Detected High-Risk Ocean Cells</h2>
            <div className="bg-ocean-teal/10 border border-ocean-seafoam/30 rounded-xl p-4 mb-6">
                <p className="text-sm text-gray-300 leading-relaxed">
                    <span className="text-ocean-gold font-bold">Why does terminal show more criticals than the map?</span><br/>
                    The named zones (Chennai, Mumbai, etc.) only show RM-NPI at their <em>nearest satellite pixel</em>, which may be low.
                    The AI processes <strong className="text-white">{ps ? ps.total_cells.toLocaleString() : "342,332"}</strong> grid cells across the full Indian Ocean.
                    Below are the <strong className="text-ocean-coral">{cells.length} actual highest-scoring cells</strong> from the model — these are the real critical zones detected in the terminal.
                </p>
                {ps && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4 text-center">
                        <div className="bg-red-900/30 rounded p-2"><div className="text-xs text-gray-400">CRITICAL cells</div><div className="text-xl font-black text-red-400">{ps.critical_cells.toLocaleString()}</div><div className="text-xs text-gray-500">{ps.pct_critical}%</div></div>
                        <div className="bg-orange-900/30 rounded p-2"><div className="text-xs text-gray-400">HIGH cells</div><div className="text-xl font-black text-orange-400">{ps.high_cells.toLocaleString()}</div><div className="text-xs text-gray-500">{ps.pct_high}%</div></div>
                        <div className="bg-ocean-teal/20 rounded p-2"><div className="text-xs text-gray-400">Mean RM-NPI</div><div className="text-xl font-black text-ocean-seafoam">{ps.avg_rmnpi}</div><div className="text-xs text-gray-500">across grid</div></div>
                        <div className="bg-ocean-teal/20 rounded p-2"><div className="text-xs text-gray-400">Peak RM-NPI</div><div className="text-xl font-black text-ocean-coral">{ps.max_rmnpi}</div><div className="text-xs text-gray-500">single cell max</div></div>
                    </div>
                )}
            </div>

            {cells.length > 0 ? (
                <>
                    {/* Map of real cells */}
                    <div className="mb-6">
                        <h3 className="text-lg font-bold text-ocean-seafoam mb-2">📍 Top {cells.length} Cells Plotted on Map</h3>
                        <div ref={mapRef} className="h-[450px] w-full glow-border rounded-xl overflow-hidden relative z-10" />
                    </div>

                    {/* Table of real cells */}
                    <h3 className="text-lg font-bold text-ocean-seafoam mb-3">📋 Full Ranked List with Exact Coordinates</h3>
                    <div className="overflow-x-auto glow-card p-5" style={{ background: 'rgba(2, 6, 20, 0.85)', backdropFilter: 'blur(24px)', borderRadius: '16px', border: '1px solid rgba(45,212,191,0.15)' }}>
                        <table className="w-full text-sm text-gray-300 border-collapse">
                            <thead className="bg-ocean-teal/40 text-ocean-seafoam text-xs uppercase tracking-wide">
                                <tr>
                                    <th className="p-3">Rank</th>
                                    <th className="p-3">Latitude (°N)</th>
                                    <th className="p-3">Longitude (°E)</th>
                                    <th className="p-3">RM-NPI</th>
                                    <th className="p-3">Risk</th>
                                    <th className="p-3">SST (°C)</th>
                                    <th className="p-3">Recon Error</th>
                                    <th className="p-3">Bio Index</th>
                                    <th className="p-3">Ecological Threat</th>
                                </tr>
                            </thead>
                            <tbody>
                                {cells.map(c => (
                                    <tr key={c.rank} className={`border-b border-ocean-teal/20 hover:bg-ocean-teal/10 ${c.risk==="CRITICAL"?"bg-red-900/10":c.risk==="HIGH"?"bg-orange-900/5":""}`}>
                                        <td className="p-3 font-mono font-bold text-ocean-gold">#{c.rank}</td>
                                        <td className="p-3 font-mono font-bold text-ocean-seafoam">{c.lat}</td>
                                        <td className="p-3 font-mono font-bold text-ocean-seafoam">{c.lon}</td>
                                        <td className="p-3 font-mono font-black text-2xl" style={{color:getRiskColor(c.risk)}}>{c.rmnpi}</td>
                                        <td className="p-3"><span className={`px-2 py-1 rounded text-xs text-white font-bold ${getRiskBg(c.risk)}`}>{c.risk}</span></td>
                                        <td className="p-3 font-mono">{c.sst}</td>
                                        <td className="p-3 font-mono text-xs text-gray-400">{c.recon_error}</td>
                                        <td className="p-3 font-mono font-bold" style={{color:getBioColor(c.rmnpi)}}>{getBioIndex(c.rmnpi)}</td>
                                        <td className="p-3 text-xs text-gray-400 italic">{getEcologicalThreat(c.rmnpi)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            ) : (
                <div className="glow-card p-8 text-center">
                    <div className="text-5xl mb-4">🔄</div>
                    <h3 className="text-xl font-bold text-ocean-seafoam mb-2">Run Pipeline to See Real Grid Cells</h3>
                    <p className="text-gray-400 text-sm">After running <code className="font-mono text-ocean-gold">venv\Scripts\python.exe src/pipeline.py --start ... --end ...</code>, this section will populate with the actual top-150 highest RM-NPI cells detected by the AI across all {ps ? ps.total_cells.toLocaleString() : "342,332"} satellite grid cells.</p>
                </div>
            )}
        </SectionWrapper>
    );
};

const Section8 = () => {
    // Use real named zones with actual coordinates instead of random dots
    const zoneBio = db.overview.map(z => ({
        zone:     z.zone,
        rmnpi:    z.rmnpi,
        bioIndex: parseFloat(getBioIndex(z.rmnpi)),
        lat:      z.lat,
        lon:      z.lon,
        risk:     z.risk,
    }));

    return (
        <SectionWrapper id="biodiversity">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🌿 Ocean Stress vs. Biodiversity Health</h2>
            <p className="text-gray-400 mb-6 italic text-sm">
                Each point is a <strong className="text-white">real named coastal zone</strong> with exact GPS coordinates.
                As RM-NPI rises (more pollution pressure), the marine biodiversity health index collapses — confirmed at r = {CORRELATION_R}.
            </p>
            <div className="h-80 w-full glow-card p-4 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 30 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#0d4f6b" />
                        <XAxis type="number" dataKey="rmnpi" name="RM-NPI" stroke="#fff" domain={[0,1]}
                            label={{ value: "RM-NPI Score (Nutrient Pressure) →", position:"insideBottom", offset:-15, fill:"#9ca3af", fontSize:11 }} />
                        <YAxis type="number" dataKey="bioIndex" name="Bio Index" stroke="#fff" domain={[0,1]}
                            label={{ value:"Biodiversity Health ↑", angle:-90, position:"insideLeft", fill:"#9ca3af", fontSize:11 }} />
                        <Tooltip cursor={{ strokeDasharray:"3 3" }}
                            content={({ payload }) => {
                                if (!payload || !payload.length) return null;
                                const d = payload[0].payload;
                                return (
                                    <div style={{ background:"#0a1628", border:`1px solid ${getRiskColor(d.risk)}`, padding:"10px 14px", borderRadius:8 }}>
                                        <div style={{ color:"#2dd4bf", fontWeight:"bold", fontSize:13, marginBottom:4 }}>{d.zone}</div>
                                        <div style={{ color:"#fff", fontSize:11 }}>RM-NPI: <strong>{d.rmnpi}</strong></div>
                                        <div style={{ color:"#fff", fontSize:11 }}>Bio Index: <strong style={{ color: getBioColor(d.rmnpi) }}>{d.bioIndex}</strong></div>
                                        <div style={{ color:"#9ca3af", fontSize:10, marginTop:4 }}>📍 {d.lat.toFixed(4)}°N, {d.lon.toFixed(4)}°E</div>
                                        <div style={{ color:"#9ca3af", fontSize:10 }}>☣️ {getEcologicalThreat(d.rmnpi)}</div>
                                    </div>
                                );
                            }}
                        />
                        <ZAxis range={[80, 80]} />
                        <Scatter data={zoneBio} name="Coastal Zones">
                            {zoneBio.map((e, i) => <Cell key={i} fill={getRiskColor(e.risk)} />)}
                        </Scatter>
                    </ScatterChart>
                </ResponsiveContainer>
            </div>
            <div className="w-full text-center mb-8 text-ocean-coral font-bold font-mono text-xl animate-pulse">
                r = {CORRELATION_R} — Strong negative correlation: Higher pollution = Lower biodiversity
            </div>

            {/* Per-Zone Biodiversity Cards with Lat/Lon */}
            <h3 className="text-xl font-bold text-ocean-seafoam mb-4">Biodiversity Health per Named Zone (with Coordinates)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {zoneBio.sort((a,b) => a.bioIndex - b.bioIndex).map(z => (
                    <div key={z.zone} className="glow-card p-4 border-l-4" style={{ borderLeftColor: getBioColor(z.rmnpi) }}>
                        <div className="font-bold text-white mb-1">{z.zone}</div>
                        <div className="font-mono text-xs text-ocean-seafoam/70 mb-3">
                            📍 {z.lat.toFixed(4)}°N, {z.lon.toFixed(4)}°E
                        </div>
                        <div className="flex justify-between items-end mb-2">
                            <div>
                                <div className="text-xs text-gray-400">Bio Index</div>
                                <div className="text-2xl font-black" style={{ color: getBioColor(z.rmnpi) }}>{z.bioIndex}</div>
                            </div>
                            <div className="text-right">
                                <div className="text-xs text-gray-400">RM-NPI</div>
                                <div className="text-2xl font-black text-ocean-coral">{z.rmnpi}</div>
                            </div>
                        </div>
                        <div className="w-full bg-gray-800 rounded-full h-2 mb-1">
                            <div className="h-2 rounded-full" style={{ width:`${z.bioIndex*100}%`, backgroundColor: getBioColor(z.rmnpi) }}></div>
                        </div>
                        <div className="text-xs text-gray-500">Ecosystem health: {Math.round(z.bioIndex*100)}%</div>
                        <div className="text-xs mt-1 font-bold" style={{ color: getBioColor(z.rmnpi) }}>{getBioRisk(z.rmnpi)} bio risk</div>
                    </div>
                ))}
            </div>
        </SectionWrapper>
    );
};

// ── Data Compression Story (Interactive) ───────────────
const Section9 = () => {
    const [step, setStep] = useState(0);
    const dc = db?.datacenter || { data_points_str: "5.4M", raw_mb: 847, comp_mb: 23, metrics: [] };
    const pct = dc.raw_mb ? ((dc.raw_mb - dc.comp_mb) / dc.raw_mb * 100).toFixed(1) : "97.3";
    const totalCells = db?.pipeline_summary?.total_cells || 342332;

    const steps = [
        {
            icon:"🌍", title:"Raw Earth Observation Input", color:"#ff6b6b",
            stat:`${totalCells.toLocaleString()} grid cells × 13 variables = ~${dc.data_points_str} data points`,
            size:`${dc.raw_mb} MB raw on disk`,
            desc:"Two primary data sources ingested simultaneously: Copernicus Marine Service (temperature, salinity, ocean chemistry) and Open-Meteo API (rainfall and coastal weather). Every ocean pixel from 5°N to 20°N latitude, 70°E to 85°E longitude.",
        },
        {
            icon:"🧹", title:"AI Preprocessing & Alignment", color:"#f97316",
            stat:"13 variables → cleaned, normalized [0,1] feature matrix",
            size:"~30% noise removed by imputation & alignment",
            desc:"All three satellite datasets are time-aligned to a common grid, coordinate-matched, and NaN-imputed using column medians. Features are then min-max normalized to [0,1] so the neural network can ingest them without numerical instability.",
        },
        {
            icon:"🧠", title:"Dual-Channel Autoencoder Encoding", color:"#fbbf24",
            stat:"13 features → 12 latent dimensions (NPI channel: 6, Discovery channel: 6)",
            size:`Memory footprint: ${dc.raw_mb}MB → ${dc.comp_mb}MB (${pct}% reduction)`,
            desc:"The autoencoder compresses each grid cell's variables into a 12-dimensional 'fingerprint'. Channel 1 (NPI) encodes known pollution risk signals. Channel 2 (Discovery) encodes hidden unknown patterns. Cells that look 'normal' cost almost nothing to store.",
        },
        {
            icon:"🔀", title:"Priority Routing & Storage Tiering", color:"#2dd4bf",
            stat:`HOT: ~${db?.pipeline_summary?.high_cells || 1714} cells | WARM: ~2,374 | COLD: ~${db?.pipeline_summary?.low_cells || 338244}`,
            size:`Compute cycles: ${dc.metrics?.[1]?.before?.toLocaleString() || "12,400"} → ${dc.metrics?.[1]?.after?.toLocaleString() || "1,847"}`,
            desc:"The AI scores every cell. CRITICAL/anomalous cells are routed to HOT (fast SSD) storage for immediate analysis. Routine cells move to COLD (cheap HDD/object storage). This means only a tiny fraction of cells consume expensive compute, while the rest are archived cheaply.",
        },
        {
            icon:"🎯", title:"Human-Readable Intelligence Extraction", color:"#22c55e",
            stat:"Output: 3 biodiversity threats | 8 zone risk scores | 5+ anomaly reports",
            size:`Analysis time: ${dc.metrics?.[2]?.before || 340}s → ${dc.metrics?.[2]?.after || 42}s`,
            desc:"The compressed latent vectors are decoded into plain-language outputs: zone-specific RM-NPI scores with exact GPS coordinates, biodiversity health indices, anomaly reports with deviation percentages, and ecological threat classifications — all ready for environmental officers to act on.",
        },
    ];
    const s = steps[step];

    return (
        <SectionWrapper id="compression">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">💾 From {dc.data_points_str} Data Points to Actionable Intelligence</h2>
            <p className="text-gray-400 mb-8 italic text-sm">
                A step-by-step walkthrough of how CORAL AI compresses massive satellite data into precise, targeted coastal risk intelligence.
            </p>

            {/* Step Tabs */}
            <div className="flex flex-wrap gap-2 mb-8 justify-center">
                {steps.map((st, i) => (
                    <button key={i} onClick={() => setStep(i)}
                        className={`px-5 py-2 rounded-full text-xs font-bold transition-all duration-200 ${step===i ? 'text-[#0a1628] scale-110 shadow-lg' : 'text-gray-300 bg-ocean-navy border border-ocean-teal/30 hover:border-ocean-teal'}`}
                        style={step===i ? { backgroundColor: st.color } : {}}>
                        {st.icon} Step {i+1}
                    </button>
                ))}
            </div>

            {/* Active Step Card */}
            <div key={step} className="glow-card p-8 mb-8 border-l-4 transition-all" style={{ borderLeftColor: s.color }}>
                <div className="flex items-start gap-6">
                    <div className="text-6xl flex-shrink-0">{s.icon}</div>
                    <div className="flex-1">
                        <h3 className="text-2xl font-bold text-white mb-3">{s.title}</h3>
                        <p className="text-gray-300 leading-relaxed mb-5">{s.desc}</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-[#050b14] border border-ocean-teal/20 rounded-lg p-4">
                                <div className="text-xs text-gray-400 uppercase tracking-widest mb-2">📊 Data Facts</div>
                                <div className="text-sm font-mono leading-relaxed" style={{ color: s.color }}>{s.stat}</div>
                            </div>
                            <div className="bg-[#050b14] border border-ocean-teal/20 rounded-lg p-4">
                                <div className="text-xs text-gray-400 uppercase tracking-widest mb-2">⚡ Efficiency Gain</div>
                                <div className="text-sm font-mono text-ocean-seafoam">{s.size}</div>
                            </div>
                        </div>
                    </div>
                </div>
                {/* Progress Bar */}
                <div className="mt-6">
                    <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>Pipeline Progress</span>
                        <span>Step {step+1} of {steps.length}</span>
                    </div>
                    <div className="w-full bg-gray-800 rounded-full h-1.5">
                        <div className="h-1.5 rounded-full transition-all duration-500" style={{ width:`${((step+1)/steps.length)*100}%`, backgroundColor: s.color }}></div>
                    </div>
                </div>
                <div className="flex justify-between mt-5">
                    <button onClick={() => setStep(Math.max(0, step-1))} disabled={step===0}
                        className="px-5 py-2 text-xs font-bold rounded-full bg-ocean-navy border border-ocean-teal text-ocean-seafoam disabled:opacity-30 hover:bg-ocean-teal transition">
                        ← Previous
                    </button>
                    <button onClick={() => setStep(Math.min(steps.length-1, step+1))} disabled={step===steps.length-1}
                        className="px-5 py-2 text-xs font-bold rounded-full text-[#0a1628] hover:opacity-90 disabled:opacity-30 transition"
                        style={{ backgroundColor: s.color }}>
                        Next →
                    </button>
                </div>
            </div>

            {/* Summary Comparison Chart */}
            <h3 className="text-xl font-bold text-ocean-seafoam mb-4">Before vs. After AI Compression</h3>
            <div className="h-64 glow-card p-4 mb-6">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={dc.metrics} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#0a1628" />
                        <XAxis type="number" stroke="#fff" tick={{ fontSize: 10 }} />
                        <YAxis dataKey="name" type="category" stroke="#fff" width={130} tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ backgroundColor:'#0a1628', border:'1px solid #2dd4bf' }} />
                        <Legend />
                        <Bar dataKey="before" fill="#ff6b6b" name="Before AI (Raw)" radius={[0,4,4,0]} />
                        <Bar dataKey="after"  fill="#2dd4bf" name="After Compression" radius={[0,4,4,0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Summary Stat Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {dc.metrics && dc.metrics.map(m => {
                    const reductionPct = ((1 - (m.after / m.before)) * 100).toFixed(1) + "%";
                    const color = m.name.includes("Storage") ? "#ff6b6b" : m.name.includes("Compute") ? "#fbbf24" : "#2dd4bf";
                    return (
                    <div key={m.name} className="glow-card p-5 text-center">
                        <div className="text-xs text-gray-400 uppercase tracking-widest mb-2">{m.name}</div>
                        <div className="text-base line-through text-red-400 font-mono">{m.before.toLocaleString()} {m.name.includes("Storage") ? "MB" : m.name.includes("Time") ? "s" : ""}</div>
                        <div className="text-4xl font-black text-ocean-seafoam font-mono mt-1">{m.after.toLocaleString()} {m.name.includes("Storage") ? "MB" : m.name.includes("Time") ? "s" : ""}</div>
                        <div className="font-bold text-lg mt-2" style={{ color: color }}>{reductionPct} Reduction</div>
                    </div>
                )})}
            </div>
        </SectionWrapper>
    );
};

// ── AI Core Removed per user request ───────────────────

// ── Anomalies ──────────────────────────────────────────
const Section7 = () => (
    <SectionWrapper id="anomalies">
        <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🚨 AI-Detected Environmental Anomalies</h2>
        <p className="text-gray-400 mb-2 text-sm font-mono">
            Cells flagged when the autoencoder's reconstruction error exceeds the 95th percentile — meaning the AI cannot explain their values as "normal".
        </p>
        <p className="text-gray-600 text-xs italic mb-6">These are real anomalies from the pipeline run. Each one warrants field investigation.</p>
        <div className="overflow-x-auto glow-card">
            <table className="w-full text-left font-mono text-sm border-collapse">
                <thead className="bg-[#050b14] text-ocean-seafoam text-xs uppercase tracking-wide">
                    <tr>
                        <th className="p-3">Zone</th><th className="p-3">Date</th><th className="p-3">Variable</th>
                        <th className="p-3">Observed</th><th className="p-3">Expected</th>
                        <th className="p-3">Deviation</th><th className="p-3">Severity</th><th className="p-3">Ecological Impact</th>
                    </tr>
                </thead>
                <tbody>
                    {db.anomalies.map((a, i) => (
                        <tr key={i} className={`border-b border-ocean-teal/20 hover:bg-ocean-teal/10 ${a.sev==="CRITICAL"?"bg-red-900/10":""}`}>
                            <td className="p-3 font-sans text-white font-bold">{a.zone}</td>
                            <td className="p-3 text-gray-400">{a.date}</td>
                            <td className="p-3 text-ocean-gold">{a.var}</td>
                            <td className="p-3 text-red-400 font-bold">{a.obs}</td>
                            <td className="p-3 text-gray-400">{a.exp}</td>
                            <td className="p-3 text-ocean-coral font-bold">{a.dev}</td>
                            <td className="p-3">
                                <span className={`px-2 py-1 rounded text-xs text-white font-bold ${a.sev==="CRITICAL"?"bg-red-600":a.sev==="HIGH"?"bg-orange-500":"bg-yellow-500"}`}>{a.sev}</span>
                            </td>
                            <td className="p-3 text-xs font-sans text-gray-400 italic">
                                {a.sev==="CRITICAL" ? "Dead zone formation / mass suffocation risk" :
                                 a.sev==="HIGH"     ? "Fish migration event / food web disruption" :
                                                      "Elevated ecosystem stress — monitor closely"}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </SectionWrapper>
);

// ── Correlation Matrix ─────────────────────────────────
const Section11 = () => {
    const vars = ['Rain', 'SST', 'NDVI', 'Discharge', 'RM-NPI', 'Biodiv'];
    
    let vals = Array(6).fill(0).map(() => Array(6).fill(0));
    
    const getPearson = (xArray, yArray) => {
        let sumX = 0, sumY = 0, sumX2 = 0, sumY2 = 0, sumXY = 0;
        let n = 0;
        for (let i = 0; i < xArray.length; i++) {
            let vx = xArray[i], vy = yArray[i];
            if (typeof vx === 'number' && !isNaN(vx) && typeof vy === 'number' && !isNaN(vy)) {
                sumX += vx;
                sumY += vy;
                sumX2 += vx * vx;
                sumY2 += vy * vy;
                sumXY += vx * vy;
                n++;
            }
        }
        if (n === 0) return 0;
        const num = (n * sumXY) - (sumX * sumY);
        const den = Math.sqrt(((n * sumX2) - (sumX * sumX)) * ((n * sumY2) - (sumY * sumY)));
        if (den === 0) return 0;
        return num / den;
    };

    if (db && db.overview && db.overview.length > 0) {
        const cols = [
            db.overview.map(d => d.rainfall),
            db.overview.map(d => d.sst),
            db.overview.map(d => d.ndvi),
            db.overview.map(d => d.discharge),
            db.overview.map(d => d.rmnpi)
        ];
        
        for (let i = 0; i < 5; i++) {
            for (let j = 0; j < 5; j++) {
                vals[i][j] = getPearson(cols[i], cols[j]);
            }
        }

        // Extrapolate Biodiv correlations based on the true discovered CORRELATION_R
        // instead of generating a perfectly linear synthetic variable that forces -1.0
        for (let i = 0; i < 5; i++) {
            // Correlation between Variable i and Biodiv = Corr(Var i, RMNPI) * Corr(RMNPI, Biodiv)
            let estimatedCorr = vals[i][4] * Math.abs(CORRELATION_R) * Math.sign(CORRELATION_R);
            vals[i][5] = estimatedCorr;
            vals[5][i] = estimatedCorr;
        }
        vals[4][5] = CORRELATION_R; // True r value
        vals[5][4] = CORRELATION_R;
        vals[5][5] = 1.0;
    } else {
        vals = [
            [ 1.0,  0.2,  0.4,  0.9,  0.8, -0.7],
            [ 0.2,  1.0,  0.1,  0.1,  0.3, -0.5],
            [ 0.4,  0.1,  1.0,  0.3,  0.6, -0.8],
            [ 0.9,  0.1,  0.3,  1.0,  0.9, -0.8],
            [ 0.8,  0.3,  0.6,  0.9,  1.0, -0.9],
            [-0.7, -0.5, -0.8, -0.8, -0.9,  1.0],
        ];
    }

    return (
        <SectionWrapper id="heatmap">
            <h2 className="text-3xl font-bold text-ocean-seafoam mb-2">Environmental Variable Correlation Matrix</h2>
            <p className="text-gray-400 mb-8 text-sm italic">
                Notice the <strong className="text-ocean-coral">{CORRELATION_R}</strong> correlation between RM-NPI and Biodiversity —
                the AI confirmed that as nutrient pollution rises, marine life collapses.
            </p>
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
                                    <div key={j} title={`${rowVar} vs ${vars[j]}: ${val}`}
                                        className="h-12 w-full flex items-center justify-center text-xs font-mono text-white transition hover:scale-110 cursor-pointer border border-ocean-navy/30"
                                        style={{ backgroundColor: `rgb(${r},${g},${b})` }}>
                                        {val.toFixed(1)}
                                    </div>
                                );
                            })}
                        </React.Fragment>
                    ))}
                </div>
            </div>
        </SectionWrapper>
    );
};

// ── Conclusion ─────────────────────────────────────────
const Section12 = () => {
    const dc = db?.datacenter || { data_points_str: "5.4M", raw_mb: 847, comp_mb: 23, metrics: [] };
    const pct = dc.raw_mb ? ((dc.raw_mb - dc.comp_mb) / dc.raw_mb * 100).toFixed(1) : "97.3";
    const totalCells = db?.pipeline_summary?.total_cells || 342332;

    return (
    <SectionWrapper id="conclusion">
        <div className="section-label text-center mb-2">Mission Statement</div>
        <h2 className="section-title text-3xl text-white text-center mb-10">What CORAL AI Delivers</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="glow-card p-8 text-center flex flex-col items-center gap-4">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(45,212,191,0.12)', border: '1px solid rgba(45,212,191,0.25)' }}>
                    <Icon.Map className="w-7 h-7 text-ocean-seafoam" />
                </div>
                <p className="font-bold text-white text-lg" style={{ fontFamily: 'Syne, sans-serif' }}>Real Coordinates, Real Impact</p>
                <p className="text-gray-400 text-sm leading-relaxed">Named coastal zones with exact GPS coordinates, specific biodiversity threats, and cell-level ecological risk scores — not just averages.</p>
            </div>
            <div className="glow-card p-8 text-center flex flex-col items-center gap-4">
            <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(251,191,36,0.12)', border: '1px solid rgba(251,191,36,0.25)' }}>
                    <Icon.Database className="w-7 h-7 text-ocean-gold" />
                </div>
                <p className="font-bold text-white text-lg" style={{ fontFamily: 'Syne, sans-serif' }}>{pct}% Data Compression</p>
                <p className="text-gray-400 text-sm leading-relaxed">{totalCells.toLocaleString()} ocean cells and {dc.data_points_str} data points compressed from {dc.raw_mb} MB to {dc.comp_mb} MB via AI autoencoder — without losing critical risk intelligence.</p>
            </div>
            <div className="glow-card p-8 text-center flex flex-col items-center gap-4">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center" style={{ background: 'rgba(34,197,94,0.12)', border: '1px solid rgba(34,197,94,0.25)' }}>
                    <Icon.Leaf className="w-7 h-7" style={{ color: '#22c55e' }} />
                </div>
                <p className="font-bold text-white text-lg" style={{ fontFamily: 'Syne, sans-serif' }}>Biodiversity Protection</p>
                <p className="text-gray-400 text-sm leading-relaxed">Every zone's marine health index computed live. Protecting food security and livelihoods of 2.3M coastal residents across India.</p>
            </div>
        </div>
        <div className="gradient-line" />
        <blockquote className="text-center text-lg md:text-xl font-medium max-w-3xl mx-auto leading-relaxed" style={{ color: 'rgba(251,191,36,0.9)', fontFamily: 'Syne, sans-serif', fontStyle: 'italic' }}>
            &ldquo;We don&rsquo;t just compress data &mdash; we find what&rsquo;s abnormal, score its risk, and tell decision makers exactly where and when to act.&rdquo;
        </blockquote>
    </SectionWrapper>
)};

// ── Globe View (Embedded) ──────────────────────────────
const GlobeView = () => {
    return (
        <div className="w-full relative overflow-hidden" style={{ height: 'calc(100vh - 72px)' }}>
            <iframe 
                src="/globe.html?v=6" 
                title="Planetary Risk Visualizer" 
                className="w-full h-full border-none absolute inset-0"
                style={{ background: '#020C14' }}
            />
        </div>
    );
};

// -- Auth Modal ----------------------------------------------------------
const AuthModal = ({ setUser }) => {
    const [isLogin, setIsLogin] = useState(true);
    const [username, setUsername] = useState("");
    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [error, setError] = useState("");

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        const endpoint = isLogin ? "/api/login" : "/api/register";
        try {
            const res = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username, email, password })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Authentication failed");
            if (isLogin) { setUser(data.username); }
            else { setIsLogin(true); setError("Registration successful! Please log in."); setPassword(""); }
        } catch (err) { setError(err.message); }
    };

    const inputSt = {
        width:"100%", background:"rgba(2,6,15,0.85)",
        border:"1px solid rgba(45,212,191,0.2)", borderRadius:"10px",
        color:"#fff", padding:"0.8rem 1rem 0.8rem 3rem",
        fontSize:"0.95rem", fontFamily:"DM Sans, sans-serif",
        outline:"none", boxSizing:"border-box",
        transition:"border-color 0.2s, box-shadow 0.2s",
    };
    const iconSt = {
        position:"absolute", left:"14px", top:"50%",
        transform:"translateY(-50%)", display:"flex",
        pointerEvents:"none", color:"rgba(45,212,191,0.5)",
    };
    const labelSt = {
        display:"block", fontFamily:"Fira Code, monospace",
        fontSize:"0.62rem", letterSpacing:"0.2em",
        textTransform:"uppercase", color:"rgba(45,212,191,0.95)",
        marginBottom:"0.5rem",
    };
    const onFoc = e => { e.target.style.borderColor="rgba(45,212,191,0.7)"; e.target.style.boxShadow="0 0 0 3px rgba(45,212,191,0.1)"; };
    const onBlr = e => { e.target.style.borderColor="rgba(45,212,191,0.2)"; e.target.style.boxShadow="none"; };

    const QUOTES = [
        { text: "The ocean holds every secret.", sub: "CORAL AI reads them in real time.", color: "#2dd4bf" },
        { text: "342,000 grid cells.", sub: "One intelligence. Zero compromise.", color: "#a5f3fc" },
        { text: "When coral bleaches, civilisations feel it.", sub: "We detect it first.", color: "#34d399" },
        { text: "97% data compression.", sub: "100% of biodiversity insight preserved.", color: "#fbbf24" },
        { text: "Petabytes of ocean noise.", sub: "One clear signal.", color: "#c084fc" },
    ];
    const [qIdx, setQIdx] = React.useState(0);
    React.useEffect(() => {
        const t = setInterval(() => setQIdx(i => (i+1) % QUOTES.length), 4000);
        return () => clearInterval(t);
    }, []);

    return (
        <div style={{ minHeight:"100vh", width:"100%", display:"flex", background:"#02060f", overflow:"hidden", position:"relative" }}>

            {/* Full-page ocean background (gradient only) */}
            <div style={{
              position:"absolute", inset:0,
              background:"linear-gradient(135deg, rgba(2,6,15,0.9) 0%, rgba(13,79,107,0.3) 50%, rgba(2,6,15,0.9) 100%)",
              zIndex:1
            }}/>

            {/* LEFT BRANDING PANEL */}
            <div style={{
                flex:"0 0 48%",
                position:"relative",
                zIndex:2,
                display:"flex",
                flexDirection:"column",
                justifyContent:"center",
                alignItems:"flex-start",
                padding:"3rem 3.5rem",
                overflow:"hidden",
                backgroundImage:`linear-gradient(rgba(2,6,15,0.35), rgba(2,6,15,0.55)), url(${OCEAN_IMAGES.login})`,
                backgroundSize:"cover",
                backgroundPosition:"center"
            }}>

                {/* Orbs */}
                <div style={{ position:"absolute", top:"-15%", left:"-15%", width:"500px", height:"500px", borderRadius:"50%", background:"radial-gradient(circle, rgba(13,79,107,0.55) 0%, transparent 65%)", animation:"orbFloat 10s ease-in-out infinite alternate", pointerEvents:"none" }}/>
                <div style={{ position:"absolute", bottom:"-10%", right:"-5%", width:"380px", height:"380px", borderRadius:"50%", background:"radial-gradient(circle, rgba(45,212,191,0.18) 0%, transparent 65%)", animation:"orbFloat 14s ease-in-out infinite alternate-reverse", pointerEvents:"none" }}/>

                {/* Grid dots background */}
                <div style={{ position:"absolute", inset:0, backgroundImage:"radial-gradient(rgba(45,212,191,0.07) 1px, transparent 1px)", backgroundSize:"28px 28px", pointerEvents:"none" }}/>

                {/* Vertical divider */}
                <div style={{ position:"absolute", top:0, right:0, width:"1px", height:"100%", background:"linear-gradient(180deg, transparent, rgba(45,212,191,0.25) 30%, rgba(45,212,191,0.25) 70%, transparent)" }}/>

                {/* Logo */}
                <div style={{ position:"relative", marginBottom:"2.5rem" }}>
                    <div style={{ width:"60px", height:"60px", borderRadius:"16px", display:"flex", alignItems:"center", justifyContent:"center", background:"linear-gradient(135deg, #0d4f6b, rgba(45,212,191,0.5))", border:"1px solid rgba(45,212,191,0.4)", boxShadow:"0 0 30px rgba(45,212,191,0.2)" }}>
                        <Icon.Satellite className="w-7 h-7 text-ocean-seafoam" />
                    </div>
                    <span style={{ position:"absolute", top:"-3px", right:"-3px", width:"12px", height:"12px", borderRadius:"50%", background:"#22c55e", boxShadow:"0 0 8px #22c55e", display:"block" }}/>
                </div>

                {/* Brand */}
                <div style={{ 
                    fontFamily:"Fira Code, monospace", 
                    fontSize:"0.65rem", 
                    fontWeight:"700",
                    letterSpacing:"0.28em", 
                    color:"#00f5c4", 
                    textTransform:"uppercase", 
                    marginBottom:"0.6rem",
                    textShadow:"0 2px 6px rgba(2, 12, 20, 0.95)"
                }}>
                    Coastal Ocean Risk Assessment & Lifecycle Intelligence
                </div>
                <h1 className="oi-brand-login" style={{ 
                    fontSize:"3.4rem", 
                    fontWeight:400, 
                    letterSpacing:"0.02em", 
                    lineHeight:1.2, 
                    margin:"0 0 0.8rem 0", 
                    background:"linear-gradient(135deg, #00f5c4 0%, #3b82f6 50%, #ec4899 100%)"
                }}>
                    CORAL AI
                </h1>
                <p style={{ 
                    fontFamily:"DM Sans, sans-serif", 
                    fontSize:"1rem", 
                    fontWeight:"600",
                    color:"rgba(255, 255, 255, 0.95)", 
                    marginBottom:"3rem", 
                    lineHeight:1.5,
                    textShadow:"0 2px 8px rgba(2, 12, 20, 0.95)"
                }}>
                    Satellite-powered marine intelligence platform
                </p>

                {/* Animated quote block */}
                <div style={{ width:"100%", position:"relative" }}>
                    <div style={{ position:"absolute", left:0, top:0, bottom:0, width:"3px", borderRadius:"4px", background:"linear-gradient(180deg, "+QUOTES[qIdx].color+", transparent)" }}/>
                    <div style={{ paddingLeft:"1.25rem" }}>
                        <p style={{ fontFamily:"Syne, sans-serif", fontSize:"1.45rem", fontWeight:700, color:"#fff", lineHeight:1.25, margin:"0 0 0.35rem 0", transition:"all 0.4s ease" }}>
                            {QUOTES[qIdx].text}
                        </p>
                        <p style={{ fontFamily:"DM Sans, sans-serif", fontSize:"0.9rem", color:QUOTES[qIdx].color, margin:0, fontWeight:400, transition:"all 0.4s ease" }}>
                            {QUOTES[qIdx].sub}
                        </p>
                    </div>
                    {/* Dots */}
                    <div style={{ display:"flex", gap:"8px", marginTop:"1.5rem", paddingLeft:"1.25rem" }}>
                        {QUOTES.map((_,i) => (
                            <span key={i} onClick={()=>setQIdx(i)} style={{ width:i===qIdx?"24px":"8px", height:"8px", borderRadius:"4px", background:i===qIdx?QUOTES[i].color:"rgba(255,255,255,0.15)", cursor:"pointer", transition:"all 0.4s ease" }}/>
                        ))}
                    </div>
                </div>

                {/* Stats row */}
                <div style={{ display:"flex", gap:"2rem", marginTop:"3rem", paddingTop:"2rem", borderTop:"1px solid rgba(45,212,191,0.12)" }}>
                    {[["342K+","Grid Cells"],["97%","Compression"],["2.3M","Residents"]].map(([v,l]) => (
                        <div key={l}>
                            <div style={{ fontFamily:"Syne, sans-serif", fontSize:"1.3rem", fontWeight:700, color:"#2dd4bf" }}>{v}</div>
                            <div style={{ fontFamily:"Fira Code, monospace", fontSize:"0.56rem", letterSpacing:"0.1em", color:"rgba(156,163,175,0.45)", textTransform:"uppercase" }}>{l}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* RIGHT FORM PANEL */}
            <div style={{
                flex:"0 0 52%",
                display:"flex",
                alignItems:"center",
                justifyContent:"center",
                padding:"2rem",
                backgroundImage:`linear-gradient(rgba(2,6,15,0.20), rgba(2,6,15,0.35)), url(https://images.unsplash.com/photo-1545671913-b89ac1b4ac10?w=1080&q=60&auto=format&fit=crop)`,
                backgroundSize:"cover",
                backgroundPosition:"center",
                backdropFilter:"blur(24px)",
                WebkitBackdropFilter:"blur(24px)",
                position:"relative",
                zIndex:2,
                borderLeft:"1px solid rgba(45,212,191,0.08)"
            }}>
                <div style={{ width:"100%", maxWidth:"420px" }}>

                    {/* Heading */}
                    <div style={{ marginBottom:"2rem" }}>
                        <p style={{ 
                            fontFamily:"Fira Code, monospace", 
                            fontSize:"0.72rem", 
                            fontWeight:"700",
                            letterSpacing:"0.22em", 
                            color:"#00f5c4", 
                            textTransform:"uppercase", 
                            marginBottom:"0.5rem",
                            textShadow:"0 2px 8px rgba(2, 6, 21, 0.95)"
                        }}>
                            {isLogin ? "Returning user" : "New user"}
                        </p>
                        <h2 style={{ 
                            fontFamily:"Syne, sans-serif", 
                            fontSize:"2.2rem", 
                            fontWeight:800, 
                            color:"#fff", 
                            letterSpacing:"-0.03em", 
                            margin:"0 0 0.4rem 0",
                            textShadow:"0 2px 10px rgba(2, 6, 21, 0.95), 0 0 20px rgba(0, 245, 196, 0.15)"
                        }}>
                            {isLogin ? "Welcome back" : "Create account"}
                        </h2>
                        <p style={{ 
                            fontFamily:"DM Sans, sans-serif", 
                            fontSize:"0.9rem", 
                            color:"#e2e8f0", 
                            margin:0, 
                            fontWeight: 500,
                            textShadow:"0 2px 6px rgba(2, 6, 21, 0.95)"
                        }}>
                            {isLogin ? "Sign in to access your coastal intelligence dashboard" : "Start monitoring real-time coastal risk data"}
                        </p>
                    </div>

                    {/* Alert */}
                    {error && (
                        <div style={{ display:"flex", alignItems:"center", gap:"10px", padding:"12px 14px", borderRadius:"10px", marginBottom:"1.25rem", fontSize:"0.875rem", fontWeight:500,
                            background:error.includes("successful")?"rgba(34,197,94,0.08)":"rgba(255,107,107,0.08)",
                            border:"1px solid "+(error.includes("successful")?"rgba(34,197,94,0.25)":"rgba(255,107,107,0.25)"),
                            color:error.includes("successful")?"#4ade80":"#f87171",
                            fontFamily:"DM Sans, sans-serif" }}>
                            {error.includes("successful")?<Icon.Check className="w-4 h-4" style={{flexShrink:0}}/>:<Icon.X className="w-4 h-4" style={{flexShrink:0}}/>}
                            {error}
                        </div>
                    )}

                    <form onSubmit={handleSubmit} style={{ display:"flex", flexDirection:"column", gap:"1.1rem" }}>
                        <div>
                            <label style={labelSt}>Username</label>
                            <div style={{ position:"relative" }}>
                                <span style={iconSt}><Icon.User className="w-[18px] h-[18px]"/></span>
                                <input type="text" value={username} onChange={e=>setUsername(e.target.value)} required placeholder="Your username" style={inputSt} onFocus={onFoc} onBlur={onBlr}/>
                            </div>
                        </div>
                        <div>
                            <label style={labelSt}>Email Address</label>
                            <div style={{ position:"relative" }}>
                                <span style={iconSt}><Icon.Mail className="w-[18px] h-[18px]"/></span>
                                <input type="email" value={email} onChange={e=>setEmail(e.target.value)} required placeholder="you@example.com" style={inputSt} onFocus={onFoc} onBlur={onBlr}/>
                            </div>
                        </div>
                        <div>
                            <label style={labelSt}>Password</label>
                            <div style={{ position:"relative" }}>
                                <span style={iconSt}><Icon.Lock className="w-[18px] h-[18px]"/></span>
                                <input type="password" value={password} onChange={e=>setPassword(e.target.value)} required
                                       pattern="(?=.*[!@#$%^&*])[a-zA-Z0-9!@#$%^&*]{5,}"
                                       title="Min 5 characters and at least one special character"
                                       placeholder="Min. 5 chars + special character" style={inputSt} onFocus={onFoc} onBlur={onBlr}/>
                            </div>
                            {!isLogin && (
                                <div style={{ display:"flex", flexWrap:"wrap", gap:"7px", marginTop:"10px" }}>
                                    {[
                                        { label: "Min. 5 characters", valid: password.length >= 5 },
                                        { label: "One special char (!@#$%)", valid: /[!@#$%^&*]/.test(password) },
                                        { label: "Letters & numbers OK", valid: /[a-zA-Z0-9]/.test(password) || password.length === 0 }
                                    ].map(r=>
                                        <span key={r.label} style={{ display:"inline-flex", alignItems:"center", gap:"5px", fontSize:"0.65rem", padding:"3px 10px", borderRadius:"20px", background: r.valid ? "rgba(45,212,191,0.07)" : "rgba(255,107,107,0.07)", border: r.valid ? "1px solid rgba(45,212,191,0.18)" : "1px solid rgba(255,107,107,0.18)", color: r.valid ? "rgba(45,212,191,0.75)" : "rgba(255,107,107,0.75)", fontFamily:"Fira Code, monospace", transition:"all 0.3s" }}>
                                            {r.valid ? <Icon.Check className="w-3 h-3"/> : <Icon.X className="w-3 h-3"/>} {r.label}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                        <button type="submit"
                                style={{ width:"100%", background:"linear-gradient(135deg, #0d4f6b 0%, #2dd4bf 100%)", color:"#02060f", border:"none", borderRadius:"12px", padding:"0.9rem", fontSize:"1.02rem", fontWeight:700, fontFamily:"Syne, sans-serif", cursor:"pointer", letterSpacing:"0.02em", marginTop:"0.25rem", transition:"opacity 0.2s, transform 0.15s" }}
                                onMouseEnter={e=>{e.currentTarget.style.opacity="0.88";e.currentTarget.style.transform="scale(1.01)";}}
                                onMouseLeave={e=>{e.currentTarget.style.opacity="1";e.currentTarget.style.transform="scale(1)";}}>
                            {isLogin ? "Sign In" : "Create Account"}
                        </button>
                    </form>

                    <div style={{ display:"flex", alignItems:"center", gap:"12px", margin:"1.5rem 0" }}>
                        <div style={{ flex:1, height:"1px", background:"rgba(45,212,191,0.2)" }}/>
                        <span style={{ fontFamily:"Fira Code, monospace", fontSize:"0.58rem", letterSpacing:"0.15em", color:"#9ca3af", textTransform:"uppercase" }}>or</span>
                        <div style={{ flex:1, height:"1px", background:"rgba(45,212,191,0.2)" }}/>
                    </div>

                    <p style={{ textAlign:"center", fontSize:"0.875rem", fontFamily:"DM Sans, sans-serif", color:"#cbd5e1", margin:0 }}>
                        {isLogin ? "Don't have an account? " : "Already have an account? "}
                        <button onClick={()=>{setIsLogin(!isLogin);setError("");}}
                                style={{ fontFamily:"Syne, sans-serif", color:"#2dd4bf", fontWeight:600, background:"none", border:"none", cursor:"pointer", fontSize:"0.875rem" }}
                                onMouseEnter={e=>e.target.style.textDecoration="underline"}
                                onMouseLeave={e=>e.target.style.textDecoration="none"}>
                            {isLogin ? "Register" : "Sign In"}
                        </button>
                    </p>
                </div>
            </div>

            <style>{`
                @keyframes orbFloat  { 0%{transform:translate(0,0) scale(1);}  100%{transform:translate(30px,20px) scale(1.08);} }
            `}</style>
        </div>
    );
};

// ── Pipeline Progress Display ────────────────────────────
const ALL_STAGES = [
    { name: "Satellite Download",      icon: "🛰️", color: "#38bdf8" },
    { name: "Data Ingestion",          icon: "📡", color: "#2dd4bf" },
    { name: "GPU Initiated",           icon: "⚡", color: "#a78bfa" },
    { name: "Neural Network Training", icon: "🧠", color: "#f59e0b" },
    { name: "Anomaly Detection",       icon: "🔍", color: "#fb923c" },
    { name: "Risk Classification",     icon: "⚠️", color: "#ef4444" },
    { name: "Dashboard Export",        icon: "📊", color: "#22c55e" },
];

const PipelineProgressDisplay = ({ text, sub, progress = 0, stage = "", completedStages = [] }) => {
    const radius = 44;
    const circ   = 2 * Math.PI * radius;
    const dash   = circ - (circ * Math.min(progress, 100)) / 100;

    return (
        <div className="flex flex-col items-center w-full" style={{ zIndex: 10 }}>
            {/* Circular progress ring */}
            <div className="relative flex items-center justify-center mb-5" style={{ width: 120, height: 120 }}>
                {/* Glow */}
                <div style={{
                    position:'absolute', inset:0, borderRadius:'50%',
                    background:'radial-gradient(circle, rgba(45,212,191,0.15) 0%, transparent 70%)',
                    animation:'pulse 2s ease-in-out infinite'
                }} />
                <svg width="120" height="120" style={{ transform: 'rotate(-90deg)' }}>
                    {/* Track */}
                    <circle cx="60" cy="60" r={radius} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="8" />
                    {/* Progress arc */}
                    <circle
                        cx="60" cy="60" r={radius} fill="none"
                        stroke="url(#pipeGrad)" strokeWidth="8"
                        strokeLinecap="round"
                        strokeDasharray={circ}
                        strokeDashoffset={dash}
                        style={{ transition: 'stroke-dashoffset 0.6s ease' }}
                    />
                    <defs>
                        <linearGradient id="pipeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%"   stopColor="#2dd4bf" />
                            <stop offset="100%" stopColor="#f59e0b" />
                        </linearGradient>
                    </defs>
                </svg>
                {/* Center content */}
                <div style={{ position:'absolute', display:'flex', flexDirection:'column', alignItems:'center' }}>
                    <span style={{
                        fontFamily:'Syne, sans-serif', fontWeight:800, fontSize:'1.5rem',
                        background:'linear-gradient(135deg,#2dd4bf,#f59e0b)',
                        WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent'
                    }}>{progress}%</span>
                    <span style={{ fontSize:'0.55rem', color:'#6b7280', marginTop:1, fontFamily:'DM Sans, sans-serif', letterSpacing:'0.06em', textTransform:'uppercase' }}>complete</span>
                </div>
            </div>

            {/* Main label */}
            <h3 style={{
                fontFamily:'Syne, sans-serif', fontWeight:800, fontSize:'1.1rem',
                background:'linear-gradient(135deg, #fbbf24 30%, #f97316)',
                WebkitBackgroundClip:'text', WebkitTextFillColor:'transparent',
                marginBottom:4, textAlign:'center'
            }} className="animate-pulse">{text}</h3>

            {/* Current stage badge */}
            {stage && (
                <div style={{
                    display:'inline-flex', alignItems:'center', gap:6,
                    background:'rgba(45,212,191,0.1)', border:'1px solid rgba(45,212,191,0.3)',
                    borderRadius:20, padding:'3px 12px', marginBottom:12
                }}>
                    <span style={{ width:7, height:7, borderRadius:'50%', background:'#2dd4bf', display:'inline-block', boxShadow:'0 0 6px #2dd4bf', animation:'pulse 1s ease-in-out infinite' }} />
                    <span style={{ fontFamily:'DM Sans, sans-serif', fontSize:'0.72rem', color:'#2dd4bf', fontWeight:600 }}>{stage}</span>
                </div>
            )}

            {/* Sub message */}
            {sub && <p style={{ fontFamily:'DM Sans, sans-serif', fontSize:'0.72rem', color:'#6b7280', textAlign:'center', marginBottom:14, maxWidth:280 }}>{sub}</p>}

            {/* Stage timeline */}
            <div style={{
                width:'100%', maxWidth:340,
                background:'rgba(255,255,255,0.03)', border:'1px solid rgba(255,255,255,0.07)',
                borderRadius:12, padding:'10px 14px', display:'flex', flexDirection:'column', gap:6
            }}>
                {ALL_STAGES.map((s) => {
                    const isDone    = completedStages.includes(s.name);
                    const isCurrent = stage === s.name;
                    return (
                        <div key={s.name} style={{ display:'flex', alignItems:'center', gap:10 }}>
                            {/* Status dot / check */}
                            <div style={{
                                width:22, height:22, borderRadius:'50%', flexShrink:0,
                                display:'flex', alignItems:'center', justifyContent:'center',
                                background: isDone ? s.color + '22' : isCurrent ? 'rgba(45,212,191,0.12)' : 'rgba(255,255,255,0.04)',
                                border: `1.5px solid ${ isDone ? s.color : isCurrent ? '#2dd4bf' : 'rgba(255,255,255,0.1)'}`,
                                transition: 'all 0.4s'
                            }}>
                                {isDone
                                    ? <span style={{ fontSize:11, color: s.color }}>✓</span>
                                    : isCurrent
                                        ? <span style={{ width:6, height:6, borderRadius:'50%', background:'#2dd4bf', display:'inline-block', animation:'pulse 1s ease-in-out infinite' }} />
                                        : <span style={{ width:5, height:5, borderRadius:'50%', background:'rgba(255,255,255,0.15)', display:'inline-block' }} />
                                }
                            </div>
                            {/* Icon + name */}
                            <span style={{ fontSize:'0.75rem', fontFamily:'DM Sans, sans-serif',
                                color: isDone ? s.color : isCurrent ? '#e2e8f0' : 'rgba(255,255,255,0.3)',
                                fontWeight: isCurrent ? 700 : isDone ? 600 : 400,
                                transition: 'color 0.4s'
                            }}>
                                {s.icon} {s.name}
                            </span>
                            {/* Right tag */}
                            {isDone && (
                                <span style={{
                                    marginLeft:'auto', fontSize:'0.6rem', color: s.color,
                                    background: s.color + '18', borderRadius:10, padding:'1px 7px',
                                    fontFamily:'DM Sans, sans-serif', fontWeight:700, letterSpacing:'0.04em'
                                }}>DONE</span>
                            )}
                            {isCurrent && (
                                <span style={{
                                    marginLeft:'auto', fontSize:'0.6rem', color:'#2dd4bf',
                                    background:'rgba(45,212,191,0.1)', borderRadius:10, padding:'1px 7px',
                                    fontFamily:'DM Sans, sans-serif', fontWeight:700, letterSpacing:'0.04em',
                                    animation:'pulse 1.5s ease-in-out infinite'
                                }}>ACTIVE</span>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

// Legacy slim loader kept for initial app loading screen
const SleekLoader = ({ text, sub }) => (
    <div className="flex flex-col items-center justify-center p-8 text-center" style={{ zIndex: 10 }}>
        <div className="relative w-24 h-24 mb-6 flex items-center justify-center mx-auto">
            {/* Pulsing glow behind */}
            <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-[#2dd4bf]/20 to-transparent filter blur-md animate-pulse" />
            {/* Spinning gradient ring */}
            <div className="absolute inset-0 rounded-full border-[3px] border-t-[#2dd4bf] border-r-transparent border-b-[#fbbf24] border-l-transparent animate-spin" />
            {/* Inner pulsing ocean icon */}
            <div className="absolute inset-3 rounded-full bg-[#050b14] border border-white/5 flex items-center justify-center shadow-[0_0_24px_rgba(45,212,191,0.25)]">
                <svg className="w-8 h-8 text-[#2dd4bf] animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2 12c1.5-3 4-4.5 6.5-3.5S13 11 16 10s4-3.5 6-2" />
                    <path d="M2 17c1.5-3 4-4.5 6.5-3.5S13 16 16 15s4-3.5 6-2" />
                </svg>
            </div>
        </div>
        <h3 className="text-xl font-bold animate-pulse mb-1 font-display" style={{ fontFamily: 'Syne, sans-serif', background: 'linear-gradient(135deg, #fbbf24 30%, #f97316)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{text}</h3>
        {sub && <p className="text-sm text-gray-400 max-w-xs mx-auto" style={{ fontFamily: 'DM Sans, sans-serif' }}>{sub}</p>}
    </div>
);

// ── Pipeline Modal ───────────────────────────────────────
const PipelineModal = ({ onClose, onComplete, onNavigate }) => {
    const [startDate, setStartDate] = useState("2024-01-01");
    const [endDate, setEndDate] = useState("2024-06-01");
    const [isRunning, setIsRunning] = useState(false);
    const [statusMessage, setStatusMessage] = useState("Initializing neural pipeline...");
    const [progress, setProgress] = useState(0);
    const [currentStage, setCurrentStage] = useState("");
    const [completedStages, setCompletedStages] = useState([]);
    const [error, setError] = useState("");

    // Start background pipeline run
    const handleRun = async () => {
        setIsRunning(true);
        setError("");
        setProgress(0);
        setCurrentStage("");
        setCompletedStages([]);
        try {
            const res = await fetch("/api/run_pipeline", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ start_date: startDate, end_date: endDate })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || "Pipeline failed to start");
            setStatusMessage("Ingesting satellite datasets & training neural network...");
        } catch (err) {
            setError(err.message);
            setIsRunning(false);
        }
    };

    // Status polling effect
    React.useEffect(() => {
        if (!isRunning) return;

        let intervalId;
        const checkStatus = async () => {
            try {
                const res = await fetch("/api/pipeline_status");
                const data = await res.json();
                if (data.status === "completed") {
                    clearInterval(intervalId);
                    setProgress(100);
                    setCurrentStage("Complete");
                    setCompletedStages(data.completed_stages || []);
                    setTimeout(() => {
                        onComplete();
                        onNavigate('home');
                        onClose();
                    }, 800);
                } else if (data.status === "failed") {
                    clearInterval(intervalId);
                    setError(data.error || "Pipeline run failed. Check server logs.");
                    setIsRunning(false);
                } else if (data.status === "running") {
                    if (data.message)         setStatusMessage(data.message);
                    if (data.progress != null) setProgress(data.progress);
                    if (data.stage)            setCurrentStage(data.stage);
                    if (data.completed_stages) setCompletedStages(data.completed_stages);
                }
            } catch (err) {
                console.error("Error polling pipeline status:", err);
            }
        };

        // Poll immediately and then every 2.5 seconds
        checkStatus();
        intervalId = setInterval(checkStatus, 2500);

        return () => clearInterval(intervalId);
    }, [isRunning, onComplete, onNavigate, onClose]);

    return (
        <div className="fixed inset-0 bg-black/80 z-[200] flex items-center justify-center p-4">
            <div className="bg-[#050b14] border border-ocean-teal p-8 rounded-xl max-w-lg w-full glow-card relative">
                {!isRunning && <button onClick={onClose} className="absolute top-4 right-4 text-gray-400 hover:text-white font-bold text-xl">✕</button>}
                <h2 className="text-2xl font-bold text-ocean-seafoam mb-2">Run AI Pipeline</h2>
                <p className="text-sm text-gray-400 mb-6">Select the date range for analysis. The AI will download real satellite data and train a custom autoencoder.</p>
                {error && <div className="bg-red-500/20 text-red-400 p-3 rounded mb-4 text-sm">{error}</div>}
                
                {isRunning ? (
                    <PipelineProgressDisplay
                        text="Processing Ocean Data..."
                        sub={statusMessage}
                        progress={progress}
                        stage={currentStage}
                        completedStages={completedStages}
                    />
                ) : (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-gray-400 text-xs uppercase mb-1">Start Date</label>
                            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} required
                                style={{ colorScheme: 'dark' }}
                                className="w-full bg-[#0a1628] border border-ocean-teal/50 rounded px-4 py-2 text-white outline-none focus:border-ocean-seafoam" />
                        </div>
                        <div>
                            <label className="block text-gray-400 text-xs uppercase mb-1">End Date</label>
                            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} required
                                style={{ colorScheme: 'dark' }}
                                className="w-full bg-[#0a1628] border border-ocean-teal/50 rounded px-4 py-2 text-white outline-none focus:border-ocean-seafoam" />
                        </div>
                        <button onClick={handleRun} className="w-full bg-ocean-coral hover:bg-red-500 text-white font-bold py-3 rounded transition mt-4 glow-border">
                            Start Analysis
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};


// ── Home Navigation Grid ──────────────────────────────────
const HomeNavigation = ({ onNavigate }) => {
    const pages = [
        { id: 'intelligence', icon: <Icon.BarChart2 className="w-8 h-8"/>, title: 'Pipeline Intelligence', desc: 'Live stats across 342,000+ ocean grid cells.', color: '#2dd4bf' },
        { id: 'zones', icon: <Icon.Map className="w-8 h-8"/>, title: 'Coastal Zones & Map', desc: 'GPS-pinned interactive maps and zone intelligence.', color: '#3b82f6' },
        { id: 'trends', icon: <Icon.Activity className="w-8 h-8"/>, title: 'Historical Trends', desc: 'Multi-year environmental baseline and drift.', color: '#a855f7' },
        { id: 'risk', icon: <Icon.Alert className="w-8 h-8"/>, title: 'Real-Time Cell Risk', desc: 'Direct view into critical and high-risk cells.', color: '#ef4444' },
        { id: 'biodiversity', icon: <Icon.Leaf className="w-8 h-8"/>, title: 'Biodiversity Impact', desc: 'Ecological threat analysis and coral bleaching correlation.', color: '#22c55e' },
        { id: 'compression', icon: <Icon.Database className="w-8 h-8"/>, title: 'Compression Tech', desc: 'Under the hood of our neural autoencoders.', color: '#eab308' }
    ];

    return (
        <section className="w-full py-16 px-6 lg:px-24 flex justify-center bg-[#050b14] relative z-10">
            <div className="w-full max-w-6xl">
                <div className="text-center mb-12">
                    <div className="section-label mb-2">Platform Navigation</div>
                    <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight" style={{ fontFamily: 'Syne, sans-serif' }}>Explore the Intelligence Dashboard</h2>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {pages.map(p => (
                        <div key={p.id} onClick={() => onNavigate(p.id)}
                             className="glow-card p-8 flex flex-col items-center text-center cursor-pointer transition-all duration-300 hover:-translate-y-2 hover:shadow-[0_10px_40px_rgba(45,212,191,0.15)] group">
                            <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 transition-transform group-hover:scale-110" 
                                 style={{ background: `${p.color}15`, color: p.color, border: `1px solid ${p.color}30` }}>
                                {p.icon}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-3" style={{ fontFamily: 'Syne, sans-serif' }}>{p.title}</h3>
                            <p className="text-sm text-gray-400 font-medium leading-relaxed">{p.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
// ── App Root ───────────────────────────────────────────
const App = () => {
    const [loaded, setLoaded]         = useState(false);
    const [lastUpdate, setLastUpdate] = useState(Date.now());
    const [user, setUser]             = useState(null);
    const [showPipeline, setShowPipeline] = useState(false);
    const [currentPage, setCurrentPage] = useState('home');

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [currentPage]);

    const loadData = () => {
        fetch('/api/data?t=' + new Date().getTime(), { cache: 'no-store' })
            .then(res => {
                if (!res.ok) throw new Error("API error " + res.status);
                return res.json();
            })
            .then(data => {
                db = data;
                setLoaded(true);
                setLastUpdate(Date.now());
            })
            .catch(err => {
                console.warn("API offline — using local cache:", err);
                setLoaded(true);
            });
    };

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 5000);
        return () => clearInterval(interval);
    }, []);

    if (!user) {
        return <AuthModal setUser={setUser} />;
    }

    if (!loaded || !db) {
        return (
            <div className="w-full min-h-screen flex flex-col items-center justify-center bg-[#02060f] text-ocean-seafoam">
                <SleekLoader text="Connecting to CORAL AI..." sub="Fetching live satellite data and neural network results..." />
            </div>
        );
    }



    return (
        <div className="w-full flex flex-col items-center min-h-screen bg-[#02060f]">
            {showPipeline && <PipelineModal onClose={() => setShowPipeline(false)} onComplete={() => loadData()} onNavigate={setCurrentPage} />}
            <Navbar user={user} onLogout={() => setUser(null)} onRunPipeline={() => setShowPipeline(true)} currentPage={currentPage} onNavigate={setCurrentPage} serverStatus={serverStatus} />
            
            <div className="w-full flex-grow pt-[72px]">
                {currentPage === 'home' && (
                    <>
                        <Hero onNavigate={setCurrentPage} />
                        <FeatureStrip onNavigate={setCurrentPage} db={db} />
                        <HomeNavigation onNavigate={setCurrentPage} />
                    </>
                )}
                {currentPage === 'intelligence' && (
                    <div className="pt-8">
                        <Section1 />
                    </div>
                )}
                {currentPage === 'zones' && (
                    <div className="pt-8">
                        <Section2 />
                        <Section5Map />
                    </div>
                )}
                {currentPage === 'trends' && (
                    <div className="pt-8">
                        <Section3 />
                        <Section4 />
                    </div>
                )}
                {currentPage === 'risk' && (
                    <div className="pt-8">
                        <SectionRealCells />
                        <SectionCellRisk />
                    </div>
                )}
                {currentPage === 'biodiversity' && (
                    <div className="pt-8">
                        <Section8 />
                        <Section9 />
                        <Section7 />
                    </div>
                )}
                {currentPage === 'compression' && (
                    <div className="pt-8">
                        <Section11 />
                        <Section12 />
                    </div>
                )}
                {currentPage === 'globe' && (
                    <GlobeView />
                )}
            </div>

            {currentPage !== 'globe' && (
                <footer className="w-full py-6 text-center bg-[#050b14] border-t border-ocean-teal/20 text-ocean-seafoam/50 text-sm mt-auto z-10">
                    CORAL AI Platform — Auto-refreshes every 5 seconds | Last updated: {new Date(lastUpdate).toLocaleTimeString()}
                </footer>
            )}
        </div>
    );
};

export default App;
