
import re

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace Navbar
old_navbar = '''const Navbar = ({ user, onLogout, onRunPipeline }) => (
    <nav className="fixed top-0 w-full z-[100] px-6 py-3 flex justify-between items-center"
         style={{ background: 'rgba(5,11,20,0.88)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(45,212,191,0.1)' }}>
        <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)' }}>
                <Icon.Wave className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight" style={{ fontFamily: 'Space Grotesk, sans-serif', background: 'linear-gradient(90deg, #2dd4bf, #a5f3fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>CORAL AI</span>
        </div>
        <div className="hidden md:flex gap-6 items-center">
            <a href="#metrics"      className="nav-link">Intelligence</a>
            <a href="#zones"        className="nav-link">Coastal Zones</a>
            <a href="#trends"       className="nav-link">Trends</a>
            <a href="#cellrisk"     className="nav-link">Cell Risk</a>
            <a href="#map"          className="nav-link">Map</a>
            <a href="#biodiversity" className="nav-link">Biodiversity</a>
            <a href="#compression"  className="nav-link">Compression</a>
            <a href="/globe.html" target="_blank"
               className="flex items-center gap-1.5 text-xs font-semibold text-ocean-coral hover:text-white transition">
                <Icon.Globe className="w-3.5 h-3.5" /> Globe
            </a>
            <button onClick={onRunPipeline}
                className="flex items-center gap-1.5 text-xs font-semibold px-4 py-2 rounded-lg transition"
                style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}>
                <Icon.Play className="w-3 h-3" /> Run Pipeline
            </button>
        </div>'''

new_navbar = '''const Navbar = ({ user, onLogout, onRunPipeline, currentPage, onNavigate }) => (
    <nav className="fixed top-0 w-full z-[100] px-6 py-3 flex justify-between items-center"
         style={{ background: 'rgba(5,11,20,0.88)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(45,212,191,0.1)' }}>
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => onNavigate('home')}>
            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)' }}>
                <Icon.Wave className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-lg tracking-tight" style={{ fontFamily: 'Space Grotesk, sans-serif', background: 'linear-gradient(90deg, #2dd4bf, #a5f3fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>CORAL AI</span>
        </div>
        <div className="hidden md:flex gap-6 items-center">
            <button onClick={() => onNavigate('intelligence')} className={`nav-link ${currentPage === 'intelligence' ? 'text-ocean-seafoam' : ''}`}>Intelligence</button>
            <button onClick={() => onNavigate('zones')}        className={`nav-link ${currentPage === 'zones' ? 'text-ocean-seafoam' : ''}`}>Coastal Zones</button>
            <button onClick={() => onNavigate('trends')}       className={`nav-link ${currentPage === 'trends' ? 'text-ocean-seafoam' : ''}`}>Trends</button>
            <button onClick={() => onNavigate('risk')}         className={`nav-link ${currentPage === 'risk' ? 'text-ocean-seafoam' : ''}`}>Cell Risk</button>
            <button onClick={() => onNavigate('biodiversity')} className={`nav-link ${currentPage === 'biodiversity' ? 'text-ocean-seafoam' : ''}`}>Biodiversity</button>
            <button onClick={() => onNavigate('compression')}  className={`nav-link ${currentPage === 'compression' ? 'text-ocean-seafoam' : ''}`}>Compression</button>
            <a href="/globe.html" target="_blank"
               className="flex items-center gap-1.5 text-xs font-semibold text-ocean-coral hover:text-white transition">
                <Icon.Globe className="w-3.5 h-3.5" /> Globe
            </a>
            <button onClick={onRunPipeline}
                className="flex items-center gap-1.5 text-xs font-semibold px-4 py-2 rounded-lg transition"
                style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}>
                <Icon.Play className="w-3 h-3" /> Run Pipeline
            </button>
        </div>'''

if old_navbar in content:
    content = content.replace(old_navbar, new_navbar)
else:
    print("WARNING: Could not find old navbar")

# 2. Replace Hero links
old_hero = '''const Hero = () => (
    <section className="relative w-full min-h-screen flex flex-col justify-center items-center text-center px-4 hero-gradient">'''

new_hero = '''const Hero = ({ onNavigate }) => (
    <section className="relative w-full min-h-screen flex flex-col justify-center items-center text-center px-4 hero-gradient">'''
if old_hero in content:
    content = content.replace(old_hero, new_hero)
else:
    print("WARNING: Could not find Hero start")

old_hero_links = '''<a href="#zones"
               className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
               style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}>
                <Icon.Map className="w-4 h-4" /> View Coastal Zones
            </a>
            <a href="#cellrisk"
               className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
               style={{ border: '1px solid rgba(255,107,107,0.6)', color: '#ff6b6b', fontFamily: 'Space Grotesk, sans-serif' }}
               onMouseEnter={e => { e.currentTarget.style.background='#ff6b6b'; e.currentTarget.style.color='#050b14'; }}
               onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='#ff6b6b'; }}>
                <Icon.Alert className="w-4 h-4" /> At-Risk Cells
            </a>'''

new_hero_links = '''<button onClick={() => onNavigate('zones')}
               className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
               style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}>
                <Icon.Map className="w-4 h-4" /> View Coastal Zones
            </button>
            <button onClick={() => onNavigate('risk')}
               className="flex items-center gap-2 font-semibold py-3 px-8 rounded-full transition-all hover:scale-105"
               style={{ border: '1px solid rgba(255,107,107,0.6)', color: '#ff6b6b', fontFamily: 'Space Grotesk, sans-serif' }}
               onMouseEnter={e => { e.currentTarget.style.background='#ff6b6b'; e.currentTarget.style.color='#050b14'; }}
               onMouseLeave={e => { e.currentTarget.style.background='transparent'; e.currentTarget.style.color='#ff6b6b'; }}>
                <Icon.Alert className="w-4 h-4" /> At-Risk Cells
            </button>'''
if old_hero_links in content:
    content = content.replace(old_hero_links, new_hero_links)
else:
    print("WARNING: Could not find Hero links")

# 3. Replace FeatureStrip links
old_fs = '''const FeatureStrip = () => {'''
new_fs = '''const FeatureStrip = ({ onNavigate }) => {'''
if old_fs in content:
    content = content.replace(old_fs, new_fs)
else:
    print("WARNING: Could not find FeatureStrip start")

old_fs_links = '''<a href="#metrics" style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", background:"linear-gradient(135deg, #0d4f6b, #2dd4bf)", color:"#050b14", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px" }}>
                            <Icon.BarChart2 className="w-4 h-4"/> View Intelligence
                        </a>
                        <a href="#compression" style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", border:"1px solid rgba(45,212,191,0.3)", color:"#2dd4bf", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px" }}>
                            <Icon.Database className="w-4 h-4"/> See Compression Data
                        </a>'''

new_fs_links = '''<button onClick={() => onNavigate('intelligence')} style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", background:"linear-gradient(135deg, #0d4f6b, #2dd4bf)", color:"#050b14", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px", border:"none", cursor:"pointer" }}>
                            <Icon.BarChart2 className="w-4 h-4"/> View Intelligence
                        </button>
                        <button onClick={() => onNavigate('compression')} style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", border:"1px solid rgba(45,212,191,0.3)", color:"#2dd4bf", background:"none", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px", cursor:"pointer" }}>
                            <Icon.Database className="w-4 h-4"/> See Compression Data
                        </button>'''
if old_fs_links in content:
    content = content.replace(old_fs_links, new_fs_links)
else:
    print("WARNING: Could not find FeatureStrip links")


# 4. Insert HomeNavigation component before App Root
home_nav_code = '''
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
                    <h2 className="text-3xl md:text-4xl font-bold text-white tracking-tight" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>Explore the Intelligence Dashboard</h2>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {pages.map(p => (
                        <div key={p.id} onClick={() => onNavigate(p.id)}
                             className="glow-card p-8 flex flex-col items-center text-center cursor-pointer transition-all duration-300 hover:-translate-y-2 hover:shadow-[0_10px_40px_rgba(45,212,191,0.15)] group">
                            <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6 transition-transform group-hover:scale-110" 
                                 style={{ background: `${p.color}15`, color: p.color, border: `1px solid ${p.color}30` }}>
                                {p.icon}
                            </div>
                            <h3 className="text-xl font-bold text-white mb-3" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>{p.title}</h3>
                            <p className="text-sm text-gray-400 font-medium leading-relaxed">{p.desc}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
'''

if 'const HomeNavigation' not in content:
    app_root_idx = content.find('// ── App Root ─')
    content = content[:app_root_idx] + home_nav_code + content[app_root_idx:]


# 5. Replace App render body
app_render_old = '''    return (
        <div className="w-full flex flex-col items-center">
            {showPipeline && <PipelineModal onClose={() => setShowPipeline(false)} onComplete={() => loadData()} />}
            <Navbar user={user} onLogout={() => setUser(null)} onRunPipeline={() => setShowPipeline(true)} />
            <Hero />
            <FeatureStrip />
            <Section1 />
            <Section2 />
            <Section3 />
            <Section4 />
            <Section5Map />
            <SectionRealCells />
            <SectionCellRisk />
            <Section8 />
            <Section9 />
            <Section7 />
            <Section11 />
            <Section12 />
            <footer className="w-full py-6 text-center bg-ocean-navy border-t border-ocean-teal/30 text-ocean-seafoam/60 text-sm mt-12 z-10">
                CORAL AI Platform — Auto-refreshes every 5 seconds | Last updated: {new Date(lastUpdate).toLocaleTimeString()}
            </footer>
        </div>
    );'''

app_render_new = '''    const [currentPage, setCurrentPage] = useState('home');

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [currentPage]);

    return (
        <div className="w-full flex flex-col items-center min-h-screen bg-[#02060f]">
            {showPipeline && <PipelineModal onClose={() => setShowPipeline(false)} onComplete={() => loadData()} />}
            <Navbar user={user} onLogout={() => setUser(null)} onRunPipeline={() => setShowPipeline(true)} currentPage={currentPage} onNavigate={setCurrentPage} />
            
            <div className="w-full flex-grow pt-[72px]">
                {currentPage === 'home' && (
                    <>
                        <Hero onNavigate={setCurrentPage} />
                        <FeatureStrip onNavigate={setCurrentPage} />
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
            </div>

            <footer className="w-full py-6 text-center bg-[#050b14] border-t border-ocean-teal/20 text-ocean-seafoam/50 text-sm mt-auto z-10">
                CORAL AI Platform — Auto-refreshes every 5 seconds | Last updated: {new Date(lastUpdate).toLocaleTimeString()}
            </footer>
        </div>
    );'''

if app_render_old in content:
    content = content.replace(app_render_old, app_render_new)
else:
    print("WARNING: Could not find old App render block")


with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patching complete!")
