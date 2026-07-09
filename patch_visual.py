"""
Visual Redesign Patch for App.jsx
- Adds ocean parallax backgrounds (real Unsplash CDN images)
- Updates SectionWrapper with parallax + section-specific overlays
- Updates Navbar with glassmorphism
- Updates Hero, FeatureStrip, Section1, Section3 with new styles
- Mixed typography: Syne (display), DM Sans (body), Fira Code (data)
"""
import re

PATH = r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx'
with open(PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# ══════════════════════════════════════════════════════════
# 1. ADD PARALLAX HOOK AND OCEAN IMAGES after the imports
# ══════════════════════════════════════════════════════════
old_db_line = 'let db = window.OCEANIQ_DATA || null;'
new_db_block = '''let db = window.OCEANIQ_DATA || null;

// ── Ocean background images (Unsplash CDN) ─────────────
const OCEAN_IMAGES = {
  home:          'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9?w=1920&q=80&auto=format&fit=crop',  // coral reef + tropical fish
  intelligence:  'https://images.unsplash.com/photo-1559827260-dc66d52bef19?w=1920&q=80&auto=format&fit=crop',  // underwater ocean blue
  zones:         'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1920&q=80&auto=format&fit=crop',  // ocean surface
  trends:        'https://images.unsplash.com/photo-1518020382113-a7e8fc38eac9?w=1920&q=80&auto=format&fit=crop',  // coral reef
  risk:          'https://images.unsplash.com/photo-1474440692490-2e83ae13ba29?w=1920&q=80&auto=format&fit=crop',  // dark ocean waves
  biodiversity:  'https://images.unsplash.com/photo-1565073624497-7144969d47b0?w=1920&q=80&auto=format&fit=crop',  // sea turtle
  compression:   'https://images.unsplash.com/photo-1519046904884-53103b34b206?w=1920&q=80&auto=format&fit=crop',  // ocean horizon
  calculator:    'https://images.unsplash.com/photo-1498623116890-37e912163d5d?w=1920&q=80&auto=format&fit=crop',  // jellyfish
  map:           'https://images.unsplash.com/photo-1505118380757-91f5f5632de0?w=1920&q=80&auto=format&fit=crop',  // ocean aerial
  cellrisk:      'https://images.unsplash.com/photo-1516026672322-bc52d61a5d15?w=1920&q=80&auto=format&fit=crop',  // dark underwater
};

// ── Parallax hook ──────────────────────────────────────
function useParallax(sectionId, strength = 0.28) {
  const bgRef = React.useRef(null);
  React.useEffect(() => {
    const el = bgRef.current;
    if (!el) return;
    const section = el.closest('section') || el.parentElement;
    const handleScroll = () => {
      const rect = section.getBoundingClientRect();
      const vh   = window.innerHeight;
      const visible = rect.top < vh && rect.bottom > 0;
      if (visible) {
        const pct    = (vh - rect.top) / (vh + rect.height);
        const offset = (pct - 0.5) * strength * rect.height;
        el.style.transform = `translateY(${offset}px) scale(1.1)`;
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [sectionId, strength]);
  return bgRef;
}'''

content = content.replace(old_db_line, new_db_block)

# ══════════════════════════════════════════════════════════
# 2. UPDATE SectionWrapper with parallax backgrounds
# ══════════════════════════════════════════════════════════
old_wrapper = '''const SectionWrapper = ({ id, children }) => (
    <section id={id} className="relative w-full py-16 px-6 lg:px-24 min-h-[50vh] flex flex-col items-center justify-center glow-border rounded-xl mb-12 bg-[#050b14]/50">
        <div className="w-full max-w-7xl z-10">{children}</div>
        <AnimatedWaves />
    </section>
);'''

new_wrapper = '''const SectionWrapper = ({ id, children, overlayClass }) => {
  const bgRef  = useParallax(id);
  const imgUrl = OCEAN_IMAGES[id] || OCEAN_IMAGES.intelligence;
  const overlay = overlayClass || `section-overlay-${id}`;
  return (
    <section id={id} className="ocean-parallax-section relative w-full py-20 px-6 lg:px-24 min-h-[50vh] flex flex-col items-center justify-center rounded-2xl mb-10 overflow-hidden"
             style={{ border: '1px solid rgba(255,255,255,0.07)', boxShadow: '0 8px 48px rgba(0,0,0,0.5)' }}>
      <div ref={bgRef} className="ocean-parallax-bg" style={{ backgroundImage: `url(${imgUrl})` }} />
      <div className={`ocean-parallax-overlay ${overlay}`} />
      <div className="section-content-z w-full max-w-7xl">{children}</div>
      <AnimatedWaves />
    </section>
  );
};'''

content = content.replace(old_wrapper, new_wrapper)

# ══════════════════════════════════════════════════════════
# 3. UPDATE NAVBAR — glassmorphism + vibrant gradient logo
# ══════════════════════════════════════════════════════════
old_nav_outer = '''    <nav className="fixed top-0 w-full z-[100] px-6 py-3 flex justify-between items-center"
         style={{ background: 'rgba(5,11,20,0.88)', backdropFilter: 'blur(20px)', borderBottom: '1px solid rgba(45,212,191,0.1)' }}>'''

new_nav_outer = '''    <nav className="glass-nav fixed top-0 w-full z-[100] px-6 py-3 flex justify-between items-center">'''

content = content.replace(old_nav_outer, new_nav_outer)

# Update logo background in Navbar
old_logo_div = '''            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)' }}>'''
new_logo_div = '''            <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)', boxShadow: '0 0 16px rgba(192,132,252,0.4)' }}>'''
content = content.replace(old_logo_div, new_logo_div, 1)

# Update logo text gradient
old_logo_text = "background: 'linear-gradient(90deg, #2dd4bf, #a5f3fc)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent'"
new_logo_text = "background: 'linear-gradient(90deg, #2dd4bf, #c084fc, #fb7185)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', fontFamily: 'Syne, sans-serif'"
content = content.replace(old_logo_text, new_logo_text)

# Update "Run Pipeline" button in nav
old_run_btn_style = "style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}"
new_run_btn_style = "style={{ background: 'linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)', color: '#fff', fontFamily: 'Syne, sans-serif', fontWeight: 700, boxShadow: '0 0 20px rgba(124,58,237,0.4)' }}"
content = content.replace(old_run_btn_style, new_run_btn_style)

# ══════════════════════════════════════════════════════════
# 4. UPDATE HERO — vibrant multi-color
# ══════════════════════════════════════════════════════════
# Hero section outer wrapper
old_hero_section = '''    <section className="relative min-h-screen flex flex-col items-center justify-center text-center overflow-hidden px-6 hero-gradient">'''
new_hero_section = '''    <section className="ocean-parallax-section relative min-h-screen flex flex-col items-center justify-center text-center overflow-hidden px-6">'''
content = content.replace(old_hero_section, new_hero_section)

# Add parallax bg to Hero right after section open — insert via emblem div
old_hero_inner = '''        {/* Animated SVG emblem */}
        <div className="mb-8 z-10 relative">
            <div className="w-24 h-24 rounded-2xl flex items-center justify-center mx-auto"
                 style={{ background: 'linear-gradient(135deg, rgba(13,79,107,0.8), rgba(45,212,191,0.3))', border: '1px solid rgba(45,212,191,0.3)', boxShadow: '0 0 40px rgba(45,212,191,0.15)' }}>'''

new_hero_inner = '''        {/* Ocean parallax bg for hero */}
        <div className="ocean-parallax-bg" style={{ backgroundImage: `url(${OCEAN_IMAGES.home})`, filter: 'brightness(0.20) saturate(1.6)' }} />
        <div className="ocean-parallax-overlay section-overlay-home" />

        {/* Animated SVG emblem */}
        <div className="mb-8 z-10 relative">
            <div className="w-24 h-24 rounded-2xl flex items-center justify-center mx-auto"
                 style={{ background: 'linear-gradient(135deg, rgba(124,58,237,0.7), rgba(13,79,107,0.8), rgba(45,212,191,0.4))', border: '1px solid rgba(192,132,252,0.4)', boxShadow: '0 0 50px rgba(124,58,237,0.3), 0 0 80px rgba(45,212,191,0.15)' }}>'''
content = content.replace(old_hero_inner, new_hero_inner)

# Update CORAL AI title in hero
old_coral_h1 = "background: 'linear-gradient(135deg, #fff 30%, #2dd4bf 70%, #a5f3fc)'"
new_coral_h1 = "background: 'linear-gradient(135deg, #ffffff 10%, #a5f3fc 35%, #c084fc 60%, #fb7185 85%)'"
content = content.replace(old_coral_h1, new_coral_h1)

# Update CORAL AI h1 font
old_h1_font = "fontFamily: 'Space Grotesk, sans-serif', fontSize: 'clamp(3rem, 8vw, 5.5rem)', lineHeight: 1"
new_h1_font = "fontFamily: 'Syne, sans-serif', fontSize: 'clamp(3rem, 8vw, 5.5rem)', lineHeight: 1"
content = content.replace(old_h1_font, new_h1_font)

# Hero buttons
old_hero_btn1 = "style={{ background: 'linear-gradient(135deg, #0d4f6b, #2dd4bf)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}"
new_hero_btn1 = "className='glass-btn' style={{ background: 'linear-gradient(135deg, rgba(13,79,107,0.6), rgba(45,212,191,0.4))', color: '#e2e8f0', fontFamily: 'Syne, sans-serif', padding:'12px 32px', borderRadius:'999px', fontWeight: 700, backdropFilter:'blur(16px)' }}"
content = content.replace(old_hero_btn1, new_hero_btn1)

old_hero_btn2 = "style={{ border: '1px solid rgba(255,107,107,0.6)', color: '#ff6b6b', fontFamily: 'Space Grotesk, sans-serif' }}"
new_hero_btn2 = "style={{ border: '1px solid rgba(251,113,133,0.6)', color: '#fb7185', fontFamily: 'Syne, sans-serif', padding:'12px 32px', borderRadius:'999px', fontWeight: 700, backdropFilter:'blur(16px)', background:'rgba(251,113,133,0.08)' }}"
content = content.replace(old_hero_btn2, new_hero_btn2)

old_hero_btn2_hover_enter = "e.currentTarget.style.background='#ff6b6b'; e.currentTarget.style.color='#050b14';"
new_hero_btn2_hover_enter = "e.currentTarget.style.background='rgba(251,113,133,0.25)'; e.currentTarget.style.color='#fff';"
content = content.replace(old_hero_btn2_hover_enter, new_hero_btn2_hover_enter)

old_hero_btn2_hover_leave = "e.currentTarget.style.background='transparent'; e.currentTarget.style.color='#ff6b6b';"
new_hero_btn2_hover_leave = "e.currentTarget.style.background='rgba(251,113,133,0.08)'; e.currentTarget.style.color='#fb7185';"
content = content.replace(old_hero_btn2_hover_leave, new_hero_btn2_hover_leave)

# ══════════════════════════════════════════════════════════
# 5. UPDATE Section titles to use Syne display font
# ══════════════════════════════════════════════════════════
# Replace all Space Grotesk references
content = content.replace("fontFamily: 'Space Grotesk, sans-serif'", "fontFamily: 'Syne, sans-serif'")
content = content.replace("fontFamily:\"Space Grotesk, sans-serif\"", "fontFamily:\"Syne, sans-serif\"")
content = content.replace("font-family:Space Grotesk, sans-serif", "font-family:Syne, sans-serif")
content = content.replace("'Space Grotesk, sans-serif'", "'Syne, sans-serif'")

# Replace JetBrains Mono references in inline styles
content = content.replace("fontFamily: 'JetBrains Mono', monospace", "fontFamily: 'Fira Code, monospace'")
content = content.replace("fontFamily:\"JetBrains Mono, monospace\"", "fontFamily:\"Fira Code, monospace\"")
content = content.replace("Inter, sans-serif", "DM Sans, sans-serif")

# ══════════════════════════════════════════════════════════
# 6. UPDATE Section1 (Intelligence) — teal/blue accent
# ══════════════════════════════════════════════════════════
old_sec1_title = '<h2 className="section-title text-3xl text-white">Live Pipeline Intelligence</h2>'
new_sec1_title = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #2dd4bf, #60a5fa)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Live Pipeline Intelligence</h2>'
content = content.replace(old_sec1_title, new_sec1_title)

# ══════════════════════════════════════════════════════════
# 7. UPDATE Feature strip - section heading
# ══════════════════════════════════════════════════════════
old_feature_h2 = '<h2 style={{ fontFamily:\'Space Grotesk, sans-serif\''
# Already replaced above

# ══════════════════════════════════════════════════════════
# 8. UPDATE Section3 (Trends) — purple accent
# ══════════════════════════════════════════════════════════
old_trends_title = '<h2 className="text-3xl font-bold text-ocean-seafoam">📈 Environmental Trends</h2>'
new_trends_title = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #c084fc, #fb7185)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>Environmental Trends</h2>'
content = content.replace(old_trends_title, new_trends_title)

# Trend card source badge label
old_source_label = 'className="text-[10px] text-center font-mono font-bold text-ocean-seafoam/70 mb-3"'
new_source_label = 'className="text-[10px] text-center mb-3" style={{ fontFamily:"Fira Code, monospace", color:"rgba(192,132,252,0.75)", letterSpacing:"0.03em" }}'
content = content.replace(old_source_label, new_source_label)

# ══════════════════════════════════════════════════════════
# 9. UPDATE Section2 (Coastal Zones) heading
# ══════════════════════════════════════════════════════════
old_zones_h2 = '<h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🗺️ Coastal Zone Intelligence</h2>'
new_zones_h2 = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #60a5fa, #818cf8)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">Coastal Zone Intelligence</h2>'
content = content.replace(old_zones_h2, new_zones_h2)

# ══════════════════════════════════════════════════════════
# 10. UPDATE Biodiversity Map heading
# ══════════════════════════════════════════════════════════
old_map_h2 = '<h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🗾 Coastal Zone Risk & Biodiversity Map</h2>'
new_map_h2 = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #34d399, #2dd4bf)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">Coastal Zone Risk & Biodiversity Map</h2>'
content = content.replace(old_map_h2, new_map_h2)

# ══════════════════════════════════════════════════════════
# 11. UPDATE Cell Risk heading
# ══════════════════════════════════════════════════════════
old_risk_h2 = '<h2 className="text-3xl font-bold text-ocean-seafoam mb-2">⚠️ At-Risk Cell Registry</h2>'
new_risk_h2 = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #fb7185, #f97316)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">At-Risk Cell Registry</h2>'
content = content.replace(old_risk_h2, new_risk_h2)

# ══════════════════════════════════════════════════════════
# 12. UPDATE RM-NPI Calculator heading
# ══════════════════════════════════════════════════════════
old_calc_h2 = '<h2 className="text-3xl font-bold text-ocean-seafoam mb-2">🧮 River Mouth Nutrient Pressure Index (RM-NPI)</h2>'
new_calc_h2 = '<h2 style={{ fontFamily:"Syne, sans-serif", fontSize:"1.8rem", fontWeight:800, background:"linear-gradient(135deg, #fbbf24, #f97316)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">RM-NPI Calculator</h2>'
content = content.replace(old_calc_h2, new_calc_h2)

# ══════════════════════════════════════════════════════════
# 13. UPDATE Compression section heading
# ══════════════════════════════════════════════════════════
old_comp_h2_1 = 'className="text-3xl font-bold text-ocean-seafoam mb-2">📦 Smart Data Compression'
new_comp_h2_1 = 'style={{ fontFamily:"Syne, sans-serif", fontSize:"2rem", fontWeight:800, background:"linear-gradient(135deg, #f97316, #fbbf24)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }} className="mb-2">Smart Data Compression'
content = content.replace(old_comp_h2_1, new_comp_h2_1)

# ══════════════════════════════════════════════════════════
# 14. UPDATE Home page nav cards with glass styling
# ══════════════════════════════════════════════════════════
old_home_cta_inner = '''            style={{ background: 'rgba(13,79,107,0.3)', border: '1px solid rgba(45,212,191,0.18)', color: '#050b14', fontFamily: 'Space Grotesk, sans-serif' }}'''
# Already covered by Space Grotesk replacement above

# Update Intelligence nav button in home page CTA strip
old_view_intel_btn = "style={{ fontFamily:\"Syne, sans-serif\", fontWeight:700, fontSize:\"0.875rem\", padding:\"0.65rem 1.5rem\", borderRadius:\"10px\", background:\"linear-gradient(135deg, #0d4f6b, #2dd4bf)\", color:\"#050b14\", textDecoration:\"none\", display:\"inline-flex\", alignItems:\"center\", gap:\"6px\", border:\"none\", cursor:\"pointer\" }}"
new_view_intel_btn = "style={{ fontFamily:\"Syne, sans-serif\", fontWeight:700, fontSize:\"0.875rem\", padding:\"0.65rem 1.5rem\", borderRadius:\"10px\", background:\"linear-gradient(135deg, #7c3aed, #0d4f6b, #2dd4bf)\", color:\"#fff\", textDecoration:\"none\", display:\"inline-flex\", alignItems:\"center\", gap:\"6px\", border:\"none\", cursor:\"pointer\", boxShadow:\"0 0 20px rgba(124,58,237,0.35)\" }}"
content = content.replace(old_view_intel_btn, new_view_intel_btn)

# ══════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════
with open(PATH, 'w', encoding='utf-8') as f:
    f.write(content)
print("DONE - Patch applied successfully!")
