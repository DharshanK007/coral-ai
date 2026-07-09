
# Patch: Insert a rich "Why CORAL AI" features section between Hero and Section1

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with "// -- Section 1"  or  "// ─ Section 1"
insert_line = None
for i, line in enumerate(lines):
    if 'Section 1' in line and 'Live Pipeline' in line:
        insert_line = i
        break

if insert_line is None:
    # fallback: find "const Section1"
    for i, line in enumerate(lines):
        if 'const Section1' in line:
            insert_line = i - 1
            break

print(f"Inserting FeatureStrip before line {insert_line+1}")

new_section = '''// -- FeatureStrip: Why CORAL AI ----------------------------------------
const FeatureStrip = () => {
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
            desc: "Strong r = -0.84 correlation between RM-NPI and marine health index. Protecting food security for 2.3M coastal residents across India.",
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

    const stats = [
        { v: "97%",    l: "Data compression",      c: "#2dd4bf" },
        { v: "342K+",  l: "Ocean grid cells",       c: "#a5f3fc" },
        { v: "2,400",  l: "MB raw → 72 MB stored",  c: "#fbbf24" },
        { v: "r=-0.84",l: "Biodiversity correlation",c: "#34d399" },
        { v: "2.3M",   l: "Residents protected",    c: "#c084fc" },
        { v: "<2 min", l: "Full pipeline runtime",  c: "#fb923c" },
    ];

    return (
        <section style={{ width:"100%", padding:"5rem 1.5rem", background:"linear-gradient(180deg, #050b14 0%, #030810 100%)", position:"relative" }}>
            <div style={{ maxWidth:"1280px", margin:"0 auto" }}>

                {/* Section Header */}
                <div style={{ textAlign:"center", marginBottom:"3.5rem" }}>
                    <p style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.62rem", letterSpacing:"0.25em", textTransform:"uppercase", color:"rgba(45,212,191,0.6)", marginBottom:"0.75rem" }}>
                        Why CORAL AI
                    </p>
                    <h2 style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"clamp(1.8rem, 4vw, 2.75rem)", fontWeight:800, color:"#fff", letterSpacing:"-0.03em", lineHeight:1.1, margin:"0 0 1rem 0" }}>
                        A platform built for impact,<br/>
                        <span style={{ background:"linear-gradient(90deg, #2dd4bf, #a5f3fc)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>not just analysis</span>
                    </h2>
                    <p style={{ fontFamily:"Inter, sans-serif", fontSize:"1rem", color:"rgba(156,163,175,0.6)", maxWidth:"560px", margin:"0 auto", lineHeight:1.7 }}>
                        From raw satellite bytes to actionable coastal risk intelligence — CORAL AI makes oceanic monitoring affordable and precise for governments and researchers.
                    </p>
                </div>

                {/* Stats Banner */}
                <div style={{ display:"grid", gridTemplateColumns:"repeat(auto-fit, minmax(150px, 1fr))", gap:"1px", background:"rgba(45,212,191,0.08)", borderRadius:"16px", overflow:"hidden", border:"1px solid rgba(45,212,191,0.12)", marginBottom:"4rem" }}>
                    {stats.map(s => (
                        <div key={s.l} style={{ padding:"1.5rem 1.25rem", textAlign:"center", background:"#030810" }}>
                            <div style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"1.75rem", fontWeight:800, color:s.c, letterSpacing:"-0.04em", lineHeight:1 }}>{s.v}</div>
                            <div style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.58rem", letterSpacing:"0.1em", color:"rgba(156,163,175,0.45)", textTransform:"uppercase", marginTop:"0.4rem" }}>{s.l}</div>
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
                            <span style={{ display:"inline-flex", alignItems:"center", fontFamily:"JetBrains Mono, monospace", fontSize:"0.58rem", letterSpacing:"0.15em", textTransform:"uppercase", color:f.color, background:`${f.color}12`, border:`1px solid ${f.color}25`, borderRadius:"20px", padding:"2px 10px", marginBottom:"1.25rem" }}>
                                {f.badge}
                            </span>

                            {/* Icon */}
                            <div style={{ width:"52px", height:"52px", borderRadius:"14px", display:"flex", alignItems:"center", justifyContent:"center", background:`${f.color}10`, border:`1px solid ${f.color}22`, marginBottom:"1.25rem" }}>
                                {f.icon}
                            </div>

                            {/* Text */}
                            <h3 style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"1.1rem", fontWeight:700, color:"#fff", margin:"0 0 0.6rem 0", letterSpacing:"-0.02em" }}>
                                {f.title}
                            </h3>
                            <p style={{ fontFamily:"Inter, sans-serif", fontSize:"0.875rem", color:"rgba(156,163,175,0.6)", margin:0, lineHeight:1.65 }}>
                                {f.desc}
                            </p>
                        </div>
                    ))}
                </div>

                {/* Bottom CTA strip */}
                <div style={{ marginTop:"4rem", padding:"2rem 2.5rem", borderRadius:"16px", background:"linear-gradient(135deg, rgba(13,79,107,0.3), rgba(45,212,191,0.08))", border:"1px solid rgba(45,212,191,0.18)", display:"flex", flexWrap:"wrap", alignItems:"center", justifyContent:"space-between", gap:"1.5rem" }}>
                    <div>
                        <h3 style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"1.25rem", fontWeight:700, color:"#fff", margin:"0 0 0.35rem 0" }}>
                            Ready to explore the data?
                        </h3>
                        <p style={{ fontFamily:"Inter, sans-serif", fontSize:"0.875rem", color:"rgba(156,163,175,0.55)", margin:0 }}>
                            Scroll down to see real coastal zone risk scores, live trends, and biodiversity intelligence.
                        </p>
                    </div>
                    <div style={{ display:"flex", gap:"1rem" }}>
                        <a href="#metrics" style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", background:"linear-gradient(135deg, #0d4f6b, #2dd4bf)", color:"#050b14", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px" }}>
                            <Icon.BarChart2 className="w-4 h-4"/> View Intelligence
                        </a>
                        <a href="#compression" style={{ fontFamily:"Space Grotesk, sans-serif", fontWeight:700, fontSize:"0.875rem", padding:"0.65rem 1.5rem", borderRadius:"10px", border:"1px solid rgba(45,212,191,0.3)", color:"#2dd4bf", textDecoration:"none", display:"inline-flex", alignItems:"center", gap:"6px" }}>
                            <Icon.Database className="w-4 h-4"/> See Compression Data
                        </a>
                    </div>
                </div>

            </div>
        </section>
    );
};

'''

# Insert after Hero and before Section1 comment
new_lines = lines[:insert_line] + [new_section] + lines[insert_line:]

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("SUCCESS. Total lines:", len(new_lines))
