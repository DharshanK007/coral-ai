
# Patch: Replace AuthModal with two-panel layout
# Uses line numbers to find the exact block to replace

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the Auth Modal block start and Pipeline Modal start
auth_start = None
pipeline_start = None
for i, line in enumerate(lines):
    if 'Auth Modal' in line and auth_start is None:
        auth_start = i
    if 'Pipeline Modal' in line and auth_start is not None:
        pipeline_start = i
        break

print(f"Auth Modal at line {auth_start+1}, Pipeline Modal at line {pipeline_start+1}")

new_component = '''// -- Auth Modal ----------------------------------------------------------
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
        fontSize:"0.95rem", fontFamily:"Inter, sans-serif",
        outline:"none", boxSizing:"border-box",
        transition:"border-color 0.2s, box-shadow 0.2s",
    };
    const iconSt = {
        position:"absolute", left:"14px", top:"50%",
        transform:"translateY(-50%)", display:"flex",
        pointerEvents:"none", color:"rgba(45,212,191,0.5)",
    };
    const labelSt = {
        display:"block", fontFamily:"JetBrains Mono, monospace",
        fontSize:"0.62rem", letterSpacing:"0.2em",
        textTransform:"uppercase", color:"rgba(45,212,191,0.65)",
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

            {/* LEFT BRANDING PANEL */}
            <div style={{ flex:"0 0 48%", position:"relative", display:"flex", flexDirection:"column", justifyContent:"center", alignItems:"flex-start", padding:"3rem 3.5rem", overflow:"hidden" }}>

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
                <div style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.58rem", letterSpacing:"0.28em", color:"rgba(45,212,191,0.5)", textTransform:"uppercase", marginBottom:"0.6rem" }}>
                    Coastal Ocean Risk Assessment & Lifecycle Intelligence
                </div>
                <h1 style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"3rem", fontWeight:800, letterSpacing:"-0.04em", lineHeight:1, margin:"0 0 0.5rem 0", background:"linear-gradient(135deg, #fff 30%, #2dd4bf 70%, #a5f3fc)", WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" }}>
                    CORAL AI
                </h1>
                <p style={{ fontFamily:"Inter, sans-serif", fontSize:"0.95rem", color:"rgba(156,163,175,0.55)", marginBottom:"3rem", lineHeight:1.5 }}>
                    Satellite-powered marine intelligence platform
                </p>

                {/* Animated quote block */}
                <div style={{ width:"100%", position:"relative" }}>
                    <div style={{ position:"absolute", left:0, top:0, bottom:0, width:"3px", borderRadius:"4px", background:"linear-gradient(180deg, "+QUOTES[qIdx].color+", transparent)" }}/>
                    <div style={{ paddingLeft:"1.25rem" }}>
                        <p style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"1.45rem", fontWeight:700, color:"#fff", lineHeight:1.25, margin:"0 0 0.35rem 0", transition:"all 0.4s ease" }}>
                            {QUOTES[qIdx].text}
                        </p>
                        <p style={{ fontFamily:"Inter, sans-serif", fontSize:"0.9rem", color:QUOTES[qIdx].color, margin:0, fontWeight:400, transition:"all 0.4s ease" }}>
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
                            <div style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"1.3rem", fontWeight:700, color:"#2dd4bf" }}>{v}</div>
                            <div style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.56rem", letterSpacing:"0.1em", color:"rgba(156,163,175,0.45)", textTransform:"uppercase" }}>{l}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* RIGHT FORM PANEL */}
            <div style={{ flex:"0 0 52%", display:"flex", alignItems:"center", justifyContent:"center", padding:"2rem", background:"rgba(5,11,20,0.55)" }}>
                <div style={{ width:"100%", maxWidth:"420px" }}>

                    {/* Heading */}
                    <div style={{ marginBottom:"2rem" }}>
                        <p style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.6rem", letterSpacing:"0.22em", color:"rgba(45,212,191,0.55)", textTransform:"uppercase", marginBottom:"0.5rem" }}>
                            {isLogin ? "Returning user" : "New user"}
                        </p>
                        <h2 style={{ fontFamily:"Space Grotesk, sans-serif", fontSize:"2rem", fontWeight:700, color:"#fff", letterSpacing:"-0.03em", margin:"0 0 0.4rem 0" }}>
                            {isLogin ? "Welcome back" : "Create account"}
                        </h2>
                        <p style={{ fontFamily:"Inter, sans-serif", fontSize:"0.875rem", color:"rgba(156,163,175,0.5)", margin:0 }}>
                            {isLogin ? "Sign in to access your coastal intelligence dashboard" : "Start monitoring real-time coastal risk data"}
                        </p>
                    </div>

                    {/* Alert */}
                    {error && (
                        <div style={{ display:"flex", alignItems:"center", gap:"10px", padding:"12px 14px", borderRadius:"10px", marginBottom:"1.25rem", fontSize:"0.875rem", fontWeight:500,
                            background:error.includes("successful")?"rgba(34,197,94,0.08)":"rgba(255,107,107,0.08)",
                            border:"1px solid "+(error.includes("successful")?"rgba(34,197,94,0.25)":"rgba(255,107,107,0.25)"),
                            color:error.includes("successful")?"#4ade80":"#f87171",
                            fontFamily:"Inter, sans-serif" }}>
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
                                    {["Min. 5 characters","One special char (!@#$%)","Letters & numbers OK"].map(r=>
                                        <span key={r} style={{ display:"inline-flex", alignItems:"center", gap:"5px", fontSize:"0.65rem", padding:"3px 10px", borderRadius:"20px", background:"rgba(45,212,191,0.07)", border:"1px solid rgba(45,212,191,0.18)", color:"rgba(45,212,191,0.75)", fontFamily:"JetBrains Mono, monospace" }}>
                                            <Icon.Check className="w-3 h-3"/> {r}
                                        </span>
                                    )}
                                </div>
                            )}
                        </div>
                        <button type="submit"
                                style={{ width:"100%", background:"linear-gradient(135deg, #0d4f6b 0%, #2dd4bf 100%)", color:"#02060f", border:"none", borderRadius:"12px", padding:"0.9rem", fontSize:"1.02rem", fontWeight:700, fontFamily:"Space Grotesk, sans-serif", cursor:"pointer", letterSpacing:"0.02em", marginTop:"0.25rem", transition:"opacity 0.2s, transform 0.15s" }}
                                onMouseEnter={e=>{e.currentTarget.style.opacity="0.88";e.currentTarget.style.transform="scale(1.01)";}}
                                onMouseLeave={e=>{e.currentTarget.style.opacity="1";e.currentTarget.style.transform="scale(1)";}}>
                            {isLogin ? "Sign In" : "Create Account"}
                        </button>
                    </form>

                    <div style={{ display:"flex", alignItems:"center", gap:"12px", margin:"1.5rem 0" }}>
                        <div style={{ flex:1, height:"1px", background:"rgba(45,212,191,0.1)" }}/>
                        <span style={{ fontFamily:"JetBrains Mono, monospace", fontSize:"0.58rem", letterSpacing:"0.15em", color:"rgba(156,163,175,0.3)", textTransform:"uppercase" }}>or</span>
                        <div style={{ flex:1, height:"1px", background:"rgba(45,212,191,0.1)" }}/>
                    </div>

                    <p style={{ textAlign:"center", fontSize:"0.875rem", fontFamily:"Inter, sans-serif", color:"rgba(156,163,175,0.5)", margin:0 }}>
                        {isLogin ? "Don\'t have an account? " : "Already have an account? "}
                        <button onClick={()=>{setIsLogin(!isLogin);setError("");}}
                                style={{ fontFamily:"Space Grotesk, sans-serif", color:"#2dd4bf", fontWeight:600, background:"none", border:"none", cursor:"pointer", fontSize:"0.875rem" }}
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

'''

# Replace lines auth_start through pipeline_start-1
new_lines = lines[:auth_start] + [new_component] + lines[pipeline_start:]

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("SUCCESS. Total lines:", len(new_lines))
