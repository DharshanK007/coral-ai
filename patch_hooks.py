
import re

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove the hooks from the bottom
bad_hooks = '''    const [currentPage, setCurrentPage] = useState('home');

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [currentPage]);'''

content = content.replace(bad_hooks, '')

# 2. Add the hooks to the top of App Root
app_root_start = '''const App = () => {
    const [loaded, setLoaded]         = useState(false);
    const [lastUpdate, setLastUpdate] = useState(Date.now());
    const [user, setUser]             = useState(null);
    const [showPipeline, setShowPipeline] = useState(false);'''

app_root_fixed = '''const App = () => {
    const [loaded, setLoaded]         = useState(false);
    const [lastUpdate, setLastUpdate] = useState(Date.now());
    const [user, setUser]             = useState(null);
    const [showPipeline, setShowPipeline] = useState(false);
    const [currentPage, setCurrentPage] = useState('home');

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [currentPage]);'''

content = content.replace(app_root_start, app_root_fixed)

with open(r'c:\Users\Dharshan.K\OneDrive\Desktop\coral-\frontend\src\App.jsx', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed React hook ordering!")
