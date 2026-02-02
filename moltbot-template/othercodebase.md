# Clone and enter directory
git clone https://github.com/kelkalot/moltbook-observatory.git
cd moltbook-observatory

# Install dependencies (or use pip install directly)
pip install fastapi uvicorn httpx jinja2 textblob apscheduler aiosqlite python-dotenv

# Configure your API key
cp .env.example .env
# Edit .env and set MOLTBOOK_API_KEY=your_key_here