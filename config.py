"""Central configuration for the marketing roundtable."""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Google Gemini (embeddings) ─────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = "gemini-embedding-2"

# ── NVIDIA NIM (OpenAI-compatible) ────────────────────────────────────────
# Get a free key at https://build.nvidia.com  (starts with "nvapi-")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# Nemotron-3-Super-120B-A12B: used for both creative voices and JSON bookkeeping.
# Override via env vars if needed.
MODEL_LARGE = os.getenv("NVIDIA_MODEL_LARGE", "nvidia/nemotron-3-super-120b-a12b")
MODEL_SMALL = os.getenv("NVIDIA_MODEL_SMALL", "nvidia/nemotron-3-super-120b-a12b")

# Seconds to sleep between LLM calls (free tier is ~40 req/min; 0 is usually fine
# because the debate is sequential anyway)
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0"))

# ── Debate shape ───────────────────────────────────────────────────────────
NUM_PERSONAS = int(os.getenv("NUM_PERSONAS", "4"))   # experts at the table
MAX_CLASH_ROUNDS = int(os.getenv("MAX_CLASH_ROUNDS", "2"))
DEFAULT_FORMATS = ["reel", "linkedin", "ad_copy"]

# ── Token caps per role (output budget) ────────────────────────────────────
CAP_STYLE_CARD = 1500
CAP_PITCH = 1500
CAP_CLASH = 1500
CAP_MODERATOR = 2000
CAP_SYNTHESIS = 3000
CAP_FORMAT = 2500

# ── Neo4j ──────────────────────────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Local persona markdown fallback (used when Neo4j is unreachable)
PERSONA_DIR = os.path.join(
    os.path.dirname(__file__), "Copywrite Research", "Copywrite Research"
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache", "persona_cards")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
