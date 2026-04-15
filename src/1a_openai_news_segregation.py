"""
Production-Grade Financial News NER Pipeline using OpenAI API
Features: Real-time checkpointing, resume capability, exponential backoff, failure tracking
Processes 300k+ articles with memory-efficient chunked readings and async API calls
"""

import os
import random
import re
import gc
import json
import signal
import asyncio
from typing import List, Dict, Optional, Set
from collections import defaultdict
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError, APIStatusError, APIConnectionError
import aiofiles

# Load environment variables from .env file
from pathlib import Path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Configuration
CHUNK_SIZE = 1000  # Process CSV in chunks
CONCURRENCY_LIMIT = 100  # Concurrent API requests
MAX_RETRIES = 5  # Maximum retry attempts for API calls
INPUT_CSV = "dataset/processed_news_dataset.csv"
OUTPUT_CSV = "dataset/tier_segregated_news.csv"
CHECKPOINT_FILE = "dataset/news_segregation_checkpoints/raw_responses.jsonl"
FAILED_ROWS_FILE = "dataset/news_segregation_checkpoints/failed_rows.txt"
MAPPING_CACHE_FILE = "dataset/news_segregation_checkpoints/sector_mapping_cache.json"
SECTOR_GPT_BATCH_SIZE = 100
SECTOR_GPT_MODEL = "gpt-4o-mini"
OTHERS_SECTOR = "Others"

# Nifty 50 Companies with aliases and sectors
NIFTY_50_DATA = {
    # IT Sector
    "TCS": {"aliases": ["tcs", "tata consultancy", "tata consultancy services"], "sector": "IT"},
    "INFY": {"aliases": ["infosys", "infy"], "sector": "IT"},
    "HCLTECH": {"aliases": ["hcl tech", "hcl technologies", "hcltech"], "sector": "IT"},
    "WIPRO": {"aliases": ["wipro"], "sector": "IT"},
    "TECHM": {"aliases": ["tech mahindra", "techm"], "sector": "IT"},
    "LTIM": {"aliases": ["ltimindtree", "lti mindtree", "ltim", "l&t infotech"], "sector": "IT"},
    
    # Banking Sector
    "HDFCBANK": {"aliases": ["hdfc bank", "hdfcbank"], "sector": "Banking"},
    "ICICIBANK": {"aliases": ["icici bank", "icicibank"], "sector": "Banking"},
    "SBIN": {"aliases": ["sbi", "state bank of india", "state bank"], "sector": "Banking"},
    "AXISBANK": {"aliases": ["axis bank", "axisbank"], "sector": "Banking"},
    "KOTAKBANK": {"aliases": ["kotak mahindra bank", "kotak bank", "kotakbank"], "sector": "Banking"},
    "INDUSINDBK": {"aliases": ["indusind bank", "indusindbk"], "sector": "Banking"},
    "BANKBARODA": {"aliases": ["bank of baroda", "bob", "bankbaroda"], "sector": "Banking"},
    "PNB": {"aliases": ["punjab national bank", "pnb"], "sector": "Banking"},
    "CANBK": {"aliases": ["canara bank", "canbk"], "sector": "Banking"},
    
    # Financial Services
    "BAJFINANCE": {"aliases": ["bajaj finance", "bajfinance"], "sector": "Financial Services"},
    "BAJAJFINSV": {"aliases": ["bajaj finserv", "bajajfinsv"], "sector": "Financial Services"},
    "HDFCLIFE": {"aliases": ["hdfc life", "hdfclife"], "sector": "Financial Services"},
    "SBILIFE": {"aliases": ["sbi life", "sbilife"], "sector": "Financial Services"},
    
    # Auto Sector
    "MARUTI": {"aliases": ["maruti suzuki", "maruti", "maruti suzuki india"], "sector": "Auto"},
    "M&M": {"aliases": ["mahindra & mahindra", "mahindra", "m&m", "m and m"], "sector": "Auto"},
    "TATAMOTORS": {"aliases": ["tata motors", "tatamotors"], "sector": "Auto"},
    "BAJAJ-AUTO": {"aliases": ["bajaj auto", "bajaj-auto"], "sector": "Auto"},
    "EICHERMOT": {"aliases": ["eicher motors", "eicher", "royal enfield"], "sector": "Auto"},
    "HEROMOTOCO": {"aliases": ["hero motocorp", "hero", "heromotoco"], "sector": "Auto"},
    
    # FMCG Sector
    "HINDUNILVR": {"aliases": ["hindustan unilever", "hul", "hindunilvr"], "sector": "FMCG"},
    "ITC": {"aliases": ["itc", "itc limited"], "sector": "FMCG"},
    "NESTLEIND": {"aliases": ["nestle india", "nestle", "nestleind"], "sector": "FMCG"},
    "BRITANNIA": {"aliases": ["britannia", "britannia industries"], "sector": "FMCG"},
    "DABUR": {"aliases": ["dabur", "dabur india"], "sector": "FMCG"},
    "TATACONSUM": {"aliases": ["tata consumer", "tata consumer products", "tataconsum"], "sector": "FMCG"},
    
    # Pharma Sector
    "SUNPHARMA": {"aliases": ["sun pharma", "sun pharmaceutical", "sunpharma"], "sector": "Pharma"},
    "DRREDDY": {"aliases": ["dr reddy", "dr reddy's", "drreddys", "drreddy"], "sector": "Pharma"},
    "CIPLA": {"aliases": ["cipla"], "sector": "Pharma"},
    "DIVISLAB": {"aliases": ["divi's lab", "divis lab", "divislab"], "sector": "Pharma"},
    
    # Energy & Power
    "RELIANCE": {"aliases": ["reliance", "reliance industries", "ril"], "sector": "Energy"},
    "ONGC": {"aliases": ["ongc", "oil and natural gas"], "sector": "Energy"},
    "BPCL": {"aliases": ["bpcl", "bharat petroleum"], "sector": "Energy"},
    "IOC": {"aliases": ["ioc", "indian oil", "indian oil corp"], "sector": "Energy"},
    "NTPC": {"aliases": ["ntpc"], "sector": "Power"},
    "POWERGRID": {"aliases": ["power grid", "powergrid"], "sector": "Power"},
    
    # Metals & Mining
    "TATASTEEL": {"aliases": ["tata steel", "tatasteel"], "sector": "Metals"},
    "HINDALCO": {"aliases": ["hindalco", "hindalco industries"], "sector": "Metals"},
    "JSWSTEEL": {"aliases": ["jsw steel", "jswsteel"], "sector": "Metals"},
    "COALINDIA": {"aliases": ["coal india", "coalindia"], "sector": "Metals"},
    
    # Telecom
    "BHARTIARTL": {"aliases": ["bharti airtel", "airtel", "bhartiartl"], "sector": "Telecom"},
    
    # Cement
    "ULTRACEMCO": {"aliases": ["ultratech cement", "ultratech", "ultracemco"], "sector": "Cement"},
    "GRASIM": {"aliases": ["grasim", "grasim industries"], "sector": "Cement"},
    
    # Infrastructure
    "LT": {"aliases": ["l&t", "larsen & toubro", "larsen and toubro", "lt"], "sector": "Infrastructure"},
    "ADANIPORTS": {"aliases": ["adani ports", "adaniports", "apsez"], "sector": "Infrastructure"},
    "ADANIENT": {
        "aliases": ["adani enterprises", "adani enterprises ltd", "adani enterprises ltd.", "adani ent", "ael"],
        "sector": "Infrastructure",
    },
    
    # Others
    "ASIANPAINT": {"aliases": ["asian paints", "asianpaint"], "sector": "Paints"},
    "TITAN": {"aliases": ["titan", "titan company"], "sector": "Retail"},
    "APOLLOHOSP": {"aliases": ["apollo hospital", "apollo hospitals", "apollohosp"], "sector": "Healthcare"},
    "INDIGO": {"aliases": ["indigo", "interglobe aviation"], "sector": "Aviation"},
}

# Sector keywords (Updated for Indian Market Jargon)
SECTOR_KEYWORDS = {
    "IT": ["it sector", "software", "technology", "digital", "cloud computing", "ai", "automation", "deal win", "tech spend"],
    "Banking": ["banking", "banks", "npa", "lending", "deposits", "credit", "casa", "psu bank", "bad loans", "nbfc"],
    "Financial Services": ["insurance", "mutual fund", "financial services", "amc", "life insurance"],
    "Auto": ["automobile", "automotive", "vehicle", "cars", "ev", "electric vehicle", "two-wheeler", "commercial vehicle", "tractor", "sales data", "auto sales"],
    "FMCG": ["fmcg", "consumer goods", "rural demand", "staples", "rural volume", "consumption"],
    "Pharma": ["pharmaceutical", "pharma", "drug", "medicine", "healthcare", "fda", "usfda", "api", "generics", "clinical trials"],
    "Energy": ["oil", "gas", "petroleum", "crude", "refinery", "brent", "o&m"],
    "Power": ["power", "electricity", "renewable energy", "solar", "wind energy", "discom", "power generation"],
    "Metals": ["steel", "aluminium", "aluminum", "copper", "metals", "mining", "iron ore", "base metals"],
    "Telecom": ["telecom", "telecommunication", "5g", "spectrum", "arpu", "telecom tariffs", "broadband"],
    "Cement": ["cement", "construction material", "clinker", "capacity expansion"],
    "Infrastructure": ["infrastructure", "construction", "ports", "highways", "capex", "nhai", "toll"],
    "Aviation": ["aviation", "airlines", "passenger traffic", "cargo traffic", "load factor"],
}

# Global/Macro keywords (Updated with critical Indian & Macro triggers)
GLOBAL_KEYWORDS = [
    # Central Banks & Rates
    "rbi", "reserve bank", "repo rate", "monetary policy", "interest rate", "fed", "federal reserve", "fomc", "ecb", "bond yield",
    
    # Economy & Data
    "budget", "fiscal policy", "gst", "tax", "inflation", "cpi", "wpi", "gdp", "economic growth", "recession", "pmi", "iip", "manufacturing pmi",
    
    # Markets & Flows
    "us market", "dow jones", "nasdaq", "s&p 500", "wall street", "nifty", "sensex", "nse", "bse", "sebi", "stock market", "foreign investment", "fii", "fpi", "dii", "fdi", "market crash", "market rally",
    
    # Commodities & Currency
    "crude oil", "brent crude", "gold price", "dollar", "rupee", "inr", "usd", "opec",
    
    # Geopolitics & Events
    "covid", "pandemic", "lockdown", "ukraine", "russia", "china", "geopolitical", "trade war", "tariff", "sanctions", "war", "monsoon", "elections"
]

VALID_SECTORS: Set[str] = set(SECTOR_KEYWORDS.keys()) | {OTHERS_SECTOR}
STANDARD_SECTOR_LOWER: Dict[str, str] = {s.lower(): s for s in VALID_SECTORS}




class FinancialNERPipeline:
    def __init__(self):
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found!\n"
                "Please create a .env file in the project root with:\n"
                "OPENAI_API_KEY=sk-your-key-here"
            )
        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.shutdown_event = asyncio.Event()  # Signal for graceful Ctrl+C shutdown
        self.write_lock = asyncio.Lock()  # Thread-safe file writing
        self.failed_lock = asyncio.Lock()  # Thread-safe failure tracking
        
        # Build comprehensive regex patterns
        self.company_patterns = self._build_company_patterns()
        self.sector_patterns = self._build_sector_patterns()
        self.global_patterns = self._build_global_patterns()
        
        # Mapping for company name to ticker
        self.company_to_ticker = self._build_company_mapping()
        
        # Mapping for sector to tickers
        self.sector_to_tickers = self._build_sector_mapping()
        
        # All tickers for global news
        self.all_tickers = list(NIFTY_50_DATA.keys())

        # Adani-specific mention repair for CSV-only aggregation.
        # This fixes legacy checkpoint rows where "Adani Enterprises" was keyed as ADANIPORTS/LT.
        self.adani_enterprises_pattern = re.compile(
            r"\b(?:adani enterprises(?:\s+ltd\.?)?|adani ent|ael)\b",
            re.IGNORECASE,
        )
        self.adani_ports_pattern = re.compile(
            r"\b(?:adani ports(?:\s*(?:&|and)\s*sez|\s+sez)?|apsez)\b",
            re.IGNORECASE,
        )
        
        # Track processed row indices for resume capability
        self.processed_indices: Set[int] = set()
        
    def _build_company_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for all company aliases"""
        patterns = []
        for ticker, data in NIFTY_50_DATA.items():
            for alias in data["aliases"]:
                # Case-insensitive word boundary matching
                pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
                patterns.append(pattern)
        return patterns
    
    def _build_sector_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for sector keywords"""
        patterns = []
        for sector, keywords in SECTOR_KEYWORDS.items():
            for keyword in keywords:
                pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                patterns.append(pattern)
        return patterns
    
    def _build_global_patterns(self) -> List[re.Pattern]:
        """Build regex patterns for global/macro keywords"""
        patterns = []
        for keyword in GLOBAL_KEYWORDS:
            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
            patterns.append(pattern)
        return patterns
    
    def _build_company_mapping(self) -> Dict[str, str]:
        """Build mapping from company names to tickers"""
        mapping = {}
        for ticker, data in NIFTY_50_DATA.items():
            for alias in data["aliases"]:
                mapping[alias.lower()] = ticker
        return mapping
    
    def _build_sector_mapping(self) -> Dict[str, List[str]]:
        """Build mapping from sector to list of tickers"""
        mapping = defaultdict(list)
        for ticker, data in NIFTY_50_DATA.items():
            mapping[data["sector"]].append(ticker)
        return dict(mapping)
    
    def is_relevant(self, text: str) -> bool:
        """Check if article contains any financial keywords"""
        if not text or pd.isna(text):
            return False
        
        # Check company patterns
        for pattern in self.company_patterns:
            if pattern.search(text):
                return True
        
        # Check sector patterns
        for pattern in self.sector_patterns:
            if pattern.search(text):
                return True
        
        # Check global patterns
        for pattern in self.global_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def _get_system_prompt(self) -> str:
        """Generate few-shot system prompt for OpenAI API"""
        return """You are a financial news analyzer. Extract information into exactly 3 categories and return valid JSON.

Categories:
1. "direct": Companies explicitly mentioned with their impact
2. "sectoral": Industry sectors or supply chains affected
3. "global": Macro/geopolitical events affecting markets

Rules:
- Return ONLY valid JSON with these 3 keys
- If a category has no information, return empty array []
- Be concise in summaries (1 to 2 sentences max)
- Extract ALL relevant information

Example 1:
Input: "TCS reported strong Q3 growth with 15% YoY increase. Tech sector benefits from digital transformation."
Output:
{
  "direct": [{"company": "TCS", "summary": "TCS reports strong Q3 growth with 15% YoY increase"}],
  "sectoral": [{"sector": "IT", "summary": "Tech sector benefits from digital transformation initiatives"}],
  "global": []
}

Example 2:
Input: "RBI increases repo rate by 25 bps to combat inflation. This will impact banking sector lending."
Output:
{
  "direct": [],
  "sectoral": [{"sector": "Banking", "summary": "RBI increases repo rate by 25 bps, impacting lending rates"}],
  "global": [{"event": "Monetary Policy", "summary": "RBI hikes repo rate to combat inflation"}]
}

Example 3:
Input: "Crude oil prices surge 5% amid geopolitical tensions. Energy stocks rally."
Output:
{
  "direct": [],
  "sectoral": [{"sector": "Energy", "summary": "Energy stocks rally on crude price surge"}],
  "global": [{"event": "Crude Oil", "summary": "Crude oil prices surge 5% amid geopolitical tensions"}]
}

Example 4:
Input: "Reliance Industries announces expansion in retail business. Maruti Suzuki launches new EV model."
Output:
{
  "direct": [
    {"company": "Reliance", "summary": "Reliance Industries announces expansion in retail business"},
    {"company": "Maruti", "summary": "Maruti Suzuki launches new EV model"}
  ],
  "sectoral": [{"sector": "Auto", "summary": "Auto sector embraces EV transition"}],
  "global": []
}

Now analyze the following news article and return only the JSON output:"""
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Aggressively sanitize text to prevent 400 JSON parse errors"""
        if not text:
            return ""
        # Strip invalid UTF-8 bytes and unpaired surrogates
        text = text.encode("utf-8", "ignore").decode("utf-8")
        # Keep only standard printable characters, newlines, and tabs
        text = re.sub(r'[^\x20-\x7E\n\t]', ' ', text)
        # Collapse excessive whitespace
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    async def extract_entities(self, text: str) -> Optional[Dict]:
        """Extract entities using OpenAI API with exponential backoff retry logic"""
        # Sanitize text before sending to API
        text = self.sanitize_text(text)
        if not text:
            return {"direct": [], "sectoral": [], "global": []}
        
        empty_retry_limit = 2  # Empty responses: retry 2x (rate-limit recovery), not 5x
        
        for attempt in range(MAX_RETRIES):
            # Acquire semaphore only for the API call, release during retry sleep
            async with self.semaphore:
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": text}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.2,
                        max_tokens=2000
                    )
                    
                    # Check for empty response content
                    content = response.choices[0].message.content
                    if not content or not content.strip():
                        if attempt < empty_retry_limit:
                            pass  # Will sleep AFTER releasing semaphore below
                        else:
                            return {"direct": [], "sectoral": [], "global": []}
                    else:
                        try:
                            result = json.loads(content)
                        except json.JSONDecodeError:
                            if attempt < empty_retry_limit:
                                pass  # Will sleep AFTER releasing semaphore below
                            else:
                                return {"direct": [], "sectoral": [], "global": []}
                        else:
                            # Validate structure
                            if not all(key in result for key in ["direct", "sectoral", "global"]):
                                return {"direct": [], "sectoral": [], "global": []}
                            return result
                    
                    # Empty/malformed — need retry, sleep OUTSIDE semaphore
                    # (falls through to the sleep below)
                
                except RateLimitError as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"Rate limit persisted after {MAX_RETRIES} attempts")
                        return None
                    wait_time = (2 ** (attempt + 1)) + 5 + random.uniform(0, 3)
                    print(f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). Backing off {wait_time:.1f}s...")
                    # sleep happens after semaphore release (below)
                
                except APIStatusError as e:
                    if e.status_code == 400:
                        print(f"Bad request (400) for article — skipping. Error: {str(e)[:100]}")
                        return None
                    elif e.status_code in (429, 500, 502, 503, 529):
                        if attempt == MAX_RETRIES - 1:
                            return None
                        wait_time = (2 ** attempt) + 3 + random.uniform(0, 2)
                        print(f"Server error {e.status_code} (attempt {attempt + 1}/{MAX_RETRIES}). Retrying in {wait_time:.1f}s...")
                    else:
                        print(f"Unhandled API error {e.status_code}: {str(e)[:100]}")
                        return None
                
                except APIConnectionError as e:
                    if attempt >= MAX_RETRIES - 1:
                        print(f"Connection failed after {MAX_RETRIES} attempts: {str(e)[:80]}")
                        return None
                    wait_time = (2 ** attempt) + 1 + random.uniform(0, 1)
                    print(f"Connection error (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)[:80]}. Retrying in {wait_time:.1f}s...")
                
                except Exception as e:
                    if attempt >= MAX_RETRIES - 1:
                        print(f"Failed after {MAX_RETRIES} attempts: {type(e).__name__}: {str(e)[:80]}")
                        return None
                    wait_time = (2 ** attempt) + 1 + random.uniform(0, 1)
                    print(f"Unexpected error (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:80]}. Retrying in {wait_time:.1f}s...")
            
            # --- Semaphore is RELEASED here --- retry sleep doesn't block other requests
            wait_time = (2 ** attempt) + 1 + random.uniform(0, 1)
            await asyncio.sleep(wait_time)
    
    def normalize_extraction(self, extraction: Dict) -> Dict:
        """
        Convert OpenAI extraction to compact normalized format.
        Returns: {
            "direct": {"TCS": ["news1"], "INFY": ["news2"]},
            "sectoral": {"Banking": ["news3"], "IT": ["news4"]},
            "global": {"RBI Rate": ["news5"]}
        }
        """
        normalized = {
            "direct": defaultdict(list),
            "sectoral": defaultdict(list),
            "global": defaultdict(list)
        }
        
        # Process direct company mentions - map company names to tickers
        for item in extraction.get("direct", []):
            company = item.get("company", "").lower()
            summary = item.get("summary", "")
            
            # Find matching ticker
            ticker = None
            for alias, tick in self.company_to_ticker.items():
                if alias in company or company in alias:
                    ticker = tick
                    break
            
            if ticker and summary:
                normalized["direct"][ticker].append(summary)
        
        # Process sectoral news - keep sector names as keys (don't expand yet)
        for item in extraction.get("sectoral", []):
            sector = item.get("sector", "")
            summary = item.get("summary", "")
            
            if sector and summary:
                normalized["sectoral"][sector].append(summary)
        
        # Process global news - keep event names as keys (don't expand yet)
        for item in extraction.get("global", []):
            event = item.get("event", "") or item.get("summary", "")[:50]  # Use event or first 50 chars
            summary = item.get("summary", "")
            
            if summary:
                normalized["global"][event].append(summary)
        
        # Convert defaultdicts to regular dicts
        return {
            "direct": dict(normalized["direct"]),
            "sectoral": dict(normalized["sectoral"]),
            "global": dict(normalized["global"])
        }
    
    async def write_checkpoint(self, row_index: int, date: str, extracted_news: Dict, skipped: bool = False):
        """True asynchronous thread-safe write to checkpoint file"""
        async with self.write_lock:
            async with aiofiles.open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
                checkpoint_data = {
                    "row_index": row_index,
                    "date": date,
                    "extracted_news": extracted_news
                }
                if skipped:
                    checkpoint_data["skipped"] = True
                await f.write(json.dumps(checkpoint_data) + "\n")
    
    async def write_failed_row(self, row_index: int, reason: str = "unknown"):
        """True asynchronous thread-safe write to failed rows file — called IMMEDIATELY on failure"""
        async with self.failed_lock:
            async with aiofiles.open(FAILED_ROWS_FILE, "a", encoding="utf-8") as f:
                await f.write(f"{row_index}|{reason}\n")
    
    async def process_article(self, row_index: int, row: pd.Series) -> bool:
        """Process a single article with checkpointing and failure tracking"""
        try:
            # Skip if already processed (successfully checkpointed)
            if row_index in self.processed_indices:
                return True
            
            # Check if shutdown was requested
            if self.shutdown_event.is_set():
                return False
            
            date = row["date"]
            text = f"{row.get('title', '')} {row.get('news', '')}"
            
            # Pre-filter: if not relevant, checkpoint as skipped so we never re-process
            if not self.is_relevant(text):
                empty_news = {"direct": {}, "sectoral": {}, "global": {}}
                await self.write_checkpoint(row_index, date, empty_news, skipped=True)
                return True  # Not relevant, but not a failure
            
            # Extract entities via OpenAI with retries
            extraction = await self.extract_entities(text)
            
            if extraction is None:
                # Failed after all retries - log to failed_rows.txt IMMEDIATELY
                await self.write_failed_row(row_index, reason="api_failure")
                return False
            
            # Normalize to compact format (defer ticker expansion to aggregation)
            extracted_news = self.normalize_extraction(extraction)
            
            # Immediately checkpoint the successful result in compact format
            await self.write_checkpoint(row_index, date, extracted_news)
            
            return True
            
        except Exception as e:
            print(f"Error processing article {row_index}: {str(e)}")
            await self.write_failed_row(row_index, reason=f"exception:{type(e).__name__}")
            return False
    
    async def process_chunk(self, chunk: pd.DataFrame, pbar) -> int:
        """Process a chunk of articles asynchronously, cancellable on shutdown"""
        tasks = []
        for row_index, row in chunk.iterrows():
            task = asyncio.ensure_future(self.process_article(row_index, row))
            tasks.append(task)
        
        success_count = 0
        for coro in asyncio.as_completed(tasks):
            try:
                success = await coro
                if success:
                    success_count += 1
            except asyncio.CancelledError:
                pass
            pbar.update(1)
            
            # If shutdown requested, cancel remaining tasks
            if self.shutdown_event.is_set():
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
        
        return success_count
    
    def load_processed_indices(self) -> Set[int]:
        """Load successfully processed row indices from checkpoint file only"""
        processed = set()
        skipped_count = 0
        if os.path.exists(CHECKPOINT_FILE):
            print(f"[Resume] Loading checkpoint from {CHECKPOINT_FILE}...")
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        processed.add(data["row_index"])
                        if data.get("skipped"):
                            skipped_count += 1
                    except:
                        continue
            print(f"[Resume] Found {len(processed):,} already processed articles ({skipped_count:,} skipped as irrelevant)")
        return processed
    
    def load_failed_indices(self) -> Set[int]:
        """Load failed row indices from failed_rows file for retry"""
        failed = set()
        if os.path.exists(FAILED_ROWS_FILE):
            with open(FAILED_ROWS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row_idx = int(line.split("|")[0])
                        failed.add(row_idx)
                    except:
                        continue
        return failed

    # ----------------------- Sector normalization (integrated from 1c) -----------------------

    def keyword_match_sector(self, label: str) -> Optional[str]:
        label_lower = (label or "").lower()
        matches: Dict[str, int] = {}

        for sector, keywords in SECTOR_KEYWORDS.items():
            for kw in keywords:
                if kw in label_lower:
                    if sector not in matches or len(kw) > matches[sector]:
                        matches[sector] = len(kw)

        if not matches:
            return None
        return max(matches, key=lambda s: matches[s])

    def canonical_sector(self, label: str) -> Optional[str]:
        if label in VALID_SECTORS:
            return label
        ci = STANDARD_SECTOR_LOWER.get((label or "").lower())
        if ci:
            return ci
        return self.keyword_match_sector(label)

    def load_mapping_cache(self) -> Dict[str, str]:
        if os.path.exists(MAPPING_CACHE_FILE):
            try:
                with open(MAPPING_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                if isinstance(cache, dict):
                    clean_cache: Dict[str, str] = {}
                    for k, v in cache.items():
                        if not isinstance(k, str) or not isinstance(v, str):
                            continue
                        mapped = v if v in VALID_SECTORS else STANDARD_SECTOR_LOWER.get(v.lower(), OTHERS_SECTOR)
                        clean_cache[k] = mapped
                    return clean_cache
            except Exception:
                pass
        return {}

    def save_mapping_cache(self, cache: Dict[str, str]) -> None:
        os.makedirs(os.path.dirname(MAPPING_CACHE_FILE), exist_ok=True)
        with open(MAPPING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def fix_cache_with_keywords(self, cache: Dict[str, str]) -> int:
        corrected = 0
        for label in list(cache.keys()):
            if cache[label] == OTHERS_SECTOR:
                match = self.keyword_match_sector(label)
                if match:
                    cache[label] = match
                    corrected += 1
        return corrected

    def collect_unknown_sectors(self, cache: Dict[str, str]) -> Set[str]:
        unknown: Set[str] = set()
        if not os.path.exists(CHECKPOINT_FILE):
            return unknown

        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sectoral = data.get("extracted_news", {}).get("sectoral", {})
                    for label in sectoral.keys():
                        if self.canonical_sector(label) is None and label not in cache:
                            unknown.add(label)
                except Exception:
                    continue
        return unknown

    def build_gpt_mapping_prompt(self, unknown_labels: List[str]) -> str:
        numbered_labels = "\n".join(f"{i + 1}. {label}" for i, label in enumerate(unknown_labels))
        valid = ", ".join(sorted(VALID_SECTORS, key=lambda x: (x == OTHERS_SECTOR, x)))
        return (
            "Map each label to exactly one valid sector.\n"
            f"Valid sectors: {valid}\n"
            "Return ONLY a JSON object where keys are original labels and values are valid sectors.\n"
            "Prefer a real sector over Others whenever possible.\n\n"
            f"Labels:\n{numbered_labels}\n"
        )

    def call_gpt_for_mapping(self, unknown_labels: List[str]) -> Dict[str, str]:
        if not unknown_labels:
            return {}
        prompt = self.build_gpt_mapping_prompt(unknown_labels)
        try:
            response = self.sync_client.chat.completions.create(
                model=SECTOR_GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a JSON-only sector classifier for Indian financial markets. "
                            "Return only the requested JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            raw_mapping = json.loads(content) if content else {}
            validated: Dict[str, str] = {}
            for label, mapped in (raw_mapping or {}).items():
                if not isinstance(label, str) or not isinstance(mapped, str):
                    continue
                if mapped in VALID_SECTORS:
                    validated[label] = mapped
                else:
                    validated[label] = STANDARD_SECTOR_LOWER.get(mapped.lower(), OTHERS_SECTOR)
            return validated
        except Exception as e:
            print(f"[Normalization] GPT mapping failed: {type(e).__name__}: {e}")
            return {}

    def prepare_sector_mapping_cache(self) -> Dict[str, str]:
        cache = self.load_mapping_cache()
        fixed = self.fix_cache_with_keywords(cache)
        if fixed:
            self.save_mapping_cache(cache)
            print(f"[Normalization] Fixed {fixed:,} cache entries locally.")

        unknown_labels = sorted(self.collect_unknown_sectors(cache))
        if not unknown_labels:
            return cache

        total_batches = (len(unknown_labels) + SECTOR_GPT_BATCH_SIZE - 1) // SECTOR_GPT_BATCH_SIZE
        print(f"[Normalization] {len(unknown_labels):,} labels unresolved locally → {total_batches} GPT batches")

        for batch_num, i in enumerate(range(0, len(unknown_labels), SECTOR_GPT_BATCH_SIZE), start=1):
            batch = unknown_labels[i : i + SECTOR_GPT_BATCH_SIZE]
            print(f"[Normalization] Batch {batch_num}/{total_batches} ({len(batch)} labels)")
            mapped = self.call_gpt_for_mapping(batch)
            if not mapped:
                print("[Normalization] Empty mapping batch. Stopping further GPT mapping to avoid cache corruption.")
                break
            cache.update(mapped)
            for label in batch:
                cache.setdefault(label, OTHERS_SECTOR)
            self.save_mapping_cache(cache)

        return cache

    def normalize_sectoral_dict(self, sectoral: Dict[str, List[str]], cache: Dict[str, str]) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = defaultdict(list)
        for label, summaries in (sectoral or {}).items():
            target = self.canonical_sector(label)
            if target is None:
                target = cache.get(label, label)
            normalized[target].extend(list(summaries))
        return dict(normalized)

    def _extract_adani_tickers_from_text(self, text: str) -> Set[str]:
        """
        Detect explicit ADANI ticker mentions from free text.
        Used during CSV aggregation to repair historical checkpoint mis-mappings.
        """
        if not text:
            return set()

        matches: Set[str] = set()
        if self.adani_enterprises_pattern.search(text):
            matches.add("ADANIENT")
        if self.adani_ports_pattern.search(text):
            matches.add("ADANIPORTS")

        return {ticker for ticker in matches if ticker in self.all_tickers}

    def _repair_direct_targets(self, original_ticker: str, summary: str) -> List[str]:
        """
        Keep existing ticker mapping by default, but override when summary text
        explicitly names Adani Enterprises/Adani Ports.
        """
        repaired = self._extract_adani_tickers_from_text(summary)
        if repaired:
            return sorted(repaired)
        return [original_ticker]
    
    def build_final_csv(self):
        """
        Build final aggregated CSV from compact checkpoint file.
        Optimized for memory efficiency to prevent laptop crashes (OOM) on large datasets.
        """
        print("\n[Aggregation] Building final CSV from checkpoint file...")
        
        if not os.path.exists(CHECKPOINT_FILE):
            print("[Error] No checkpoint file found. Nothing to aggregate.")
            return

        mapping_cache = self.prepare_sector_mapping_cache()
        
        # 1. Group raw news by date first (uses minimal memory compared to expanding all tickers)
        date_to_news = defaultdict(list)
        total_valid = 0
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("skipped"):
                        continue
                    date_to_news[data["date"]].append(data["extracted_news"])
                    total_valid += 1
                except Exception:
                    continue
        
        # 2. Process one date at a time and write immediately to CSV
        import csv
        total_rows_written = 0
        
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=["date", "symbol", "direct_news", "sectoral_news", "global_news"])
            writer.writeheader()
            
            for date, news_list in date_to_news.items():
                # Accumulator for just this single date
                tickers_data = defaultdict(lambda: {"direct": [], "sectoral": [], "global": []})
                
                for extracted_news in news_list:
                    # Direct news
                    for ticker, summaries in extracted_news.get("direct", {}).items():
                        for summary in summaries:
                            for repaired_ticker in self._repair_direct_targets(ticker, summary):
                                tickers_data[repaired_ticker]["direct"].append(summary)
                    
                    # Sectoral news
                    normalized_sectoral = self.normalize_sectoral_dict(
                        extracted_news.get("sectoral", {}),
                        mapping_cache,
                    )
                    for sector_name, summaries in normalized_sectoral.items():
                        # Always route sectoral summaries to explicitly mentioned Adani tickers
                        # (helps with "Others" / legacy-sector labels).
                        for summary in summaries:
                            for adani_ticker in self._extract_adani_tickers_from_text(summary):
                                tickers_data[adani_ticker]["sectoral"].append(summary)

                        if sector_name == "Others":
                            continue

                        matched_tickers = self.sector_to_tickers.get(sector_name, [])
                        if not matched_tickers:
                            for sect, keywords in SECTOR_KEYWORDS.items():
                                if sect == "Others":
                                    continue
                                if any(keyword.lower() in sector_name.lower() for keyword in keywords) or \
                                   sect.lower() in sector_name.lower():
                                    matched_tickers.extend(self.sector_to_tickers.get(sect, []))

                        for ticker in set(matched_tickers):
                            tickers_data[ticker]["sectoral"].extend(summaries)
                    
                    # Global news
                    for event_name, summaries in extracted_news.get("global", {}).items():
                        for ticker in self.all_tickers:
                            tickers_data[ticker]["global"].extend(summaries)
                
                # Write this date's rows
                for ticker, news in tickers_data.items():
                    writer.writerow({
                        "date": date,
                        "symbol": ticker,
                        "direct_news": " || ".join(news["direct"]) if news["direct"] else "",
                        "sectoral_news": " || ".join(news["sectoral"]) if news["sectoral"] else "",
                        "global_news": " || ".join(news["global"]) if news["global"] else ""
                    })
                    total_rows_written += 1
        
        print(f"[Aggregation] Final CSV saved to: {OUTPUT_CSV}")
        print(f"[Aggregation] Total unique Date/Ticker combinations written: {total_rows_written:,}")
        
        return None
    
    def remove_from_failed_file(self, row_index: int):
        """Remove a single row from failed_rows.txt (called synchronously after successful retry)"""
        if not os.path.exists(FAILED_ROWS_FILE):
            return
        with open(FAILED_ROWS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(FAILED_ROWS_FILE, "w", encoding="utf-8") as f:
            for line in lines:
                try:
                    idx = int(line.strip().split("|")[0])
                    if idx != row_index:
                        f.write(line)
                except:
                    f.write(line)
    
    async def process_failed_article(self, row_index: int, row: pd.Series) -> bool:
        """Process a failed article — on success, remove it from failed_rows.txt"""
        success = await self.process_article(row_index, row)
        if success:
            # Remove this row from failed_rows.txt individually
            self.remove_from_failed_file(row_index)
        return success
    
    async def process_failed_chunk(self, chunk: pd.DataFrame, pbar) -> int:
        """Process a chunk of failed articles, removing each from failed_rows.txt on success"""
        tasks = []
        for row_index, row in chunk.iterrows():
            task = asyncio.ensure_future(self.process_failed_article(row_index, row))
            tasks.append(task)
        
        success_count = 0
        for coro in asyncio.as_completed(tasks):
            try:
                success = await coro
                if success:
                    success_count += 1
            except asyncio.CancelledError:
                pass
            pbar.update(1)
            
            if self.shutdown_event.is_set():
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
        
        return success_count
    
    async def retry_failed_rows(self, full_df: pd.DataFrame):
        """Retry previously failed rows — removes individual rows from file on success"""
        failed_indices = self.load_failed_indices()
        # Remove any that have since been successfully processed
        failed_indices -= self.processed_indices
        
        if not failed_indices:
            return 0
        
        print(f"\n[Retry] Retrying {len(failed_indices):,} previously failed rows...")
        
        # Get the rows that need retry from the dataframe
        retry_df = full_df[full_df["original_index"].isin(failed_indices)].copy()
        retry_df = retry_df.set_index("original_index")
        
        if retry_df.empty:
            return 0
        
        success_count = 0
        with tqdm_asyncio(total=len(retry_df), desc="Retrying failed rows") as pbar:
            for start in range(0, len(retry_df), CHUNK_SIZE):
                if self.shutdown_event.is_set():
                    break
                chunk = retry_df.iloc[start:start + CHUNK_SIZE]
                count = await self.process_failed_chunk(chunk, pbar)
                success_count += count
                gc.collect()
        
        print(f"[Retry] Successfully recovered: {success_count:,} / {len(failed_indices):,}")
        return success_count
    
    async def process_dataset(self, build_csv: bool = True):
        """Process entire dataset with chunked reading, checkpointing, and graceful Ctrl+C"""
        print(f"=== Production-Grade Financial NER Pipeline ===")
        print(f"Input: {INPUT_CSV}")
        print(f"Checkpoint: {CHECKPOINT_FILE}")
        print(f"Failed Rows: {FAILED_ROWS_FILE}")
        print(f"Output: {OUTPUT_CSV}")
        print(f"Concurrency: {CONCURRENCY_LIMIT} requests")
        print(f"Max Retries: {MAX_RETRIES}")
        print(f"Chunk size: {CHUNK_SIZE} rows")
        
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Load previously processed indices for resume capability
        self.processed_indices = self.load_processed_indices()
        
        # Count total rows for progress bar
        total_rows = sum(1 for _ in open(INPUT_CSV)) - 1  # Exclude header
        remaining_rows = total_rows - len(self.processed_indices)
        print(f"Total articles: {total_rows:,}")
        print(f"Remaining to process: {remaining_rows:,}")
        
        if remaining_rows == 0:
            print("\n[Info] All articles already processed.")
            if build_csv:
                self.build_final_csv()
            else:
                print("[Info] Skipping CSV build (mode = NER only).")
            return
        
        # Load full dataset and reverse order: process from latest date to earliest
        full_df = pd.read_csv(INPUT_CSV, low_memory=False)
        full_df = full_df.iloc[::-1].reset_index(drop=False)  # Preserve original index as column
        full_df.rename(columns={"index": "original_index"}, inplace=True)
        
        # Set up signal handler: 1st Ctrl+C = graceful stop, 2nd = force exit
        loop = asyncio.get_running_loop()
        
        def handle_sigint():
            if self.shutdown_event.is_set():
                # Second Ctrl+C — force exit
                print("\n[Ctrl+C] Force exit! Progress already saved.")
                os._exit(1)
            print("\n\n[Ctrl+C] Graceful shutdown... cancelling in-flight tasks...")
            print("[Ctrl+C] Press Ctrl+C again to force exit immediately.")
            self.shutdown_event.set()
        
        try:
            loop.add_signal_handler(signal.SIGINT, handle_sigint)
        except NotImplementedError:
            pass
        
        # === PHASE 1: Process remaining rows first ===
        total_success = 0
        interrupted = False
        
        print(f"\n[Phase 1] Processing remaining {remaining_rows:,} rows...")
        
        with tqdm_asyncio(total=total_rows, desc="Processing articles (end→start)", initial=len(self.processed_indices)) as pbar:
            for start in range(0, len(full_df), CHUNK_SIZE):
                if self.shutdown_event.is_set():
                    interrupted = True
                    break
                
                chunk = full_df.iloc[start:start + CHUNK_SIZE]
                # Restore original index for checkpoint compatibility
                chunk = chunk.set_index("original_index")
                
                # Process chunk asynchronously
                success_count = await self.process_chunk(chunk, pbar)
                total_success += success_count
                
                # Free memory periodically
                gc.collect()
        
        if interrupted:
            print(f"\n[Interrupted] Stopped gracefully after processing {total_success:,} articles this session")
            print(f"[Interrupted] All progress saved. Re-run to continue from where you left off.")
        else:
            print(f"\n[Phase 1 Complete] Successfully processed: {total_success:,}")
        
        # === PHASE 2: Retry failed rows (only if not interrupted) ===
        if not interrupted:
            self.processed_indices = self.load_processed_indices()
            await self.retry_failed_rows(full_df)
        
        # Check for remaining failures
        if os.path.exists(FAILED_ROWS_FILE):
            failed_count = sum(1 for line in open(FAILED_ROWS_FILE) if line.strip())
            if failed_count > 0:
                print(f"✗ Failed rows remaining: {failed_count:,} (see {FAILED_ROWS_FILE})")
        
        # Build final aggregated CSV
        self.build_final_csv()
        
        gc.collect()
        print(f"\n✓ Pipeline {'stopped' if interrupted else 'completed'}!")


def show_menu() -> int:
    """
    Show an interactive startup menu and return the user's choice:
      1  — NER only    : process raw articles → raw_responses.jsonl (no CSV)
      2  — CSV only    : aggregate raw_responses.jsonl → tier_segregated_news.csv
      3  — Full pipeline: NER + CSV (default end-to-end run)
    """
    print()
    print("=" * 55)
    print("  Financial NER Pipeline — Mode Selection")
    print("=" * 55)
    print("  1)  NER only     — process articles → raw_responses.jsonl")
    print("  2)  CSV only     — aggregate  raw_responses.jsonl → CSV")
    print("  3)  Full pipeline— NER + CSV  (end-to-end)")
    print("=" * 55)

    while True:
        try:
            choice = input("  Select mode [1/2/3]: ").strip()
            if choice in ("1", "2", "3"):
                return int(choice)
            print("  Please enter 1, 2, or 3.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(0)


async def run(mode: int):
    """Async runner dispatched by mode selection."""
    pipeline = FinancialNERPipeline()
    os.makedirs("output", exist_ok=True)

    if mode == 1:
        # NER only: run the pipeline but skip CSV building at the end
        print("\n[Mode 1] NER only — will produce raw_responses.jsonl")
        await pipeline.process_dataset(build_csv=False)

    elif mode == 2:
        # CSV only: skip NER, just aggregate the existing checkpoint
        print("\n[Mode 2] CSV only — aggregating existing raw_responses.jsonl...")
        pipeline.processed_indices = pipeline.load_processed_indices()
        pipeline.build_final_csv()

    else:
        # Full pipeline: NER + CSV (original behaviour)
        print("\n[Mode 3] Full pipeline — NER + CSV")
        await pipeline.process_dataset(build_csv=True)


if __name__ == "__main__":
    mode = show_menu()
    asyncio.run(run(mode))
