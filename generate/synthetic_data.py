"""
Synthetic data generator for fintech-smb-health.

Simulates the customer portfolio of Folio, a generic B2B fintech platform.

Produces three CSVs in data/raw/:
  - companies.csv     (1,000 SMB companies)
  - transactions.csv  (~80,000 rows)
  - payments.csv      (~25,000 rows)

Health segmentation via latent stress score (beta distribution):
  - healthy   ~60%  low stress
  - watch     ~25%  medium stress
  - at_risk   ~15%  high stress
"""

import csv
import random
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from faker import Faker

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
fake.seed_instance(SEED)
rng = np.random.default_rng(SEED)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

TODAY         = date(2026, 5, 18)
LOOKBACK_DAYS = 3 * 365

# ── industries ────────────────────────────────────────────────────────────────
INDUSTRIES = [
    "SaaS",
    "E-commerce",
    "Professional Services",
    "Healthcare",
    "Construction",
    "Restaurants",
    "Retail",
    "Manufacturing",
]

# ── industry spend profiles ───────────────────────────────────────────────────
INDUSTRY_SPEND_PROFILES: dict[str, list[dict]] = {
    "SaaS": [
        {"mcc": "7372", "category": "Software/SaaS",        "merchants": ["AWS", "Google Cloud", "Azure", "Snowflake", "Datadog", "GitHub", "Vercel", "PagerDuty", "Cloudflare", "Sentry"]},
        {"mcc": "7371", "category": "Computer Programming", "merchants": ["Toptal", "Upwork", "Andela", "Braintrust", "Turing", "Crossover"]},
        {"mcc": "5045", "category": "Computers/Peripherals","merchants": ["Apple Business", "Dell Technologies", "CDW", "Insight Direct", "SHI International"]},
        {"mcc": "7379", "category": "IT Support Services",  "merchants": ["Okta", "1Password", "Jamf", "Kandji", "CrowdStrike"]},
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Stripe", "Brex", "Ramp", "Carta", "DocuSign", "Rippling"]},
    ],
    "E-commerce": [
        {"mcc": "5999", "category": "Wholesale/Fulfillment","merchants": ["Shopify Plus", "ShipBob", "ShipStation", "Flexport", "Freightos", "EasyPost"]},
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Klaviyo", "Gorgias", "Recharge", "Yotpo", "Attentive", "Postscript"]},
        {"mcc": "4215", "category": "Courier/Shipping",     "merchants": ["UPS Business", "FedEx Business", "USPS Commercial", "DHL Express", "OnTrac"]},
        {"mcc": "5065", "category": "Electronic Parts",     "merchants": ["Ingram Micro", "Tech Data", "D&H Distributing", "Synnex"]},
        {"mcc": "7372", "category": "Software/SaaS",        "merchants": ["AWS", "Cloudflare", "Fastly", "Segment", "Census", "Mixpanel"]},
    ],
    "Professional Services": [
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Salesforce", "HubSpot", "Zoom", "Slack", "Notion", "Loom", "Monday.com"]},
        {"mcc": "5044", "category": "Office/Photo Equipment","merchants": ["Staples Business", "Office Depot B2B", "W.B. Mason", "Quill"]},
        {"mcc": "7011", "category": "Hotels/Lodging",       "merchants": ["Marriott Business", "Hilton for Business", "Hyatt", "IHG Business"]},
        {"mcc": "4511", "category": "Airlines",             "merchants": ["United Business", "Delta Corporate", "American Airlines Business", "Southwest Business"]},
        {"mcc": "5812", "category": "Restaurants/Dining",   "merchants": ["Grubhub Corporate", "Seamless Business", "Catering by Design", "Fooda"]},
    ],
    "Healthcare": [
        {"mcc": "5047", "category": "Medical Supplies",     "merchants": ["McKesson", "Cardinal Health", "Medline Industries", "Henry Schein", "Patterson Dental"]},
        {"mcc": "5122", "category": "Drugs/Pharmaceuticals","merchants": ["AmerisourceBergen", "Morris & Dickson", "Smith Drug", "Besse Medical"]},
        {"mcc": "7372", "category": "Software/SaaS",        "merchants": ["Epic Systems", "athenahealth", "Modernizing Medicine", "Kareo", "Veeva Systems"]},
        {"mcc": "5999", "category": "Medical Equipment",    "merchants": ["GE Healthcare", "Philips Medical", "Stryker", "Medtronic", "Baxter"]},
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Availity", "Waystar", "Change Healthcare", "Cotiviti", "Optum360"]},
    ],
    "Construction": [
        {"mcc": "5251", "category": "Hardware/Building Materials","merchants": ["Home Depot Pro", "Lowes Pro", "Menards Business", "84 Lumber", "ProBuild", "BlueLinx"]},
        {"mcc": "5065", "category": "Electrical Supplies",  "merchants": ["Graybar Electric", "Anixter", "Wesco International", "Rexel USA", "Eaton"]},
        {"mcc": "7359", "category": "Equipment Rental",     "merchants": ["United Rentals", "Sunbelt Rentals", "RSC Equipment Rental", "BlueLine Rental", "Ahern Rentals"]},
        {"mcc": "1711", "category": "Plumbing/HVAC",        "merchants": ["Ferguson Enterprises", "Winsupply", "Hajoca Corporation", "Reeves-Sain"]},
        {"mcc": "5082", "category": "Construction Equipment","merchants": ["Caterpillar Financial", "John Deere Financial", "Komatsu America", "Volvo CE"]},
    ],
    "Restaurants": [
        {"mcc": "5141", "category": "Food Distributors",    "merchants": ["Sysco", "US Foods", "Performance Food Group", "Gordon Food Service", "Ben E. Keith", "Reinhart Foodservice"]},
        {"mcc": "5812", "category": "Restaurant Supplies",  "merchants": ["Restaurant Depot", "WebstaurantStore", "Central Restaurant Products", "Wasserstrom"]},
        {"mcc": "5946", "category": "Beverage Distributors","merchants": ["Southern Glazer's", "Republic National Distributing", "Breakthru Beverage", "Young's Market"]},
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Toast POS", "Square for Restaurants", "OpenTable", "Yelp for Business", "SpotOn"]},
        {"mcc": "5251", "category": "Maintenance/Repairs",  "merchants": ["ServiceMaster Clean", "EcoLab", "Cintas", "Aramark", "Stericycle"]},
    ],
    "Retail": [
        {"mcc": "5999", "category": "Wholesale/Distribution","merchants": ["UNFI", "KeHE Distributors", "McLane Company", "C&S Wholesale", "Nash Finch"]},
        {"mcc": "7389", "category": "Business Services",    "merchants": ["Square", "Lightspeed", "Shopify POS", "Vend by Lightspeed", "Clover"]},
        {"mcc": "4215", "category": "Shipping/Fulfillment", "merchants": ["UPS Business", "FedEx Business", "OnTrac", "LSO", "LaserShip"]},
        {"mcc": "5065", "category": "Display/Fixtures",     "merchants": ["Displays2go", "Store Supply Warehouse", "Handy Store Fixtures", "REB Storage Systems"]},
        {"mcc": "5044", "category": "Office/Admin",         "merchants": ["Staples Business", "Quill", "W.B. Mason", "Office Depot B2B"]},
    ],
    "Manufacturing": [
        {"mcc": "5085", "category": "Industrial Supplies",  "merchants": ["Grainger", "MSC Industrial", "Fastenal", "Zoro Tools", "Motion Industries", "Applied Industrial"]},
        {"mcc": "5065", "category": "Electronic Components","merchants": ["Arrow Electronics", "Avnet", "TTI Inc.", "Mouser Electronics", "Digi-Key"]},
        {"mcc": "7359", "category": "Equipment Rental",     "merchants": ["United Rentals", "BlueLine Rental", "Ahern Rentals", "Neff Rentals"]},
        {"mcc": "5082", "category": "Machinery",            "merchants": ["Haas Automation", "Mazak", "DMG Mori", "Lincoln Electric", "Miller Welding"]},
        {"mcc": "5169", "category": "Chemicals/Materials",  "merchants": ["Univar Solutions", "Brenntag", "Ashland Global", "Cabot Corp", "Elementis"]},
    ],
}

# ── company name generators ───────────────────────────────────────────────────
# Each industry has a list of name templates; {adj}, {noun}, {name} are filled in
INDUSTRY_NAME_TEMPLATES: dict[str, list[str]] = {
    "SaaS": [
        "{adj}{noun} Software", "{adj}{noun} AI", "{noun}ly", "{adj} Systems",
        "{name} Technologies", "{name} Labs", "{adj}{noun} Cloud", "{noun}Hub",
        "{name} Analytics", "{adj} Data", "{noun}IQ", "{name} Platform",
    ],
    "E-commerce": [
        "{adj} {noun} Commerce", "{noun} Direct", "{adj} Goods Co.", "{name} Brands",
        "{adj} Market", "{noun} Supply Co.", "{name} Shop", "{adj}{noun} Store",
        "{name} Exchange", "{adj} Finds",
    ],
    "Professional Services": [
        "{name} & {name2} Consulting", "{adj} {noun} Advisors", "{noun} Partners",
        "{name} Advisory Group", "{adj} Strategies", "{noun} Associates",
        "{name} Solutions Group", "{adj} Capital Partners", "{name} & Associates",
        "{noun} Consulting",
    ],
    "Healthcare": [
        "{adj} {noun} Health", "{name} Medical Group", "{adj} Care Partners",
        "{noun} Wellness", "{name} Clinic", "{adj} Health Partners",
        "{name} Healthcare", "{adj} {noun} Medical", "{noun}Rx", "{name} Practice",
    ],
    "Construction": [
        "{name} Construction", "{adj} {noun} Builders", "{name} Contractors",
        "{adj} {noun} Development", "{noun} Construction Group", "{name} Build",
        "{adj} Structures", "{name} & Sons Construction", "{adj} {noun} Works",
        "{noun} Development Co.",
    ],
    "Restaurants": [
        "The {adj} {noun}", "{noun} Kitchen", "{adj} {noun} Bistro",
        "{name}'s Grill", "{adj} Table", "{noun} Eatery", "{name}'s Diner",
        "{adj} {noun} Tavern", "{noun} House", "{adj} Corner Cafe",
    ],
    "Retail": [
        "{adj} {noun} Retail", "{name}'s Boutique", "{adj} Goods Co.",
        "{noun} Mercantile", "{adj} {noun} Outfitters", "{name} Trading",
        "{adj} Supply Co.", "{noun} Exchange", "{adj} Market", "{name} & Co.",
    ],
    "Manufacturing": [
        "{name} Manufacturing", "{adj} {noun} Industries", "{noun} Fabrication",
        "{adj} Products Co.", "{name} Works", "{adj} {noun} Corp",
        "{noun} Components", "{adj} Systems Inc.", "{name} Industrial",
        "{adj} {noun} Manufacturing",
    ],
}

_ADJECTIVES = [
    "Allied", "Apex", "Bright", "Central", "Clear", "Core", "Delta", "Dynamic",
    "Elite", "Empire", "First", "Forge", "Frontier", "Global", "Golden", "Grand",
    "Harbor", "Highland", "Horizon", "Iron", "Keystone", "Metro", "National",
    "Noble", "North", "Pacific", "Peak", "Pioneer", "Premier", "Prime",
    "Prism", "Rapid", "Ridge", "Slate", "Solid", "Sterling", "Summit",
    "Swift", "Titan", "United", "Vantage", "Versa", "Victory", "Western",
]

_NOUNS = [
    "Axis", "Beacon", "Bridge", "Capital", "Cedar", "Coast", "Crest", "Crown",
    "Edge", "Field", "Flux", "Grove", "Harbor", "Haven", "Heights", "Keystone",
    "Lakeview", "Lantern", "Meadow", "Mill", "Oak", "Orbit", "Path", "Pine",
    "Point", "Rail", "River", "Rock", "Shore", "Signal", "Spring", "Stone",
    "Stream", "Timber", "Tower", "Trail", "Valley", "Vista", "Water", "Wood",
]

_FIRST_NAMES = [
    "Adams", "Allen", "Baker", "Bennett", "Brooks", "Carter", "Chen", "Clark",
    "Collins", "Davis", "Evans", "Fisher", "Foster", "Garcia", "Grant", "Green",
    "Hall", "Harris", "Hayes", "Hill", "Hughes", "Jackson", "James", "Johnson",
    "Jones", "Kelly", "Kim", "King", "Lee", "Lewis", "Lopez", "Martin",
    "Martinez", "Miller", "Mitchell", "Moore", "Morgan", "Murphy", "Nelson",
    "Nguyen", "Parker", "Patel", "Phillips", "Price", "Reed", "Rivera", "Roberts",
    "Robinson", "Rogers", "Ross", "Scott", "Smith", "Taylor", "Thomas", "Thompson",
    "Turner", "Walker", "Ward", "White", "Williams", "Wilson", "Wood", "Wright",
    "Young", "Zhang",
]


def _make_company_name(industry: str, used: set[str]) -> str:
    template = random.choice(INDUSTRY_NAME_TEMPLATES[industry])
    for _ in range(300):
        adj   = random.choice(_ADJECTIVES)
        noun  = random.choice(_NOUNS)
        name  = random.choice(_FIRST_NAMES)
        name2 = random.choice(_FIRST_NAMES)
        candidate = (
            template
            .replace("{adj}",   adj)
            .replace("{noun}",  noun)
            .replace("{name2}", name2)
            .replace("{name}",  name)
        )
        if candidate not in used:
            return candidate
    # numeric fallback
    base = f"{random.choice(_ADJECTIVES)} {random.choice(_NOUNS)}"
    i = 2
    while f"{base} {i}" in used:
        i += 1
    return f"{base} {i}"


# ── geography ─────────────────────────────────────────────────────────────────
STATES = [
    "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
    "WA", "AZ", "CO", "MN", "OR", "NV", "MA", "TN", "VA", "MO",
    "NJ", "CT", "WI", "IN", "SC", "AL", "LA", "KY", "UT", "OK",
]

STATE_REGION: dict[str, str] = {
    "NY": "Northeast", "PA": "Northeast", "MA": "Northeast", "NJ": "Northeast",
    "CT": "Northeast", "ME": "Northeast", "NH": "Northeast", "VT": "Northeast",
    "RI": "Northeast",
    "TX": "South", "FL": "South", "GA": "South", "NC": "South",
    "VA": "South", "TN": "South", "AL": "South", "SC": "South",
    "LA": "South", "AR": "South", "MS": "South", "KY": "South",
    "WV": "South", "OK": "South",
    "IL": "Midwest", "OH": "Midwest", "MI": "Midwest", "MN": "Midwest",
    "WI": "Midwest", "MO": "Midwest", "IN": "Midwest", "IA": "Midwest",
    "KS": "Midwest", "NE": "Midwest", "ND": "Midwest", "SD": "Midwest",
    "CA": "West", "WA": "West", "OR": "West", "CO": "West",
    "AZ": "West", "NV": "West", "UT": "West", "ID": "West",
    "MT": "West", "WY": "West", "NM": "West", "AK": "West", "HI": "West",
}

CSM_NAMES = [
    "Jordan Rivera", "Alex Chen", "Samantha Patel", "Marcus Johnson", "Priya Nair",
    "Derek Walsh", "Aisha Thompson", "Carlos Mendez", "Rachel Kim", "Tyler Brooks",
    "Morgan Lee", "Casey Park", "Dana Okonkwo", "Riley Fernandez", "Quinn Nakamura",
]

# ── failure reasons ───────────────────────────────────────────────────────────
# Proportions: insufficient_funds 40%, bank_declined 28%, invalid_account 14%,
#              timeout 10%, duplicate_detected 8%
FAILURE_REASONS        = ["insufficient_funds", "bank_declined", "invalid_account", "timeout", "duplicate_detected"]
FAILURE_REASON_WEIGHTS = [0.40, 0.28, 0.14, 0.10, 0.08]

PAYMENT_METHODS = ["ACH", "virtual_card", "check"]

INDUSTRY_PAYMENT_METHOD: dict[str, list[tuple[str, float]]] = {
    "SaaS":                  [("ACH", 0.30), ("virtual_card", 0.60), ("check", 0.10)],
    "E-commerce":            [("ACH", 0.35), ("virtual_card", 0.55), ("check", 0.10)],
    "Professional Services": [("ACH", 0.50), ("virtual_card", 0.30), ("check", 0.20)],
    "Healthcare":            [("ACH", 0.55), ("virtual_card", 0.25), ("check", 0.20)],
    "Construction":          [("ACH", 0.40), ("virtual_card", 0.20), ("check", 0.40)],
    "Restaurants":           [("ACH", 0.45), ("virtual_card", 0.25), ("check", 0.30)],
    "Retail":                [("ACH", 0.40), ("virtual_card", 0.40), ("check", 0.20)],
    "Manufacturing":         [("ACH", 0.50), ("virtual_card", 0.15), ("check", 0.35)],
}

# ── seasonality ───────────────────────────────────────────────────────────────
_SEASONALITY: dict[str, dict[int, float]] = {
    "default": {
        1: 0.78, 2: 0.82, 3: 0.92, 4: 0.97, 5: 1.02,
        6: 1.06, 7: 1.05, 8: 1.10, 9: 1.08, 10: 1.14,
        11: 1.09, 12: 0.87,
    },
    "Retail": {
        1: 0.72, 2: 0.75, 3: 0.85, 4: 0.90, 5: 0.95,
        6: 0.97, 7: 0.98, 8: 1.05, 9: 1.10, 10: 1.20,
        11: 1.35, 12: 1.40,
    },
    "Restaurants": {
        1: 0.70, 2: 0.74, 3: 0.85, 4: 0.92, 5: 1.05,
        6: 1.18, 7: 1.22, 8: 1.15, 9: 1.00, 10: 1.04,
        11: 1.12, 12: 1.13,
    },
    "Construction": {
        1: 0.55, 2: 0.60, 3: 0.80, 4: 1.05, 5: 1.20,
        6: 1.25, 7: 1.25, 8: 1.20, 9: 1.15, 10: 1.00,
        11: 0.75, 12: 0.60,
    },
    "Healthcare": {
        1: 1.10, 2: 1.05, 3: 1.02, 4: 0.98, 5: 0.97,
        6: 0.95, 7: 0.94, 8: 0.96, 9: 1.00, 10: 1.02,
        11: 1.05, 12: 0.96,
    },
}


def seasonality(month: int, industry: str) -> float:
    table = _SEASONALITY.get(industry, _SEASONALITY["default"])
    return table[month]


# ── lifecycle phases ──────────────────────────────────────────────────────────
LIFECYCLE_PHASES = ["stable", "declining", "recovering", "volatile"]

LIFECYCLE_WEIGHTS = {
    "healthy":  [0.70, 0.08, 0.15, 0.07],
    "watch":    [0.30, 0.35, 0.20, 0.15],
    "at_risk":  [0.10, 0.55, 0.15, 0.20],
}


def pick_lifecycle(segment: str) -> str:
    return random.choices(LIFECYCLE_PHASES, weights=LIFECYCLE_WEIGHTS[segment], k=1)[0]


# ── latent stress score → health segment ─────────────────────────────────────
def _assign_segment_from_stress(stress: float) -> str:
    """Probabilistic label from stress score with label noise."""
    if stress < 0.35:
        return random.choices(["healthy", "watch", "at_risk"], weights=[0.85, 0.12, 0.03])[0]
    elif stress < 0.65:
        return random.choices(["healthy", "watch", "at_risk"], weights=[0.20, 0.60, 0.20])[0]
    else:
        return random.choices(["healthy", "watch", "at_risk"], weights=[0.05, 0.20, 0.75])[0]


# ── utilities ─────────────────────────────────────────────────────────────────
def weighted_choice(options: list[tuple[str, float]]) -> str:
    items, weights = zip(*options)
    return random.choices(items, weights=weights, k=1)[0]


def random_date_weighted_early(start: date, end: date) -> date:
    """Returns a date weighted toward the earlier end of the range."""
    delta = (end - start).days
    # Use a beta(1, 2) draw so earlier dates are more likely
    frac = rng.beta(1, 2)
    return start + timedelta(days=int(frac * delta))


# ── 1. COMPANIES ─────────────────────────────────────────────────────────────
def generate_companies(n: int = 1000) -> list[dict]:
    companies   = []
    used_names: set[str] = set()

    industry_pool: list[str] = []
    per_industry = n // len(INDUSTRIES)
    remainder    = n % len(INDUSTRIES)
    for idx, ind in enumerate(INDUSTRIES):
        count = per_industry + (1 if idx < remainder else 0)
        industry_pool.extend([ind] * count)
    random.shuffle(industry_pool)

    for i in range(n):
        industry = industry_pool[i]

        # Latent stress score drives segment (beta gives right-skewed distribution)
        stress = float(rng.beta(1.5, 3.5))
        segment = _assign_segment_from_stress(stress)

        size = random.choices(["small", "medium", "large"], weights=[0.50, 0.35, 0.15])[0]

        # Non-round credit limits (random within range, then add random cents)
        base_limit = {
            "small":  rng.integers(8_000,  80_000),
            "medium": rng.integers(80_000, 275_000),
            "large":  rng.integers(275_000, 550_000),
        }[size]
        # Add a non-round offset
        credit_limit = int(base_limit) + random.choice([250, 500, 750, 1_250, 2_500, 3_750, 5_000])

        if segment == "at_risk":
            credit_limit = int(credit_limit * rng.uniform(0.38, 0.72))

        name = _make_company_name(industry, used_names)
        used_names.add(name)

        signup_date = random_date_weighted_early(
            TODAY - timedelta(days=LOOKBACK_DAYS),
            TODAY - timedelta(days=45),
        )
        state = random.choice(STATES)
        days_since_signup = (TODAY - signup_date).days
        tenure_bucket = (
            "new"         if days_since_signup <= 183 else
            "growing"     if days_since_signup <= 548 else
            "established"
        )

        lifecycle = pick_lifecycle(segment)

        companies.append({
            "company_id":     f"CO{i+1:04d}",
            "company_name":   name,
            "industry":       industry,
            "company_size":   size,
            "state":          state,
            "region":         STATE_REGION.get(state, "Other"),
            "signup_date":    signup_date.isoformat(),
            "tenure_bucket":  tenure_bucket,
            "credit_limit":   credit_limit,
            "assigned_csm":   random.choice(CSM_NAMES),
            "health_segment": segment,
            "lifecycle_phase": lifecycle,
            "_stress":        round(stress, 4),   # kept for downstream signal; removed before CSV
            "dominant_payment_method":    None,
            "most_common_failure_reason": None,
        })

    return companies


# ── 2. TRANSACTIONS ───────────────────────────────────────────────────────────
def _base_failure_rate(stress: float, lifecycle: str) -> float:
    """Smooth failure rate from stress score + lifecycle modifier."""
    base = 0.02 + stress * 0.22           # range ~2% – 24%
    modifiers = {
        "stable":    0.0,
        "declining": 0.04,
        "recovering": -0.02,
        "volatile":  0.06,
    }
    return min(0.35, max(0.01, base + modifiers[lifecycle]))


def _recency_factor(txn_date: date, lifecycle: str) -> float:
    """Spend multiplier based on recency and lifecycle phase."""
    days_ago = (TODAY - txn_date).days
    if lifecycle == "declining":
        # Spend shrinks linearly from 100% (oldest) to 50% (today)
        return 1.0 - 0.5 * max(0.0, 1.0 - days_ago / LOOKBACK_DAYS)
    if lifecycle == "recovering":
        # Spend grows linearly from 60% (oldest) to 100% (today)
        return 0.60 + 0.40 * max(0.0, 1.0 - days_ago / LOOKBACK_DAYS)
    if lifecycle == "volatile":
        # Random ±30% noise
        return rng.uniform(0.70, 1.30)
    return 1.0  # stable


def generate_transactions(companies: list[dict], target: int = 80_000) -> list[dict]:
    transactions = []
    txn_id = 1

    size_weight    = {"small": 1.0, "medium": 2.0, "large": 3.5}
    segment_weight = {"healthy": 1.0, "watch": 0.80, "at_risk": 0.55}
    raw_weights    = [
        size_weight[c["company_size"]] * segment_weight[c["health_segment"]]
        for c in companies
    ]
    total_weight = sum(raw_weights)
    txn_counts   = [max(15, round(target * w / total_weight)) for w in raw_weights]

    for company, n_txns in zip(companies, txn_counts):
        industry      = company["industry"]
        lifecycle     = company["lifecycle_phase"]
        stress        = company["_stress"]
        spend_profile = INDUSTRY_SPEND_PROFILES[industry]
        pay_methods   = INDUSTRY_PAYMENT_METHOD[industry]

        signup     = date.fromisoformat(company["signup_date"])
        start_date = max(signup, TODAY - timedelta(days=LOOKBACK_DAYS))

        fail_rate = _base_failure_rate(stress, lifecycle)

        # Base amount scale by company size
        amount_scale = {"small": 400.0, "medium": 1_500.0, "large": 6_000.0}[company["company_size"]]

        for _ in range(n_txns):
            txn_date = start_date + timedelta(
                days=int(rng.integers(0, max(1, (TODAY - start_date).days)))
            )

            spend_cat = random.choice(spend_profile)
            merchant  = random.choice(spend_cat["merchants"])

            # Gamma-distributed amount, rounded to cents
            raw_amount = rng.gamma(shape=2.0, scale=amount_scale)
            # 3% large transactions
            if rng.random() < 0.03:
                raw_amount *= rng.uniform(5, 18)

            amount = round(
                float(raw_amount)
                * seasonality(txn_date.month, industry)
                * _recency_factor(txn_date, lifecycle),
                2,
            )
            amount = max(1.00, amount)

            rand_fail = rng.random()
            if rand_fail < fail_rate:
                status         = "failed"
                failure_reason = random.choices(FAILURE_REASONS, weights=FAILURE_REASON_WEIGHTS)[0]
            elif rand_fail < fail_rate + 0.035:
                status         = "pending"
                failure_reason = None
            else:
                status         = "completed"
                failure_reason = None

            transactions.append({
                "txn_id":         f"TXN{txn_id:07d}",
                "company_id":     company["company_id"],
                "amount":         amount,
                "merchant_name":  merchant,
                "mcc_code":       spend_cat["mcc"],
                "mcc_category":   spend_cat["category"],
                "payment_method": weighted_choice(pay_methods),
                "txn_date":       txn_date.isoformat(),
                "txn_status":     status,
                "failure_reason": failure_reason,
            })
            txn_id += 1

    return transactions


# ── 3. PAYMENTS ───────────────────────────────────────────────────────────────
def _next_billing_date(ref: date) -> date:
    """Returns the next 1st or 15th on or after ref."""
    if ref.day <= 1:
        return ref.replace(day=1)
    if ref.day <= 15:
        return ref.replace(day=15)
    # after 15th → next month's 1st
    if ref.month == 12:
        return date(ref.year + 1, 1, 1)
    return date(ref.year, ref.month + 1, 1)


def _days_late_sample(stress: float, lifecycle: str) -> int:
    """Right-skewed days-late draw calibrated to stress score."""
    # Base mean: low stress ~-2 days (early), high stress ~20 days late
    mean = -3 + stress * 30
    # Lifecycle shifts
    shifts = {"declining": 6, "recovering": -4, "volatile": 8, "stable": 0}
    mean += shifts[lifecycle]
    # Right-skewed via exponential + normal
    skew_component = float(rng.exponential(scale=max(1.0, stress * 15)))
    noise          = float(rng.normal(0, 3))
    raw            = mean + skew_component * 0.5 + noise
    return int(max(-14, min(raw, 120)))


def generate_payments(companies: list[dict], target: int = 25_000) -> list[dict]:
    payments   = []
    payment_id = 1

    total_months = sum(
        min(30, max(1, round((TODAY - date.fromisoformat(c["signup_date"])).days / 30)))
        for c in companies
    )

    for company in companies:
        stress    = company["_stress"]
        lifecycle = company["lifecycle_phase"]
        signup    = date.fromisoformat(company["signup_date"])

        months_active = min(30, max(1, round((TODAY - signup).days / 30)))
        n_payments    = max(1, round(target * months_active / total_months))

        pay_methods = INDUSTRY_PAYMENT_METHOD[company["industry"]]

        # Miss probability driven by stress score
        miss_prob = 0.005 + stress * 0.20
        if lifecycle == "declining":
            miss_prob = min(0.35, miss_prob + 0.07)

        # Generate monthly payment records
        billing_start = _next_billing_date(signup + timedelta(days=28))

        period_dates: list[date] = []
        d = billing_start
        while d <= TODAY and len(period_dates) < n_payments:
            period_dates.append(d)
            # Advance ~1 month
            if d.month == 12:
                d = d.replace(year=d.year + 1, month=1)
            else:
                d = d.replace(month=d.month + 1)

        # Invoice amount ~ fraction of credit limit, declining for at_risk
        credit = company["credit_limit"]
        n_periods = max(len(period_dates), 1)

        for idx, due_date in enumerate(period_dates):
            if due_date > TODAY:
                continue

            # Invoice amount with some noise
            util_frac = float(rng.uniform(0.04, 0.40))
            if company["health_segment"] == "at_risk" and lifecycle == "declining":
                progress = idx / n_periods
                util_frac *= (0.45 + 0.55 * progress)

            amount = round(credit * util_frac, 2)

            # Missed?
            if rng.random() < miss_prob:
                days_late = int(rng.integers(45, 121))
                status    = "missed"
                # Geometric retry count (p=0.4, min 1)
                retry_count = 1 + int(rng.geometric(p=0.40)) - 1
                retry_count = min(retry_count, 6)
                paid_date   = None
            else:
                days_late   = _days_late_sample(stress, lifecycle)
                status      = "late" if days_late > 5 else "on_time"
                retry_count = 0 if status == "on_time" else min(int(rng.geometric(p=0.55)), 3)
                paid = due_date + timedelta(days=days_late)
                paid_date = paid if paid <= TODAY else None
                if paid_date is None:
                    status    = "missed"
                    days_late = None

            payments.append({
                "payment_id":     f"PAY{payment_id:07d}",
                "company_id":     company["company_id"],
                "amount":         amount,
                "payment_method": weighted_choice(pay_methods),
                "due_date":       due_date.isoformat(),
                "paid_date":      paid_date.isoformat() if paid_date else None,
                "days_late":      days_late,
                "payment_status": status,
                "retry_count":    retry_count,
                "period":         due_date.strftime("%Y-%m"),
            })
            payment_id += 1

    return payments


# ── enrichment ────────────────────────────────────────────────────────────────
def enrich_companies(
    companies: list[dict],
    transactions: list[dict],
    payments: list[dict],
) -> None:
    pay_methods_by_co: dict[str, list[str]] = {}
    for p in payments:
        pay_methods_by_co.setdefault(p["company_id"], []).append(p["payment_method"])

    failure_reasons_by_co: dict[str, list[str]] = {}
    for t in transactions:
        if t["failure_reason"]:
            failure_reasons_by_co.setdefault(t["company_id"], []).append(t["failure_reason"])

    for c in companies:
        cid = c["company_id"]
        methods = pay_methods_by_co.get(cid, [])
        c["dominant_payment_method"] = Counter(methods).most_common(1)[0][0] if methods else "ACH"
        reasons = failure_reasons_by_co.get(cid, [])
        c["most_common_failure_reason"] = Counter(reasons).most_common(1)[0][0] if reasons else "none"


# ── CSV writer ────────────────────────────────────────────────────────────────
_INTERNAL_FIELDS = {"_stress", "lifecycle_phase"}

def write_csv(records: list[dict], path: Path, exclude: set[str] | None = None) -> None:
    if not records:
        return
    exclude = exclude or set()
    fields  = [k for k in records[0].keys() if k not in exclude]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ── summary ───────────────────────────────────────────────────────────────────
def print_summary(companies: list[dict], transactions: list[dict], payments: list[dict]) -> None:
    print("\n" + "=" * 62)
    print("  SYNTHETIC DATA GENERATION SUMMARY")
    print("=" * 62)

    print(f"\n{'TABLE':<20} {'ROWS':>10}")
    print("-" * 32)
    for label, data in [("companies", companies), ("transactions", transactions), ("payments", payments)]:
        print(f"  {label:<18} {len(data):>10,}")

    # Health segment breakdown
    seg_counts: dict[str, int] = {}
    for c in companies:
        seg_counts[c["health_segment"]] = seg_counts.get(c["health_segment"], 0) + 1
    print(f"\n{'HEALTH SEGMENTS':<20}")
    print("-" * 32)
    for seg in ["healthy", "watch", "at_risk"]:
        cnt = seg_counts.get(seg, 0)
        pct = cnt / len(companies) * 100
        print(f"  {seg:<14} {cnt:>4}  ({pct:.0f}%)")

    # Lifecycle phases
    phase_counts: dict[str, int] = {}
    for c in companies:
        phase_counts[c["lifecycle_phase"]] = phase_counts.get(c["lifecycle_phase"], 0) + 1
    print(f"\n{'LIFECYCLE PHASES':<20}")
    print("-" * 32)
    for phase in LIFECYCLE_PHASES:
        cnt = phase_counts.get(phase, 0)
        print(f"  {phase:<14} {cnt:>4}")

    # Signal separation
    seg_txns: dict[str, list] = {"healthy": [], "watch": [], "at_risk": []}
    co_seg   = {c["company_id"]: c["health_segment"] for c in companies}
    for t in transactions:
        seg = co_seg.get(t["company_id"])
        if seg:
            seg_txns[seg].append(t["txn_status"] == "failed")
    print(f"\n{'SIGNAL SEPARATION':<20}")
    print("-" * 62)
    print(f"  {'Segment':<12}  {'Failure Rate':>14}")
    for seg in ["healthy", "watch", "at_risk"]:
        vals = seg_txns[seg]
        rate = sum(vals) / len(vals) * 100 if vals else 0
        print(f"  {seg:<12}  {rate:>13.1f}%")

    failed  = sum(1 for t in transactions if t["txn_status"] == "failed")
    pending = sum(1 for t in transactions if t["txn_status"] == "pending")
    print(f"\n{'TRANSACTION QUALITY':<20}")
    print("-" * 32)
    print(f"  Failure rate:   {failed / len(transactions) * 100:.1f}%  ({failed:,})")
    print(f"  Pending:        {pending / len(transactions) * 100:.1f}%  ({pending:,})")

    missed = sum(1 for p in payments if p["payment_status"] == "missed")
    lated  = [p["days_late"] for p in payments if p["days_late"] is not None]
    avg_dl = sum(lated) / len(lated) if lated else 0
    print(f"\n{'PAYMENT BEHAVIOR':<20}")
    print("-" * 32)
    print(f"  Missed rate:    {missed / len(payments) * 100:.1f}%  ({missed:,})")
    print(f"  Avg days late:  {avg_dl:.1f}  (non-missed)")

    # Sample 10 company names
    sample = random.sample(companies, min(10, len(companies)))
    print(f"\n{'SAMPLE COMPANY NAMES':<20}")
    print("-" * 62)
    for c in sorted(sample, key=lambda x: x["industry"]):
        print(f"  {c['company_name']:<40}  {c['industry']:<24}  {c['health_segment']}")

    print("\n" + "=" * 62)
    print(f"  CSVs written to: {RAW_DIR}")
    print("=" * 62 + "\n")


# ── entrypoint ────────────────────────────────────────────────────────────────
def main() -> None:
    print("Generating companies (n=1,000)...")
    companies = generate_companies(1_000)

    print("Generating transactions (target=80,000)...")
    transactions = generate_transactions(companies, target=80_000)

    print("Generating payments (target=25,000)...")
    payments = generate_payments(companies, target=25_000)

    print("Enriching companies...")
    enrich_companies(companies, transactions, payments)

    write_csv(companies, RAW_DIR / "companies.csv", exclude={"_stress", "lifecycle_phase"})
    write_csv(transactions, RAW_DIR / "transactions.csv")
    write_csv(payments,     RAW_DIR / "payments.csv")

    print_summary(companies, transactions, payments)


if __name__ == "__main__":
    main()
