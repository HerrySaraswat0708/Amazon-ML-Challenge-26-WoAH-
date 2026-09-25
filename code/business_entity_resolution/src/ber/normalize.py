"""Record-level text normalization for business names and addresses.

All rules are country-agnostic string transforms plus small static lookup tables
(state/region names, street-type and legal-form abbreviations). Nothing here
filters on country: unknown countries fall through the same generic path.
Resources learned from training pairs (native-script state names, native-script
token dictionary) are passed in via `Resources` so they can be fit on a train fold.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from unidecode import unidecode

# --------------------------------------------------------------------------- tables
NULL_TOKENS = {"n/a", "na", "null", "<null>", "none", "nan", "-", "--", "unknown"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc", "south dakota": "sd",
    "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "puerto rico": "pr", "guam": "gu", "virgin islands": "vi",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk",
    "jammu & kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py", "andaman and nicobar islands": "an", "lakshadweep": "ld",
    "dadra and nagar haveli": "dn", "daman and diu": "dd",
}
FR_REGIONS = {
    "hauts-de-france": "hdf", "nouvelle-aquitaine": "naq", "pays de la loire": "pdl",
    "ile-de-france": "idf", "occitanie": "occ", "bretagne": "bre", "normandie": "nor",
    "grand est": "ges", "provence-alpes-cote d'azur": "pac", "auvergne-rhone-alpes": "ara",
    "bourgogne-franche-comte": "bfc", "centre-val de loire": "cvl", "corse": "cor",
    # departments seen in S2/S3 -> their region code
    "nord": "hdf", "pas-de-calais": "hdf", "gironde": "naq", "loire-atlantique": "pdl",
}
# Address-component values that are a state/region -> (canonical code). Codes are
# namespaced by table so "in"/"la"/"ga" from different countries never collide.
STATE_LOOKUP: dict[str, str] = {}
for _tbl, _ns in ((US_STATES, "us"), (IN_STATES, "in"), (FR_REGIONS, "fr")):
    for _full, _code in _tbl.items():
        STATE_LOOKUP[_full] = f"{_ns}:{_code}"
        STATE_LOOKUP[_full.replace("-", " ")] = f"{_ns}:{_code}"
STATE_CODES = {"us": set(US_STATES.values()), "in": set(IN_STATES.values())}

# Normalized city aliases (applied token-wise on the cleaned address).
CITY_ALIASES = {"bombay": "mumbai", "calcutta": "kolkata", "bengaluru": "bangalore",
                "madras": "chennai", "gurugram": "gurgaon", "trivandrum": "thiruvananthapuram",
                "ahmadabad": "ahmedabad", "poona": "pune"}

STREET_TYPES = {
    # English
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "avn": "ave",
    "road": "rd", "drive": "dr", "lane": "ln", "court": "ct", "boulevard": "blvd", "bd": "blvd",
    "bld": "blvd", "highway": "hwy", "parkway": "pkwy", "pky": "pkwy", "place": "pl",
    "circle": "cir", "trail": "trl", "terrace": "ter", "square": "sq", "north": "n",
    "south": "s", "east": "e", "west": "w", "suite": "ste", "apartment": "apt",
    "building": "bldg", "floor": "flr", "fl": "flr", "mount": "mt", "fort": "ft",
    "expressway": "expy", "freeway": "fwy", "point": "pt", "way": "wy",
    # French
    "rue": "r", "route": "rte", "chemin": "ch", "impasse": "imp", "allee": "all",
    "quai": "qu", "cours": "crs", "sainte": "ste", "batiment": "bat",
    # Indian
    "marg": "rd", "nagar": "ngr", "sector": "sec", "colony": "col",
    "opposite": "opp", "near": "nr",
}
# Unit / door-number designators carry no identity once the number is kept.
ADDR_DROP = {"no", "door", "h", "hno", "house", "flat", "plot", "unit", "apt", "ste", "bldg",
             "cdp", "po", "box", "num", "number", "shop", "office", "room"}

LEGAL_FORMS = {
    # canonical -> variants (all lowercase, punctuation-free)
    "inc": ["inc", "incorporated", "incorp"],
    "llc": ["llc", "l l c"],
    "ltd": ["ltd", "limited", "ltda"],
    "pvt": ["pvt", "private", "pvtltd", "prv"],
    "corp": ["corp", "corporation", "corpn"],
    "co": ["co", "company", "cos", "compny"],
    "lp": ["lp"],
    "llp": ["llp"],
    "plc": ["plc"],
    "pllc": ["pllc"],
    "pc": ["pc"],
    "sarl": ["sarl"], "sas": ["sas"], "sasu": ["sasu"], "sa": ["sa"], "eurl": ["eurl"],
    "sci": ["sci"], "ei": ["ei"], "snc": ["snc"], "scop": ["scop"],
    "gmbh": ["gmbh"],
}
LEGAL_LOOKUP = {v: k for k, vs in LEGAL_FORMS.items() for v in vs}

ALIAS_RE = re.compile(
    r"\b(?:doing business as|d\s*/\s*b\s*/\s*a|dba|a\s*/\s*k\s*/\s*a|aka|also known as|"
    r"formerly known as|formerly|f\s*/\s*k\s*/\s*a|fka|trading as|t\s*/\s*a)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|fr|us|biz|info|co)\b",
                    re.IGNORECASE)
TRAILING_COM_RE = re.compile(r"^([a-z0-9]{4,})com$")
# digit->letter look-alikes for OCR-style typos inside alphabetic tokens
OCR_MAP = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
NON_WORD_RE = re.compile(r"[^\w\s]")
SPACES_RE = re.compile(r"\s+")
NAME_STOP = {"and", "the", "of"}
INDIC_RE = re.compile(r"[ऀ-෿]")
NONLATIN_RE = re.compile(r"[^\x00-ɏḀ-ỿ]")


@dataclass
class Resources:
    """Data-driven lookup tables fit on training pairs (see ber.resources)."""
    native_state: dict[str, str] = field(default_factory=dict)   # native-script component -> state code
    native_token: dict[str, str] = field(default_factory=dict)   # squashed translit token -> english token


# --------------------------------------------------------------------------- helpers
def is_null(s: str) -> bool:
    return s is None or s.strip().lower() in NULL_TOKENS or not s.strip()


def script_of(s: str) -> str:
    for ch in s:
        if ch.isalpha():
            return unicodedata.name(ch, "UNKNOWN").split()[0]
    return "NONE"


def fold(s: str) -> str:
    """NFKC + strip accents + lowercase, leaving non-Latin scripts untouched."""
    s = unicodedata.normalize("NFKC", s)
    if NONLATIN_RE.search(s):
        # keep Indic chars; strip combining marks from Latin only
        return "".join(unidecode(ch) if ord(ch) <= 0x24F or 0x1E00 <= ord(ch) <= 0x1EFF else ch
                       for ch in s).lower()
    return unidecode(s).lower()


def squash_phonetic(tok: str) -> str:
    """Collapse the doubled letters / aspirations unidecode emits for Indic scripts."""
    tok = tok.lower()
    tok = re.sub(r"([a-z])\1+", r"\1", tok)                 # dd->d, aa->a, tt->t
    for a, b in (("ph", "f"), ("bh", "v"), ("th", "t"), ("dh", "d"), ("kh", "k"),
                 ("gh", "g"), ("sh", "s"), ("ch", "c"), ("w", "v"), ("z", "j"), ("q", "k")):
        tok = tok.replace(a, b)
    tok = tok[:1] + re.sub(r"[aeiouy]", "", tok[1:])  # consonant skeleton, keep first letter
    return tok


def fix_ocr_token(tok: str) -> str:
    """'5ecure'->'secure', 'c0mpany'->'company'; leave numbers and codes alone."""
    if not tok or tok.isdigit():
        return tok
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    if digits and letters >= 3 and letters >= 2 * digits:
        return tok.translate(OCR_MAP)
    return tok


def transliterate_token(tok: str, res: Resources) -> str:
    """Indic token -> learned English token, else rough unidecode romanization."""
    rom = unidecode(tok).lower()
    return res.native_token.get(squash_phonetic(rom), rom)


def clean_text(s: str) -> str:
    s = s.replace("°", " ").replace("º", " ")
    s = s.replace("&", " and ").replace("+", " and ")
    s = NON_WORD_RE.sub(" ", s).replace("_", " ")
    return SPACES_RE.sub(" ", s).strip()


# --------------------------------------------------------------------------- names
def normalize_name(raw: str, res: Resources | None = None) -> dict:
    res = res or Resources()
    out = {"name_script": "NONE", "name_norm": "", "name_core": "", "name_legal": "",
           "name_alias": "", "name_domain": "", "name_is_translit": False}
    if is_null(raw):
        return out
    out["name_script"] = script_of(raw)
    s = fold(raw)

    # web domains: keep the stem as an extra signal, remove from the name
    m = URL_RE.search(s)
    if m:
        out["name_domain"] = m.group(1).replace("-", "")
        s = URL_RE.sub(" ", s)
    s = s.replace("|", " ")

    # alias split: "X dba Y" / "Y formerly X" -> primary + alias
    parts = [p for p in ALIAS_RE.split(s) if p.strip()]
    if len(parts) > 1:
        out["name_alias"] = clean_text(parts[0])
        s = parts[1]  # the text after the marker is the one S1 usually carries
        if len(clean_text(s)) == 0:
            s = parts[0]

    if INDIC_RE.search(s):
        # transliterate before punctuation stripping: \w does not match Indic vowel signs
        out["name_is_translit"] = True
        s = " ".join(transliterate_token(t, res) for t in s.split())
    s = clean_text(s)

    toks = [fix_ocr_token(t) for t in s.split()]
    if not out["name_domain"] and len(toks) == 1:
        m2 = TRAILING_COM_RE.match(toks[0])
        if m2:  # 'emerakeystonecom'
            out["name_domain"] = m2.group(1)
            toks = [m2.group(1)]
    if not toks and out["name_domain"]:  # 'ipower.com'
        toks = [out["name_domain"]]
    out["name_norm"] = " ".join(toks)

    # legal forms (also catch "private limited", "pvt ltd" split tokens)
    legal, core = [], []
    for t in toks:
        if t in NAME_STOP:
            continue
        (legal if t in LEGAL_LOOKUP else core).append(LEGAL_LOOKUP.get(t, t))
    out["name_legal"] = " ".join(sorted(set(legal)))
    out["name_core"] = " ".join(core)
    return out


# --------------------------------------------------------------------------- addresses
def normalize_address(raw: str, res: Resources | None = None) -> dict:
    res = res or Resources()
    out = {"addr_norm": "", "addr_state": "", "addr_numbers": "", "addr_postcode": "",
           "addr_house": "", "addr_has_native": False, "addr_ncomp": 0}
    if is_null(raw):
        return out
    s = unicodedata.normalize("NFKC", raw)
    comps = [c.strip() for c in s.split(",")]
    comps = [c for c in comps if not is_null(c)]
    out["addr_ncomp"] = len(comps)
    state, kept = "", []
    for c in comps:
        if INDIC_RE.search(c):
            out["addr_has_native"] = True
            code = res.native_state.get(c.strip())
            if code:
                state = state or code
                continue
            c = unidecode(c)
        key = unidecode(c).lower().strip().rstrip(".")
        code = STATE_LOOKUP.get(key)
        if code is None and len(comps) > 1:
            # bare 2-letter code or "erlanger ky" (city + trailing state code)
            low = key.split()
            if len(low) == 1 and len(key) == 2 and any(key in v for v in STATE_CODES.values()):
                code = f"?:{key}"
        if code:
            state = state or code
            continue
        kept.append(key)

    text = clean_text(" , ".join(kept).replace(",", " "))
    toks = []
    for t in text.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        t = STREET_TYPES.get(t, t)
        t = CITY_ALIASES.get(t, t)
        if t.endswith("cdp") and len(t) > 3:   # "chicagocdp"
            t = t[:-3]
        if t in ADDR_DROP:
            continue
        toks.append(t)
    # drop trailing duplicate state code glued to city ("erlanger ky")
    if state and toks and toks[-1] == state.split(":")[1]:
        toks = toks[:-1]
    out["addr_norm"] = " ".join(toks)
    out["addr_state"] = state.split(":")[1] if state else ""

    nums = re.findall(r"\d+(?:[a-z](?![a-z]))?", out["addr_norm"])
    out["addr_numbers"] = " ".join(nums)
    pcs = re.findall(r"\b\d{5,6}\b", out["addr_norm"])
    out["addr_postcode"] = pcs[-1] if pcs else ""
    m = re.match(r"^(\d+)", out["addr_norm"])
    out["addr_house"] = m.group(1) if m else (re.findall(r"\d+", out["addr_norm"]) or [""])[0]
    return out
