from __future__ import annotations

import csv
import os
import sys
import time
import re
import imaplib
import email
import logging
import pickle
import datetime
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from email.header import decode_header
from email.utils import parseaddr, parsedate_tz, mktime_tz

# ─────── LOGGING ───────
log_path = os.path.join(os.getcwd(), "ctrader.log")
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    encoding="utf-8",
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger()
logger.info("=== Script started ===")

# ─────── ENV VARS ───────
try:
    IMAP_SERVER = os.environ["IMAP_HOST"]
    IMAP_USER   = os.environ["IMAP_USER"]
    IMAP_PASS   = os.environ["IMAP_PASS"]
    SLACK_URL   = os.environ["SLACK_URL"]
    CHECK_SEC   = int(os.getenv("CHECK_SEC", "10"))
    logger.info("Environment variables loaded")
except KeyError as e:
    logger.exception(f"Missing required environment variable: {e}")
    raise

UID_PATH = os.path.join(os.getcwd(), "ctrader_uids.pkl")
SYMBOLS_PATH = os.path.join(os.getcwd(), "symbols.csv")


@dataclass
class SymbolRule:
    symbol: str
    formula: str
    volume_limit: float
    contract_size: int


SYMBOL_RULES: dict[str, SymbolRule] = {}
SYMBOLS_MTIME: float | None = None

# ─────── UID FILE STORAGE ───────
def load_uids() -> set[str]:
    try:
        logger.debug(f"Opening UID file: {UID_PATH}")
        with open(UID_PATH, "rb") as f:
            uids = pickle.load(f)
            logger.info(f"Loaded {len(uids)} UID(s)")
            logger.debug(f"UIDs loaded: {list(uids)}")
            return uids
    except FileNotFoundError:
        logger.warning("UID file not found. Starting fresh.")
        return set()
    except Exception:
        logger.exception("Unexpected error reading UID file")
        return set()

def save_uids(uids: set[str]) -> None:
    try:
        with open(UID_PATH, "wb") as f:
            pickle.dump(uids, f)
            logger.info(f"Saved {len(uids)} UID(s) to file")
    except Exception:
        logger.exception("Unexpected error writing UID file")

# ─────── SLACK ───────
def slack(text: str) -> bool:
    logger.debug(f"Sending to Slack: {text[:120]}")
    try:
        r = requests.post(SLACK_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        logger.info("Slack message sent")
        return True
    except Exception:
        logger.exception("Slack POST failed")
        return False

# ─────── HELPERS ───────
def decode_header_value(v: str) -> str:
    parts = decode_header(v)
    out = ""
    for frag, enc in parts:
        if isinstance(frag, bytes):
            try:
                out += frag.decode(enc or "utf-8")
            except Exception:
                out += frag.decode("utf-8", errors="replace")
        else:
            out += frag
    return out

def format_address(addr_str: str) -> str:
    real, addr = parseaddr(addr_str)
    real = real.strip()
    return f"{real} `<{addr}>`" if real else f"<{addr}>"

# ─────── PARSE EMAIL HTML ───────
FIELDS = [
    "Account",
    "Symbol",
    "Trade Side",
    "Volume",
    "Volume in USD",
    r"Resolved Time \(UTC\)",
]

def extract_email_data(html: str) -> dict[str, str | None]:
    txt = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    out = {}
    for f in FIELDS:
        m = re.search(rf"{f}:\s*(.+)", txt, re.I)
        plain = f.replace(r"\(", "(").replace(r"\)", ")")
        out[plain] = m.group(1).splitlines()[0].strip() if m else None
    return out

# ─────── SLACK MESSAGE BUILDER ───────
USER_CURRENCY = "USD"

def parse_numeric(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d[\d\s,\.]*", value)
    if not match:
        return None
    num_str = match.group(0).replace(" ", "")
    if "," in num_str and "." not in num_str:
        num_str = num_str.replace(",", ".")
    else:
        num_str = num_str.replace(",", "")
    try:
        return float(num_str)
    except ValueError:
        logger.warning("Failed to parse numeric value from '%s'", value)
        return None


def extract_currency(value: str | None) -> str | None:
    if not value:
        return None
    tokens = re.findall(r"[A-Za-z]{3,}", value)
    return tokens[-1].upper() if tokens else None


def compute_volume(rule: SymbolRule, data: dict[str, str | None]) -> tuple[float, str] | None:
    if rule.formula == "lots":
        vol_value = parse_numeric(data.get("Volume"))
        if vol_value is None or rule.contract_size <= 0:
            logger.warning(
                "Cannot compute 'lots' volume for %s (volume=%s, contract_size=%s)",
                rule.symbol,
                data.get("Volume"),
                rule.contract_size,
            )
            return None
        computed = round(vol_value / rule.contract_size, 2)
        unit = "lot"
        return computed, unit

    if rule.formula == "money":
        usd_value = parse_numeric(data.get("Volume in USD"))
        if usd_value is None:
            logger.warning(
                "Cannot compute 'money' volume for %s (Volume in USD=%s)",
                rule.symbol,
                data.get("Volume in USD"),
            )
            return None
        computed = round(usd_value / 1_000_000, 2)
        currency = extract_currency(data.get("Volume in USD")) or "USD"
        logger.debug(
            "Computed 'money' volume for %s: %.2f m (currency %s)",
            rule.symbol,
            computed,
            currency,
        )
        unit = "m"
        return computed, unit

    logger.warning("Unknown formula '%s' for symbol %s", rule.formula, rule.symbol)
    return None


def make_slack_message(msg: email.message.Message) -> tuple[str | None, bool]:
    subj = decode_header_value(msg.get("Subject", "No subject"))
    from_f = format_address(decode_header_value(msg.get("From", "")))
    raw_cc = decode_header_value(msg.get("Cc", ""))
    cc_list = [format_address(a.strip()) for a in raw_cc.split(",") if a.strip()]

    date_str = ""
    if msg.get("Date"):
        tup = parsedate_tz(msg["Date"])
        if tup:
            date_str = datetime.datetime.fromtimestamp(mktime_tz(tup)).strftime("%Y-%m-%d %H:%M:%S")

    html_body = None
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/html" and "attachment" not in str(p.get("Content-Disposition", "")).lower():
                html_body = (p.get_payload(decode=True) or b"").decode(errors="replace")
                break
    elif msg.get_content_type() == "text/html":
        html_body = (msg.get_payload(decode=True) or b"").decode(errors="replace")

    data = extract_email_data(html_body) if html_body else {}
    acc = data.get("Account")
    side = data.get("Trade Side")
    vol_usd = data.get("Volume in USD")
    sym = data.get("Symbol")

    if all([acc, side, vol_usd, sym]):
        rule = SYMBOL_RULES.get(sym.upper())
        if rule:
            computed = compute_volume(rule, data)
            if computed:
                volume_value, unit = computed
                if volume_value > rule.volume_limit:
                    formatted_volume = f"{volume_value:.2f} {unit}".strip()
                    return (
                        (
                            f":exclamation:`Large Volume Traded` *[{USER_CURRENCY}]*: "
                            f"*CT-LIVE-{acc}<https://admin.litefinance.com/c-trader/view?server=90&login={acc}|[web]>*  "
                            f"*{side}*  *{formatted_volume}*  *{sym}*"
                        ),
                        True,
                    )
                logger.info(
                    "Volume %.2f %s for %s did not exceed limit %.2f",
                    volume_value,
                    unit,
                    sym,
                    rule.volume_limit,
                )
                return None, False
            logger.warning("Falling back to default Slack message for %s", sym)

        try:
            vol_div = round(float(vol_usd.replace(",", "").replace(" ", "")) / 100000, 2)
        except Exception:
            logger.warning("Failed to divide volume: %s", vol_usd)
            vol_div = vol_usd
        return (
            (
                f":exclamation:`Large Volume Traded` *[{USER_CURRENCY}]*: "
                f"*CT-LIVE-{acc}<https://admin.litefinance.com/c-trader/view?server=90&login={acc}|[web]>*  "
                f"*{side}*  *{vol_div}* eur/lot  *{sym}*"
            ),
            True,
        )

    # fallback – plain message
    body_txt = ""
    if msg.is_multipart():
        for p in msg.walk():
            if p.get_content_type() == "text/plain" and "attachment" not in str(p.get("Content-Disposition", "")).lower():
                body_txt = (p.get_payload(decode=True) or b"").decode(errors="replace")
                break
    else:
        body_txt = (msg.get_payload(decode=True) or b"").decode(errors="replace")

    parts = []
    if date_str: parts.append(f"Date: {date_str}")
    parts.append(f"From: {from_f}")
    if cc_list: parts.append(f"Cc: {', '.join(cc_list)}")
    parts.append(f"Subject: {subj}")
    parts.append("Body:")
    parts.append("```")
    parts.append(body_txt)
    parts.append("```")
    return "\n".join(parts), True

# ─────── IMAP PROCESSING ───────
def process_box(conn: imaplib.IMAP4_SSL, done: set[str]) -> None:
    logger.debug("Searching IMAP for UIDs...")
    st, data = conn.uid("search", None, "ALL")
    if st != "OK":
        logger.error("IMAP UID SEARCH failed: %s", st)
        return

    for uid_b in data[0].split():
        uid = uid_b.decode()
        if uid in done:
            continue
        logger.debug(f"New UID {uid} found — processing...")

        try:
            st, d = conn.uid("fetch", uid, "(RFC822)")
            if st != "OK":
                logger.error("FETCH %s failed: %s", uid, st)
                continue

            msg = email.message_from_bytes(d[0][1])
            text, should_send = make_slack_message(msg)
            if should_send and text:
                if slack(text):
                    done.add(uid)
                    save_uids(done)
                    logger.info(f"UID {uid} processed and saved")
                else:
                    logger.warning(f"Slack failed for UID {uid}")
            else:
                done.add(uid)
                save_uids(done)
                logger.info(f"UID {uid} processed without Slack notification")

        except Exception:
            logger.exception(f"Error while processing UID {uid}")

# ─────── MAIN LOOP ───────
def restart_program():
    logger.warning("Restarting process to pick up new configuration")
    python = sys.executable
    os.execv(python, [python, *sys.argv])


def load_symbol_rules() -> tuple[dict[str, SymbolRule], float | None]:
    rules: dict[str, SymbolRule] = {}
    try:
        mtime = os.path.getmtime(SYMBOLS_PATH)
    except FileNotFoundError:
        logger.warning("symbols.csv not found at %s", SYMBOLS_PATH)
        return rules, None

    try:
        with open(SYMBOLS_PATH, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                symbol = (row.get("symbol") or "").strip().upper()
                formula = (row.get("formula") or "").strip().lower()
                if not symbol or formula not in {"lots", "money"}:
                    logger.warning("Skipping invalid row in symbols.csv: %s", row)
                    continue
                raw_limit = (row.get("volume_limit") or "0").replace(" ", "")
                raw_contract = (row.get("contract_size") or "0").strip()
                try:
                    volume_limit = float(raw_limit.replace(",", "."))
                    contract_size = int(raw_contract)
                except (ValueError, AttributeError):
                    logger.warning("Skipping row with invalid numbers: %s", row)
                    continue
                rules[symbol] = SymbolRule(
                    symbol=symbol,
                    formula=formula,
                    volume_limit=volume_limit,
                    contract_size=contract_size,
                )
        logger.info("Loaded %d symbol rule(s)", len(rules))
        return rules, mtime
    except Exception:
        logger.exception("Failed to read symbols.csv")
        return rules, None


SYMBOL_RULES, SYMBOLS_MTIME = load_symbol_rules()


def ensure_symbols_watch() -> None:
    global SYMBOLS_MTIME
    try:
        current = os.path.getmtime(SYMBOLS_PATH)
    except FileNotFoundError:
        current = None

    if SYMBOLS_MTIME is None:
        SYMBOLS_MTIME = current
        if current is not None:
            logger.info("symbols.csv appeared; restarting to load rules")
            restart_program()
        return

    if current != SYMBOLS_MTIME:
        logger.info("Detected change in symbols.csv")
        restart_program()


def main():
    logger.info("Starting main loop")
    done = load_uids()

    while True:
        try:
            ensure_symbols_watch()
            logger.debug("Connecting to IMAP...")
            with imaplib.IMAP4_SSL(IMAP_SERVER) as im:
                im.login(IMAP_USER, IMAP_PASS)
                logger.info("IMAP login successful")
                im.select("INBOX")
                process_box(im, done)
        except Exception:
            logger.exception("IMAP cycle failed")

        logger.debug(f"Sleeping for {CHECK_SEC} seconds...")
        time.sleep(CHECK_SEC)

# ─────── ENTRYPOINT ───────
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Fatal error in main()")
