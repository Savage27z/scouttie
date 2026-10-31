"""Fetch a project's website for the stage-5 summary.

The URL comes from a stranger's X bio and this runs on a VPS, so this is an SSRF
surface. Defences:

* http/https only, no credentials in the URL, no odd ports.
* The host is resolved and EVERY address must be globally routable. That rejects
  loopback, RFC1918, link-local (cloud metadata at 169.254.169.254) and IPv6
  equivalents.
* Redirects are followed manually, re-validating each hop, so a public URL cannot
  bounce us onto an internal one.
* Hard caps on bytes, time, and redirects; HTML only.

Known residual risk: DNS rebinding (the name could resolve differently between our
check and httpx's own connection). Closing it fully means pinning the connection to
the validated IP; for a read-only summariser on a box with no internal services
beyond the metadata endpoint (blocked above), that is an accepted gap -- revisit if
this ever runs somewhere with sensitive internal HTTP services.

The returned text is untrusted DATA. Callers must never treat it as instructions.
"""

from __future__ import annotations

import html
import ipaddress
import logging
import re
import socket
from urllib.parse import urljoin, urlparse

import httpx

log = logging.getLogger(__name__)

MAX_BYTES = 300_000
MAX_REDIRECTS = 3
ALLOWED_PORTS = {None, 80, 443}
USER_AGENT = "Mozilla/5.0 (compatible; ct-follow-scout/0.1)"


class UnsafeURL(ValueError):
    pass


def _resolve_all(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve {host}") from exc
    return sorted({info[4][0] for info in infos})


def check_url_is_public(url: str) -> str:
    """Return the URL if safe to fetch, else raise UnsafeURL."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme {parsed.scheme!r} not allowed")
    if parsed.username or parsed.password:
        raise UnsafeURL("credentials in URL not allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeURL("no host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURL("bad port") from exc
    if port not in ALLOWED_PORTS:
        raise UnsafeURL(f"port {port} not allowed")

    addrs = _resolve_all(host)
    if not addrs:
        raise UnsafeURL("no addresses")
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError as exc:
            raise UnsafeURL(f"unparseable address {addr}") from exc
        # An IPv4-mapped IPv6 address must be judged as the IPv4 it wraps.
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise UnsafeURL(f"{host} resolves to non-public address {ip}")
    return url


_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg)\b.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_META_DESC = re.compile(
    r"""<meta[^>]+name=["']description["'][^>]+content=["'](.*?)["']""", re.I | re.S
)


def html_to_text(raw: str, max_chars: int = 6000) -> str:
    title = _TITLE.search(raw)
    desc = _META_DESC.search(raw)
    body = _SCRIPT_STYLE.sub(" ", raw)
    body = _TAGS.sub(" ", body)
    body = html.unescape(_WS.sub(" ", body)).strip()
    head = []
    if title:
        head.append("TITLE: " + html.unescape(_WS.sub(" ", title.group(1))).strip())
    if desc:
        head.append("DESCRIPTION: " + html.unescape(_WS.sub(" ", desc.group(1))).strip())
    return ("\n".join(head) + "\n" + body)[:max_chars].strip()


def fetch_text(url: str, *, timeout_s: float = 8.0, client: httpx.Client | None = None) -> str | None:
    """Fetch and flatten a page. Returns None on ANY failure -- the summary then
    degrades to 'unavailable' rather than blocking the alert."""
    own = client is None
    client = client or httpx.Client(
        timeout=timeout_s, follow_redirects=False, headers={"User-Agent": USER_AGENT}
    )
    try:
        current = url if "://" in url else "https://" + url
        for _hop in range(MAX_REDIRECTS + 1):
            check_url_is_public(current)
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    loc = resp.headers.get("location")
                    if not loc:
                        return None
                    current = urljoin(current, loc)
                    continue
                if resp.status_code >= 400:
                    return None
                ctype = resp.headers.get("content-type", "").lower()
                if "html" not in ctype and "text" not in ctype:
                    return None
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) >= MAX_BYTES:
                        break
                return html_to_text(bytes(buf).decode(resp.encoding or "utf-8", errors="replace"))
        log.info("too many redirects for %s", url)
        return None
    except (UnsafeURL, httpx.HTTPError, httpx.InvalidURL, OSError, ValueError) as exc:
        log.info("fetch skipped for %s: %s", url, exc)
        return None
    finally:
        if own:
            client.close()


# ---------------------------------------------------------------- short links

_META_REFRESH = re.compile(r"""url=([^"'>\s]+)""", re.I)


def expand_short_url(url: str, *, timeout_s: float = 5.0, client: httpx.Client | None = None) -> str | None:
    """Resolve ONE hop of a shortener (t.co) by reading its redirect, without ever
    fetching the destination. The provider returns only t.co links for profile
    URLs, so without this the custom-domain signal is blind.

    Only the shortener itself is contacted (after the same public-address check as
    everything else); the destination is returned as a string and never requested,
    so a hostile profile link cannot make us hit an arbitrary host from here.
    Returns None on any failure; the caller then treats the link as unresolved.
    """
    own = client is None
    client = client or httpx.Client(
        timeout=timeout_s, follow_redirects=False, headers={"User-Agent": "curl/8.0"}
    )
    try:
        check_url_is_public(url)
        resp = client.get(url)
        target = resp.headers.get("location")
        if not target and "html" in resp.headers.get("content-type", "").lower():
            m = _META_REFRESH.search(resp.text[:4000])
            target = html.unescape(m.group(1)) if m else None
        if not target:
            return None
        target = target.strip()
        if urlparse(target).scheme not in ("http", "https"):
            return None
        return target[:500]
    except (UnsafeURL, httpx.HTTPError, httpx.InvalidURL, OSError, ValueError) as exc:
        log.debug("could not expand %s: %s", url, exc)
        return None
    finally:
        if own:
            client.close()


def make_expander(timeout_s: float = 5.0):
    """An expander closure that reuses one HTTP client across many links."""
    client = httpx.Client(timeout=timeout_s, follow_redirects=False, headers={"User-Agent": "curl/8.0"})
    return lambda url: expand_short_url(url, client=client)
