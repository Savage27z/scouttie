"""Telegram alerts via the Bot HTTP API (httpx; no extra dependency).

Security notes:
* Every dynamic field (bio, name, LLM summary) is stranger-controlled text, so it is
  HTML-escaped before going into a message. An unescaped bio could otherwise inject
  tags or a misleading link into your alert.
* Button callbacks are honoured ONLY when they come from the configured chat. Anyone
  can find a bot and press buttons on a forwarded message or message it directly.
* Callback data is a short opcode + account id, validated against a strict pattern
  before it touches the database.
"""

from __future__ import annotations

import html
import logging
import re
import time
from typing import Any

import httpx

from ..db import Database

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_MESSAGE = 4000  # hard limit is 4096; leave headroom

# opcode -> candidate state written back. These human labels are the training data
# for tuning the gate, so "not_project" is kept distinct from the LLM's "rejected".
ACTIONS = {"i": "interested", "n": "not_project", "m": "muted"}
_CALLBACK = re.compile(r"^([inm]):(\d{1,25})$")


CONTRACTS_LABEL = {
    "yes": "✅ <b>Own smart contracts:</b> yes",
    "no": "⛔ <b>Own smart contracts:</b> no (front-end / aggregator: little to audit)",
    "unclear": "❔ <b>Own smart contracts:</b> unclear",
}


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=False)


def keyboard(account_id: str) -> dict[str, Any]:
    return {
        "inline_keyboard": [[
            {"text": "⭐ Interested", "callback_data": f"i:{account_id}"},
            {"text": "🚫 Not a project", "callback_data": f"n:{account_id}"},
            {"text": "🔇 Mute", "callback_data": f"m:{account_id}"},
        ]]
    }


def format_alert(c: dict[str, Any], *, kind: str = "digest") -> str:
    """c: candidate joined with its account (see Database.candidates_by_state)."""
    head = "🚨 <b>New project following the ecosystem</b>" if kind == "instant" else "📌 <b>New project</b>"
    handle = esc(c.get("username"))
    lines = [
        head,
        f"<b>{esc(c.get('name'))}</b>  (<a href=\"https://x.com/{handle}\">@{handle}</a>)",
    ]
    if c.get("llm_summary"):
        lines.append(f"\n{esc(c['llm_summary'])}")
    elif c.get("bio"):
        lines.append(f"\n<i>{esc(str(c['bio'])[:300])}</i>")

    # The auditor's first question. Fixed labels only: nothing model-written is echoed.
    contracts = CONTRACTS_LABEL.get(c.get("llm_contracts") or "")
    if contracts:
        lines.append("\n" + contracts)

    meta = []
    if c.get("llm_category"):
        meta.append(esc(c["llm_category"]))
    if c.get("llm_chain") and c["llm_chain"] != "unknown":
        meta.append(esc(c["llm_chain"]))
    if c.get("llm_stack") and c["llm_stack"] != "unknown":
        meta.append("stack: " + esc(c["llm_stack"]))
    if meta:
        lines.append("\n" + " · ".join(meta))

    followed = c.get("followed") or []
    if followed:
        lines.append("Followed: " + ", ".join("@" + esc(t) for t in followed))
    if c.get("url"):
        lines.append("Link: " + esc(c["url"]))
    lines.append(
        f"{esc(c.get('followers_count'))} followers · created {esc(str(c.get('created_at') or '?')[:10])}"
        f" · score {esc(c.get('score'))}"
        + (f" · conf {float(c['llm_confidence']):.2f}" if c.get("llm_confidence") is not None else "")
    )
    return "\n".join(lines)[:MAX_MESSAGE]


class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str, chat_id: str, client: httpx.Client | None = None):
        if not token or not chat_id:
            raise TelegramError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are required")
        try:
            self.chat_id = int(chat_id)
        except ValueError as exc:
            raise TelegramError("TELEGRAM_CHAT_ID must be numeric") from exc
        self._base = f"{API}/bot{token}"
        self._client = client or httpx.Client(timeout=40.0)
        self._last_send = 0.0

    def _call(self, method: str, **payload: Any) -> Any:
        try:
            resp = self._client.post(f"{self._base}/{method}", json=payload)
        except httpx.HTTPError as exc:
            # str(exc) can contain the URL, which contains the bot token. Don't log it.
            raise TelegramError(f"{method}: network error ({type(exc).__name__})") from None
        try:
            body = resp.json()
        except ValueError:
            raise TelegramError(f"{method}: non-JSON response ({resp.status_code})") from None
        if not body.get("ok"):
            raise TelegramError(f"{method}: {body.get('description', 'error')}")
        return body["result"]

    def send(self, text: str, *, account_id: str | None = None) -> int:
        # Telegram allows ~1 message/second to one chat.
        wait = 1.1 - (time.monotonic() - self._last_send)
        if wait > 0:
            time.sleep(wait)
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if account_id:
            payload["reply_markup"] = keyboard(account_id)
        try:
            result = self._call("sendMessage", **payload)
        finally:
            self._last_send = time.monotonic()
        return int(result["message_id"])

    def get_updates(self, offset: int | None, timeout: int = 0) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"timeout": timeout, "allowed_updates": ["callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return self._call("getUpdates", **params)

    def answer_callback(self, callback_id: str, text: str) -> None:
        try:
            self._call("answerCallbackQuery", callback_query_id=callback_id, text=text[:150])
        except TelegramError as exc:
            log.debug("answerCallbackQuery: %s", exc)

    def clear_buttons(self, message_id: int) -> None:
        try:
            self._call("editMessageReplyMarkup", chat_id=self.chat_id, message_id=message_id,
                       reply_markup={"inline_keyboard": []})
        except TelegramError as exc:
            log.debug("editMessageReplyMarkup: %s", exc)


def handle_callback(db: Database, tg: Telegram, cb: dict[str, Any]) -> str | None:
    """Apply one button press. Returns the new state, or None if ignored."""
    chat = ((cb.get("message") or {}).get("chat") or {}).get("id")
    if chat != tg.chat_id:
        log.warning("ignoring callback from foreign chat %r", chat)
        return None
    m = _CALLBACK.match(cb.get("data") or "")
    if not m:
        return None
    state = ACTIONS[m.group(1)]
    account_id = m.group(2)
    if db.get_candidate(account_id) is None:
        tg.answer_callback(cb["id"], "Unknown candidate")
        return None
    db.set_candidate_state(account_id, state)
    tg.answer_callback(cb["id"], f"Marked {state.replace('_', ' ')}")
    mid = (cb.get("message") or {}).get("message_id")
    if mid:
        tg.clear_buttons(int(mid))
    return state


def poll_callbacks(db: Database, tg: Telegram, offset: int | None) -> int | None:
    """Drain pending button presses. Returns the next update offset."""
    for upd in tg.get_updates(offset, timeout=0):
        offset = int(upd["update_id"]) + 1
        cb = upd.get("callback_query")
        if cb:
            handle_callback(db, tg, cb)
    return offset
