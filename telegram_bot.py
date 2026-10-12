"""Telegram front door to the robot: chat, ask for photos / the map, start AI tasks, stop.

Plain HTTPS long polling (no webhook, no public URL, no extra library). Only chat ids listed in
TELEGRAM_ALLOWED_IDS are served; anyone else just gets told their own id so you can add it.
The robot only moves after you press the Start button on a proposed task."""
import threading
import time

import requests

API = "https://api.telegram.org/bot{token}/{method}"
HELP = ("I'm your robot. Just talk to me - ask what I see, or tell me what to do.\n\n"
        "/photo - what the camera sees now\n/map - the map\n/status - where I am\n"
        "/stop - stop moving immediately\n/new - start a fresh conversation\n\n"
        "When you ask for a task I'll show a ▶ Start button; the robot only moves after you press it.")


class TelegramBot(threading.Thread):
    def __init__(self, token, allowed, h, log=print):
        """h: dict of callbacks - context() -> (status, map_png|None, cam_jpg|None); turn(text) -> (reply, actions);
        new_chat(); start_task(task) -> (ok, error); stop(); ai_state() -> (count, [log lines])."""
        super().__init__(daemon=True)
        self.token, self.allowed, self.h, self.log = token, set(allowed), h, log
        self.pending = {}            # callback id -> task text
        self.seq = 0
        self.http = requests.Session()

    # ── Telegram API ──
    def call(self, method, http_timeout=40, **params):
        r = self.http.post(API.format(token=self.token, method=method), json=params, timeout=http_timeout)
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"{method}: {data.get('description')}")
        return data["result"]

    def say(self, chat, text, keyboard=None):
        for i in range(0, max(len(text), 1), 3900):                      # Telegram limit is 4096 chars
            kw = {"reply_markup": {"inline_keyboard": keyboard}} if keyboard and i + 3900 >= len(text) else {}
            self.call("sendMessage", chat_id=chat, text=text[i:i + 3900] or "(empty)", **kw)

    def photo(self, chat, data, caption=""):
        r = self.http.post(API.format(token=self.token, method="sendPhoto"), data={"chat_id": chat, "caption": caption[:900]},
                           files={"photo": ("image.jpg", data)}, timeout=60)
        if not r.json().get("ok"):
            raise RuntimeError("sendPhoto: " + str(r.json().get("description")))

    # ── incoming ──
    def run(self):
        self.log("[telegram] bot started" + ("" if self.allowed else " - TELEGRAM_ALLOWED_IDS is empty: nobody can use it yet"))
        threading.Thread(target=self._notifier, daemon=True).start()
        offset = None
        while True:
            try:
                args = {"timeout": 25} if offset is None else {"timeout": 25, "offset": offset}     # Telegram's long-poll seconds
                for upd in self.call("getUpdates", http_timeout=40, **args):
                    offset = upd["update_id"] + 1
                    try:
                        self.handle(upd)
                    except Exception as e:
                        self.log(f"[telegram] handler error: {e!r}")
            except Exception as e:
                self.log(f"[telegram] polling error: {e!r} - retrying")
                time.sleep(5)

    def handle(self, upd):
        cq = upd.get("callback_query")
        if cq:
            chat = cq["message"]["chat"]["id"]
            if chat not in self.allowed:
                return
            self.call("answerCallbackQuery", callback_query_id=cq["id"])
            task = self.pending.pop(cq["data"], None)
            if task is None:
                return self.say(chat, "That button has expired - ask me again.")
            ok, err = self.h["start_task"](task)
            return self.say(chat, f"▶ Started: {task}\nI'll report each step. Send /stop to halt." if ok else f"Could not start: {err}")
        msg = upd.get("message")
        if not msg or "text" not in msg:
            return
        chat, text = msg["chat"]["id"], msg["text"].strip()
        if chat not in self.allowed:
            return self.say(chat, f"Not authorised. Your chat id is {chat}. To allow it, restart the robot server with "
                                  f"TELEGRAM_ALLOWED_IDS={chat}")
        cmd = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
        if cmd in ("/start", "/help"):
            return self.say(chat, HELP)
        if cmd == "/stop":
            self.h["stop"]()
            return self.say(chat, "■ Stopped.")
        if cmd == "/new":
            self.h["new_chat"]()
            return self.say(chat, "New conversation.")
        status, png, cam = self.h["context"]()
        if cmd == "/status":
            return self.say(chat, status)
        if cmd == "/photo":
            return self.photo(chat, cam, "What I see now") if cam else self.say(chat, "Camera not available.")
        if cmd == "/map":
            return self.photo(chat, png, "The map") if png else self.say(chat, "No map yet.")
        self.call("sendChatAction", chat_id=chat, action="typing")
        reply, actions = self.h["turn"](text)
        for a in actions:
            if a["type"] == "propose":
                self.seq += 1
                key = f"go:{self.seq}"
                self.pending[key] = a["task"]
                reply += f"\n\nProposed task: {a['task']}\nThe robot will drive by itself (keep it away from stairs and watch it)."
                self.say(chat, reply, [[{"text": "▶ Start", "callback_data": key}]])
                reply = None
            elif a["type"] == "pinned":
                self.say(chat, f"📷 Photo pinned: {a['note']}")
        if reply:
            self.say(chat, reply)

    # ── outgoing: the AI's own decisions ──
    def _notifier(self):
        seen = 0
        while True:
            time.sleep(2)
            try:
                n, lines = self.h["ai_state"]()
                if n < seen:
                    seen = n
                if n > seen:
                    fresh = lines[-min(n - seen, len(lines)):]
                    seen = n
                    for chat in self.allowed:
                        self.say(chat, "🤖 " + "\n".join(fresh))
            except Exception as e:
                self.log(f"[telegram] notify error: {e!r}")
