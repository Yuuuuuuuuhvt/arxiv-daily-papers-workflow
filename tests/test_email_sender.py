from __future__ import annotations

import asyncio
import json
import smtplib
import ssl
from email import message_from_string
from types import SimpleNamespace

import pytest

from src import email_sender, main
from src.models import AnalysisResult


MAIL = {
    "MAIL_SMTP_HOST": "mail.example.test", "MAIL_SMTP_PORT": "587",
    "MAIL_USERNAME": "login@example.test", "MAIL_PASSWORD": "private-test-password",
    "MAIL_FROM": "sender@example.test", "MAIL_TO": "reader@example.test",
}


def send() -> bool:
    return email_sender.send_digest(
        "日报测试", "<p>日报正文</p>", MAIL["MAIL_TO"], MAIL["MAIL_FROM"],
        MAIL["MAIL_USERNAME"], MAIL["MAIL_PASSWORD"], MAIL["MAIL_SMTP_HOST"], MAIL["MAIL_SMTP_PORT"],
    )


def test_starttls_precedes_auth_and_preserves_distinct_addresses(monkeypatch):
    events = []
    class SMTP:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == (MAIL["MAIL_SMTP_HOST"], 587, 30)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def ehlo(self):
            events.append("ehlo")
        def starttls(self, *, context):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            events.append("tls")
        def login(self, username, password):
            assert (username, password) == (MAIL["MAIL_USERNAME"], MAIL["MAIL_PASSWORD"])
            events.append("login")
        def sendmail(self, sender, recipients, body):
            assert (sender, recipients) == (MAIL["MAIL_FROM"], [MAIL["MAIL_TO"]])
            msg = message_from_string(body)
            assert (msg["From"], msg["To"]) == (sender, recipients[0])
            assert msg["Date"] and msg["Message-ID"]
            assert msg.get_payload()[0].get_payload(decode=True).decode() == "<p>日报正文</p>"
            events.append("send")
    monkeypatch.setattr(email_sender.smtplib, "SMTP", SMTP)
    assert send() is True
    assert events == ["ehlo", "tls", "ehlo", "login", "send"]


def test_smtp_errors_return_false_without_leaking_response(monkeypatch, caplog):
    for error in [TimeoutError("timeout"), ssl.SSLError("certificate"),
                  smtplib.SMTPAuthenticationError(535, MAIL["MAIL_PASSWORD"].encode()),
                  smtplib.SMTPNotSupportedError("STARTTLS unavailable"),
                  smtplib.SMTPRecipientsRefused({MAIL["MAIL_TO"]: (550, b"rejected")})]:
        def fail(*args, **kwargs):
            raise error
        monkeypatch.setattr(email_sender.smtplib, "SMTP", fail)
        assert send() is False
    assert MAIL["MAIL_PASSWORD"] not in caplog.text


def test_invalid_port_never_opens_connection(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid port must not connect")
    monkeypatch.setattr(email_sender.smtplib, "SMTP", unexpected)
    for port in ["not-a-port", "0", "65536"]:
        assert email_sender.send_digest("s", "html", "a@b.test", "a@b.test", "u", "p", "host", port) is False


@pytest.mark.parametrize("mode", ["missing", "skip", "failed", "sent"])
def test_pipeline_keeps_report_and_records_actual_mail_outcome(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    monkeypatch.setattr(main, "setup_logging", lambda: None)
    config = {"models": {"relevance_filter": {}, "deep_analysis": {}}, "scoring": {}}
    monkeypatch.setattr(main, "load_config", lambda: config)
    paper = SimpleNamespace(arxiv_id="test-paper")
    monkeypatch.setattr(main, "hybrid_fetch", lambda *args: [paper])
    monkeypatch.setattr(main, "load_index", lambda *args: {})
    monkeypatch.setattr(main, "dedup", lambda papers, index: papers)
    async def relevance(*args):
        return [paper], []
    async def analysis(*args):
        return AnalysisResult()
    monkeypatch.setattr(main, "filter_relevance", relevance)
    monkeypatch.setattr(main, "analyze_paper", analysis)
    monkeypatch.setattr(main, "paper_to_index_entry", lambda *args, **kwargs: {})
    monkeypatch.setattr(main, "save_index", lambda *args: None)
    monkeypatch.setattr(main, "generate_daily_report", lambda *args: "完整日报")
    monkeypatch.setattr(main, "save_report", lambda body, date: (tmp_path / "report.md").write_text(body))
    monkeypatch.setattr(main, "generate_email_html", lambda *args: "<p>日报</p>")
    calls = []
    def sender(**kwargs):
        calls.append(kwargs)
        return mode == "sent"
    monkeypatch.setattr(main, "send_digest", sender)
    for key, value in MAIL.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("REPORT_DATE", "2026-10-14")
    monkeypatch.setenv("SKIP_DEEP_RESEARCH", "1")
    monkeypatch.setenv("SKIP_GIT_PUSH", "1")
    monkeypatch.setenv("MAX_PAPERS", "0")
    monkeypatch.setenv("SKIP_EMAIL", "1" if mode == "skip" else "0")
    if mode == "missing":
        monkeypatch.delenv("MAIL_PASSWORD")
    asyncio.run(main.run_pipeline())
    assert (tmp_path / "report.md").read_text() == "完整日报"
    summary = json.loads((tmp_path / "run-metadata/2026-10-14.json").read_text())
    assert summary["status"] == "completed"
    assert summary["email_sent"] is (mode == "sent")
    assert len(calls) == (1 if mode in {"failed", "sent"} else 0)
    if calls:
        assert calls[0]["to_addr"] == MAIL["MAIL_TO"]
        assert calls[0]["from_addr"] == MAIL["MAIL_FROM"]
