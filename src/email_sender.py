from __future__ import annotations

import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, make_msgid

logger = logging.getLogger(__name__)


def send_digest(
    subject: str,
    html_body: str,
    to_addr: str,
    from_addr: str,
    username: str,
    password: str,
    smtp_host: str,
    smtp_port: str | int,
) -> bool:
    """通过 STARTTLS 提交日报，失败时返回 False，保留已生成的报告。"""
    try:
        port = int(smtp_port)
        if not 1 <= port <= 65535:
            raise ValueError("SMTP port out of range")
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Date"] = formatdate(localtime=False)
        msg["Message-ID"] = make_msgid(domain=from_addr.rsplit("@", 1)[-1])
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(smtp_host, port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
            server.login(username, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("Email sent to %s, message_id=%s", to_addr, msg["Message-ID"])
        return True
    except (OSError, ValueError) as exc:
        # SMTP 响应可能回显认证数据，仅记录异常类型和协议状态码。
        logger.error("Email failed: %s, smtp_code=%s", type(exc).__name__, getattr(exc, "smtp_code", None))
        return False
