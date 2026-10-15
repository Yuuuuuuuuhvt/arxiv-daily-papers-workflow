# 论文日报自建邮箱切换验收

## 目标与实施结果

按 2026-10-14 用户决定，日报使用 eai-papers@luonto.me 收发。
Zouter-JP 的 mailcow 已通过 API 创建并回读邮箱，active=1，quota=1073741824 字节，即 1024 MiB。
凭据保存于 TX 私有 recitativo 的 secret/mail/eai-papers.env，权限为 0600，文件注明日报用途。
Forgejo 已配置 MAIL_SMTP_HOST、MAIL_SMTP_PORT、MAIL_USERNAME、MAIL_PASSWORD、MAIL_FROM、MAIL_TO。
主机为 mail.luonto.me，端口为 587，使用 STARTTLS。代码和工作流已移除 QQ 专用变量及配置。
旧 QQ Actions secret 未被本次代码引用，本次没有删除其历史值。

## 实现与验证范围

功能提交为 a663b7c1f447186f18d94215f2f7880c1db7182c。
发送函数校验证书，设置 30 秒连接及套接字操作超时，返回实际 SMTP 提交状态。
缺少任一邮件变量、显式跳过邮件、端口错误或发送失败均保留已生成的报告。
错误日志只记录异常类型与 SMTP 状态码，避免服务器响应回显凭据。

测试使用原 G4 的 Python 3.12 环境与已安装依赖，没有新增依赖。
执行命令如下，标准输出和错误写入 tests.log，退出码为 0，共 12 项通过。

```sh
docker run --rm --network none \
  -v /home/yurun/workspace/.worktrees/arxiv-p1-mail:/work:ro \
  -v /home/yurun/.local/state/agent-handoff/g4-batch3/paper-tests/.venv:/venv:ro \
  -w /work -e PYTHONPATH=/venv/lib/python3.12/site-packages \
  -e PYTHONDONTWRITEBYTECODE=1 forgejo-linux-validation:20261014 \
  python3 -m pytest -p no:cacheprovider tests/test_email_sender.py tests/test_migration.py -q \
  > tests.log 2>&1; rc=$?
```

测试覆盖 STARTTLS 与认证顺序、系统证书验证、独立发收件地址、网络和认证失败、无效端口、
日志脱敏、邮件缺项及显式跳过、发送失败与成功的运行摘要，以及既有补收报告保留行为。
Python AST、Forgejo YAML 解析、git diff --check 与凭据扫描均已通过。

## Runner 实际收件与邮件头

[Forgejo 独立验证作业](https://git.luonto.me/yurun/arxiv-daily-papers-workflow/actions/runs/11)
在 task/p1-mail 分支运行 mail-check.yml，状态为 success，使用生产 tx-linux Runner 容器与 Actions secrets。
作业读取 2026-10-14 既有日报，发送后通过 IMAP 993 回读 INBOX，确认正文一致。
该日报原文为 300 字节，本次未重新采集论文，也未调用模型、发布附件或修改生产数据。
TX 上另行 IMAP 回读再次确认正文与数据仓库文件一致，SHA256 见 received.json。

Runner 的 host 网络使用 TX 生效的 DOMAIN,mail.luonto.me,DIRECT 规则。
服务器队列 4816A12A67C 的日志确认 TLSv1.3、SMTP 认证身份和 Dovecot LMTP 本地投递成功，未经过 SES。
邮件 Message-ID 为 <179203092192.235.13912102745132107140@luonto.me>。

1. SPF：邮件没有 Authentication-Results 字段，本次没有取得 SPF pass 结果。
2. DKIM：邮件带有 d=luonto.me、s=dkim 的 DKIM-Signature；本次没有独立验证签名的密码学有效性，不能记为 DKIM pass。
3. DMARC：邮件没有 Authentication-Results 字段，本次没有取得 DMARC pass 结果。

本次已经验证本域真实收件和正文一致性，未验证外域收件认证。
不为获得外域认证结果改动邮件服务器策略或发送到任务范围之外的收件人。
已保存 run.json、received.json、postfix.log 和 tests.log；Postfix 日志中的 TX 公网地址已脱敏。

## 后续定时运行

每日工作日 05:30 UTC 的生产定时表达式保持不变。
本次未手动触发正式采集，没有改变 2026-10-15 06:10 UTC 补收安排，也未停止任何现有作业。
下一次定时与补收的完整运行结果尚未验证。
