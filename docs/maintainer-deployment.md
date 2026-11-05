# Maintainer deployment

This page records the existing maintainer infrastructure. New users should start with the root README or the self-hosting guide. These hostnames, accounts and repositories are not prerequisites for running your own instance.

## Forgejo production workflow and backfill

Production now uses `.forgejo/workflows/daily-run.yml` on the TX Linux Runner.
The `PAPERS_WRITE_TOKEN` secret supplies `FORGEJO_PAPERS_TOKEN`, which grants write access only to this code repository and the
`embodied-daily-papers` data repository. The pipeline, PDF release attachments,
data push and submodule pointer push all use Forgejo. The retained GitHub workflow
stays disabled during migration.

生产邮件使用上表的六个 Forgejo Actions secrets，来源为 recitativo 的
`secret/mail/eai-papers.env`。任一变量缺失时仅跳过邮件，采集、报告、PDF 发布和数据更新继续执行。
`SKIP_EMAIL=1` 显式禁用邮件。无效端口、TLS、认证或发送失败会记录错误并令
`data/run-metadata` 中的 `email_sent` 为 false，不删除已生成的报告。
SMTP 使用系统证书校验和 30 秒连接及套接字操作超时，不自动重试以避免重复投递。
TX 的 Runner 使用 host 网络，邮件域名由 `DOMAIN,mail.luonto.me,DIRECT` 规则直连。
邮箱配额为 1024 MiB，本域收发不经过 SES 外部中继。

SMTP 调用依据 [Python smtplib 文档](https://docs.python.org/3/library/smtplib.html#smtplib.SMTP.starttls)，
先建立 TLS，再重新 EHLO 并认证。`email_sent` 仅表示 SMTP 接受提交，实际收件仍须经邮箱回读验证。

Manual input `report_date` selects a UTC first-submission date in `YYYY-MM-DD`
format. Historical runs query every page for that date and do not combine the
current RSS feed with historical results. API failures fail the run instead of
being recorded as empty days. The resulting report and `data/run-metadata` record
include the requested date and processing outcome. A weekday run without an
explicit date retains the current API/RSS feed behavior.
An empty retry preserves an existing report and its corresponding run summary.
A legacy report without a summary receives a `retained_existing_report` marker
in the newly recorded attempt summary.

The `validation` input limits analysis to two new papers, skips DeepResearch,
email, release uploads and production pushes. It verifies real collection and
analysis in an isolated job checkout before production scheduling is accepted.

Historical collection selects the first UTC submission date and analyzes the
currently available arXiv version. It does not reconstruct past RSS entries or
past versions. Backfill uses skip_email to prevent delayed notification bursts.
PDF attachments are verified against existing files and published to both
Forgejo and the GitHub fallback before production data commits.


## 日报邮件独立验证

手动运行 `.forgejo/workflows/mail-check.yml`，输入已有日报日期 `report_date`。
作业只读取该日报并发送一封验证邮件，不调用模型，不发布附件，不写入生产分支。
当前收发使用同一邮箱，作业通过 IMAP 993 回读 INBOX，核对完整正文并输出认证邮件头。
SMTP 提交成功与收件确认分别验证，域内认证结果按实际邮件头记录，不等同于外域投递认证。
