import pytest
import json
from unittest.mock import patch, AsyncMock


def test_slack_post_no_webhook(client):
    """POST /run_digest with no Slack webhook configured returns ok."""
    with patch("app.integrations.slack.settings") as mock_settings:
        mock_settings.SLACK_WEBHOOK_URL = ""
        r = client.post("/run_digest")
        assert r.status_code == 200
        assert r.json()["status"] == "digest_sent"


@pytest.mark.asyncio
async def test_slack_payload_structure():
    """Verify Slack block structure for digest."""
    from app.integrations.slack import post_digest

    mentions = [
        {
            "id": "1",
            "title": "Vitalik.eth launches $ETH upgrade",
            "summary": "ETH 2.0 upgrade completed successfully.",
            "source": "CoinDesk",
            "reach": 100000,
            "sentiment": "positive",
            "web3_signals": {
                "tickers": ["$ETH"],
                "ens_names": [{"ens": "vitalik.eth", "valid": True}],
                "eth_addresses": []
            }
        }
    ]

    sent_payload = None

    async def mock_post(url, json=None, **kwargs):
        nonlocal sent_payload
        sent_payload = json

        class MockResp:
            def raise_for_status(self): pass
        return MockResp()

    with patch("app.integrations.slack.settings") as mock_settings:
        mock_settings.SLACK_WEBHOOK_URL = "https://hooks.slack.com/test"
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client
            await post_digest(mentions)

    assert sent_payload is not None
    assert "blocks" in sent_payload
    blocks = sent_payload["blocks"]
    # Header block
    assert blocks[0]["type"] == "header"
    # At least one section block
    section_types = [b["type"] for b in blocks]
    assert "section" in section_types


@pytest.mark.asyncio
async def test_slack_skips_when_no_webhook():
    """No HTTP call if webhook not configured."""
    from app.integrations.slack import post_digest

    with patch("app.integrations.slack.settings") as mock_settings:
        mock_settings.SLACK_WEBHOOK_URL = ""
        with patch("httpx.AsyncClient") as mock_client:
            await post_digest([{"id": "1", "title": "Test", "summary": "S", "reach": 0, "sentiment": "positive", "web3_signals": {}}])
            mock_client.assert_not_called()
