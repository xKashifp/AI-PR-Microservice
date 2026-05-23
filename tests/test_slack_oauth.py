import pytest
from unittest.mock import patch, AsyncMock
from app.models.db import db_conn
from fastapi.testclient import TestClient

def test_slack_install_no_config(client):
    """GET /slack/install returns 400 if client id or redirect uri is not set."""
    with patch("app.api.digest.settings") as mock_settings:
        mock_settings.SLACK_CLIENT_ID = ""
        mock_settings.SLACK_REDIRECT_URI = ""
        r = client.get("/slack/install", follow_redirects=False)
        assert r.status_code == 400

def test_slack_install_redirects(client):
    """GET /slack/install redirects to Slack authorize endpoint when credentials are configured."""
    with patch("app.api.digest.settings") as mock_settings:
        mock_settings.SLACK_CLIENT_ID = "testclientid"
        mock_settings.SLACK_REDIRECT_URI = "https://example.com/callback"
        r = client.get("/slack/install", follow_redirects=False)
        assert r.status_code == 307
        assert "https://slack.com/oauth/v2/authorize" in r.headers["location"]
        assert "client_id=testclientid" in r.headers["location"]
        assert "scope=incoming-webhook" in r.headers["location"]

@pytest.mark.asyncio
async def test_slack_oauth_callback_success(client):
    """GET /slack/oauth/callback handles code exchange, inserts in DB, and redirects to success."""
    mock_resp_data = {
        "ok": True,
        "access_token": "xoxb-test-token",
        "team": {"name": "Test Workspace", "id": "T12345"},
        "incoming_webhook": {
            "channel": "#general",
            "channel_id": "C12345",
            "url": "https://hooks.slack.com/services/T1/B1/W1"
        }
    }

    async def mock_post(*args, **kwargs):
        class MockResp:
            def raise_for_status(self): pass
            def json(self): return mock_resp_data
        return MockResp()

    # Clear existing installations
    with db_conn() as conn:
        conn.execute("DELETE FROM slack_oauth_installations")

    with patch("app.api.digest.settings") as mock_settings:
        mock_settings.SLACK_CLIENT_ID = "testclientid"
        mock_settings.SLACK_CLIENT_SECRET = "testsecret"
        mock_settings.SLACK_REDIRECT_URI = "https://example.com/callback"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            # Hit the callback route
            r = client.get("/slack/oauth/callback?code=mycode", follow_redirects=False)
            assert r.status_code == 307
            assert "slack_success=connected" in r.headers["location"]

    # Verify db entry
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM slack_oauth_installations WHERE team_id = 'T12345'").fetchone()
        assert row is not None
        assert row["team_name"] == "Test Workspace"
        assert row["channel_name"] == "#general"
        assert row["access_token"] == "xoxb-test-token"

def test_slack_oauth_callback_fails(client):
    """GET /slack/oauth/callback redirects to error URL if slack API returns error."""
    mock_resp_data = {
        "ok": False,
        "error": "invalid_code"
    }

    async def mock_post(*args, **kwargs):
        class MockResp:
            def raise_for_status(self): pass
            def json(self): return mock_resp_data
        return MockResp()

    with patch("app.api.digest.settings") as mock_settings:
        mock_settings.SLACK_CLIENT_ID = "testclientid"
        mock_settings.SLACK_CLIENT_SECRET = "testsecret"
        mock_settings.SLACK_REDIRECT_URI = "https://example.com/callback"

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=mock_post)
            mock_client_cls.return_value = mock_client

            r = client.get("/slack/oauth/callback?code=badcode", follow_redirects=False)
            assert r.status_code == 307
            assert "slack_error=invalid_code" in r.headers["location"]

def test_slack_oauth_list_installations(client):
    """GET /slack/oauth/installations lists installations without sensitive fields."""
    with db_conn() as conn:
        conn.execute("DELETE FROM slack_oauth_installations")
        conn.execute("""
            INSERT INTO slack_oauth_installations (team_id, team_name, channel_id, channel_name, webhook_url, access_token)
            VALUES ('T99', 'List Team', 'C99', '#alerts', 'https://hooks.slack.com/services/T/B/W', 'xoxb-token')
        """)

    r = client.get("/slack/oauth/installations")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 1
    # Check fields
    inst = data[0]
    assert inst["team_name"] == "List Team"
    assert inst["channel_name"] == "#alerts"
    assert "access_token" not in inst
    assert "webhook_url" not in inst

@pytest.mark.asyncio
async def test_slack_oauth_disconnect(client):
    """POST /slack/oauth/disconnect/{id} revokes token and removes database entry."""
    with db_conn() as conn:
        conn.execute("DELETE FROM slack_oauth_installations")
        conn.execute("""
            INSERT INTO slack_oauth_installations (id, team_id, team_name, channel_id, channel_name, webhook_url, access_token)
            VALUES (42, 'T42', 'Disconnect Team', 'C42', '#temp', 'https://hooks.slack.com/services/T/B/W2', 'xoxb-token-to-revoke')
        """)

    async def mock_post(*args, **kwargs):
        class MockResp:
            def raise_for_status(self): pass
            def json(self): return {"ok": True}
        return MockResp()

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        r = client.post("/slack/oauth/disconnect/42")
        assert r.status_code == 200
        assert r.json()["status"] == "disconnected"

    # Verify deleted from DB
    with db_conn() as conn:
        row = conn.execute("SELECT * FROM slack_oauth_installations WHERE id = 42").fetchone()
        assert row is None
