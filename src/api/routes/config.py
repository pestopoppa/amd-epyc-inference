"""Runtime configuration endpoint for hot-reloading feature flags."""

from fastapi import APIRouter, Request

from src.features import features, set_features, Features

router = APIRouter()


@router.post("/config")
async def update_config(request: Request):
    """Update feature flags at runtime without restarting the API.

    Accepts a JSON body with feature flag names as keys and booleans as values.
    Only known feature flags are applied; unknown keys are ignored.

    Returns:
        Updated feature flag summary.
    """
    body = await request.json()
    current = features()
    current_summary = current.summary()
    overrides = {k: bool(v) for k, v in body.items() if k in current_summary}
    new = Features(**{**current_summary, **overrides})
    set_features(new)
    return {"status": "ok", "features": new.summary()}
