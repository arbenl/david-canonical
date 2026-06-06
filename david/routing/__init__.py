"""Routing-layer alias for engine.router. Exists so that thematic imports

  from david.routing import forecast_router

work the same as

  from david.engine.router import apply_forecast_routing
"""

from ..engine import router as forecast_router  # re-export

__all__ = ["forecast_router"]
