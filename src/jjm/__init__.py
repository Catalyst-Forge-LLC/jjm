"""JJM (Just Judgment Machine): an open-source local judgment runtime.

Typed findings over explicit criteria, with method, provenance, uncertainty, and
acceptance checks. JJM does not generate prose on the decision path and never
executes an action. The application still decides.
"""

__version__ = "0.1.0.dev0"
CONTRACT_VERSION = "0.1"
ENGINE_VERSION = f"jjm-engine {__version__}"

__all__ = ["__version__", "CONTRACT_VERSION", "ENGINE_VERSION"]
