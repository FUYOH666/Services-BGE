"""MCP server that turns Services-BGE into shared long-term memory and search for AI agents."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("services-bge")
except PackageNotFoundError:  # running from a plain checkout
    __version__ = "0+local"
