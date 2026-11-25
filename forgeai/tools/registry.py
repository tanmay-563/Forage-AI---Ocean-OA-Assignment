from __future__ import annotations
import ast
import csv
import operator
from datetime import date
from pathlib import Path
import httpx
from ..config import ensure_within


class ToolRegistry:
    def __init__(self): self._tools = {}
    def register(self, name, description, schema, function): self._tools[name] = {"name": name, "description": description, "schema": schema, "execute": function}
    def list(self): return [{k: v for k, v in tool.items() if k != "execute"} for tool in self._tools.values()]
    def execute(self, name, args):
        if name not in self._tools: raise ValueError(f"Tool not registered: {name}")
        if not isinstance(args, dict): raise ValueError("Tool arguments must be an object")
        required = set(self._tools[name]["schema"])
        if not required.issubset(args): raise ValueError(f"Missing tool arguments: {sorted(required - set(args))}")
        if set(args) - required - ({"limit"} if name == "csv_lookup" else set()): raise ValueError("Unexpected tool arguments")
        return self._tools[name]["execute"](**args)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
def safe_calculate(expression: str) -> float:
    if not isinstance(expression, str) or len(expression) > 120: raise ValueError("Expression must be a string of at most 120 characters")
    def visit(node):
        if isinstance(node, ast.Expression): return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(visit(node.right)) > 8: raise ValueError("Exponent is too large")
            result = _OPS[type(node.op)](visit(node.left), visit(node.right))
            if abs(result) > 1e15: raise ValueError("Result is too large")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS: return _OPS[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic is allowed")
    try: return visit(ast.parse(expression, mode="eval"))
    except (SyntaxError, ZeroDivisionError, OverflowError) as exc: raise ValueError(f"Invalid arithmetic expression: {exc}") from exc


def csv_lookup(root: Path, filename: str, query: str, limit: int = 10):
    if not isinstance(query, str) or not query.strip(): raise ValueError("query must not be empty")
    if not 1 <= int(limit) <= 100: raise ValueError("limit must be between 1 and 100")
    path = ensure_within(root, root / filename)
    if path.suffix.lower() != ".csv" or not path.is_file(): raise ValueError("Only an existing approved CSV can be searched")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    needle = query.casefold()
    return [{**row, "row": number + 2} for number, row in enumerate(rows) if needle in " ".join(str(x) for x in row.values()).casefold()][:int(limit)]


def build_registry(collection_root: Path | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("calculator", "Evaluate safe basic arithmetic", {"expression": "string"}, lambda expression: safe_calculate(expression))
    registry.register("current_date", "Return the local system date", {}, lambda: date.today().isoformat())
    def weather(latitude: float, longitude: float):
        try: latitude, longitude = float(latitude), float(longitude)
        except (TypeError, ValueError) as exc: raise ValueError("Latitude and longitude must be numeric") from exc
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180: raise ValueError("Coordinates are out of range")
        response = httpx.get("https://api.open-meteo.com/v1/forecast", params={"latitude": latitude, "longitude": longitude, "current": "temperature_2m,precipitation,wind_speed_10m", "timezone": "UTC"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data.get("current"), dict): raise ValueError("Weather service returned a malformed response")
        return data["current"]
    registry.register("weather_forecast", "Fetch current weather from Open-Meteo using coordinates", {"latitude": "number", "longitude": "number"}, weather)
    if collection_root:
        root = collection_root.resolve()
        registry.register("csv_lookup", "Search an approved CSV document", {"filename": "relative path", "query": "string", "limit": "int"}, lambda filename, query, limit=10: csv_lookup(root, filename, query, limit))
    return registry
