# main.py
# from __future__ import annotations

from typing import List, Dict, Any
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# Import your packing helper (from the uploaded pack_solution.py)
from pack_solution import choose_container  # noqa: F401

app = Flask(__name__)


def _require_fields(data: Dict[str, Any], fields: List[str]) -> None:
    missing = [f for f in fields if f not in data]
    if missing:
        raise BadRequest("Missing required field(s): " + ", ".join(missing))


def _validate_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes the incoming JSON payload.
    Expected:
      - sample_mailers: List[str]
      - sample_boxes:   List[str]
      - sample_items:   List[{id:str, dimension:str, quantity:int>=1}]
      - flat_height:    int
      - box_offset:     int
      - mailer_offset:  int
    """
    _require_fields(
        data,
        [
            "sample_mailers",
            "sample_boxes",
            "sample_items",
            "flat_height",
            "box_offset",
            "mailer_offset",
        ],
    )

    sample_mailers = data["sample_mailers"]
    sample_boxes = data["sample_boxes"]
    sample_items = data["sample_items"]
    flat_height = data["flat_height"]
    box_offset = data["box_offset"]
    mailer_offset = data["mailer_offset"]

    if not isinstance(sample_mailers, list) or not all(isinstance(x, str) for x in sample_mailers):
        raise BadRequest("sample_mailers must be a list of strings")
    if not isinstance(sample_boxes, list) or not all(isinstance(x, str) for x in sample_boxes):
        raise BadRequest("sample_boxes must be a list of strings")
    if not isinstance(sample_items, list):
        raise BadRequest("sample_items must be a list")

    normalized_items: List[Dict[str, Any]] = []
    for idx, it in enumerate(sample_items):
        if not isinstance(it, dict):
            raise BadRequest("sample_items[%d] must be an object" % idx)
        _require_fields(it, ["id", "dimension", "quantity"])
        if not isinstance(it["id"], str):
            raise BadRequest("sample_items[%d].id must be a string" % idx)
        if not isinstance(it["dimension"], str):
            raise BadRequest("sample_items[%d].dimension must be a string" % idx)
        if not isinstance(it["quantity"], int) or it["quantity"] < 1:
            raise BadRequest("sample_items[%d].quantity must be an integer >= 1" % idx)

        # Keep only the expected keys to avoid leaking extras into the solver
        normalized_items.append(
            {"id": it["id"], "dimension": it["dimension"], "quantity": it["quantity"]}
        )

    for name, val in [("flat_height", flat_height), ("box_offset", box_offset), ("mailer_offset", mailer_offset)]:
        if not isinstance(val, int):
            raise BadRequest("%s must be an integer" % name)

    return {
        "sample_mailers": sample_mailers,
        "sample_boxes": sample_boxes,
        "sample_items": normalized_items,
        "flat_height": flat_height,
        "box_offset": box_offset,
        "mailer_offset": mailer_offset,
    }

@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"})

@app.route("/solve", methods=["POST"])
def solve():
    """
    Decide the smallest container that fits the provided items.
    Returns JSON:
      {
        "container": "WxL" or "WxLxH",
        "info": [...] or null
      }
    """
    try:
        data = request.get_json(force=True, silent=False)
    except BadRequest:
        # Malformed JSON only → standardize this message
        return jsonify({"error": "Invalid JSON payload"}), 400

    try:
        payload = _validate_payload(data)
    except BadRequest as e:
        # Validation issues keep their detailed messages
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Any other unexpected parsing/validation error
        return jsonify({"error": f"Invalid JSON payload: {e}"}), 400

    try:
        container, info = choose_container(
            padded_mailers=payload["sample_mailers"],
            boxes=payload["sample_boxes"],
            items=payload["sample_items"],
            debug=False,  # do not plot in server environments
            flat_height=payload["flat_height"],
            box_offset=payload["box_offset"],
            mailer_offset=payload["mailer_offset"],
        )
        return jsonify({"container": container, "info": info})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Catch-all to avoid exposing internals
        return jsonify({"error": "Internal error: %s" % e}), 500


if __name__ == "__main__":
    # Dev server; for production use gunicorn:
    # gunicorn -w 2 -t 120 -b 0.0.0.0:8002 main:app
    app.run(host="0.0.0.0", port=8002)
