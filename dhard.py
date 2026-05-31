"""
Snath Aviation — the D_hard curriculum.

Logs each anomaly (TRIGGER_REPLAN) to a signed JSONL queue. When flight logs 
are verified (post-flight ground truth), the queue is labelled. 
"""
from dataclasses import dataclass, asdict
import json, hmac, hashlib
from pathlib import Path

_DHARD_KEY = b"snath_aviation_dhard_2026"
CLASSES = ("pitch_up", "level_flight", "pitch_down")

@dataclass
class DHardEvent:
    asof: str
    scenario_id: str
    decision: str
    basis: float
    conf_a: float
    conf_b: float
    v_a: list
    v_b: list
    realised_outcome: str | None = None
    realised_class: str | None = None
    winner: str | None = None
    sig: str = ""

    _IMMUTABLE = ("asof", "scenario_id", "decision", "basis", "conf_a", "conf_b", "v_a", "v_b")

    def _payload(self) -> bytes:
        return json.dumps({k: getattr(self, k) for k in self._IMMUTABLE}, sort_keys=True).encode()

    def sign(self) -> "DHardEvent":
        self.sig = hmac.new(_DHARD_KEY, self._payload(), hashlib.sha256).hexdigest()
        return self

def _argmax_class(dist: list) -> str:
    return CLASSES[max(range(len(dist)), key=lambda i: dist[i])]

class DHardQueue:
    def __init__(self, path: str = "d_hard.jsonl"):
        self.path = Path(path)

    def clear(self):
        self.path.unlink(missing_ok=True)

    def log(self, scenario_id, v_a, c_a, v_b, c_b, basis, decision, asof):
        dec = decision.value if hasattr(decision, "value") else str(decision)
        if dec != "TRIGGER_REPLAN":
            return None
        ev = DHardEvent(
            asof=asof, scenario_id=scenario_id, decision=dec, basis=round(float(basis), 4),
            conf_a=round(float(c_a), 4), conf_b=round(float(c_b), 4),
            v_a=[round(float(x), 4) for x in v_a], v_b=[round(float(x), 4) for x in v_b],
        ).sign()
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def all(self) -> list[DHardEvent]:
        if not self.path.exists(): return []
        return [DHardEvent(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
