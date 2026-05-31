import torch
import hmac
import hashlib
import os

_ADAPTER_HMAC_KEY = b"snath_aviation_adapter_sovereignty_2026"
dir_path = "models/adapters_live/"
for f in os.listdir(dir_path):
    if f.endswith(".pt"):
        pt_path = os.path.join(dir_path, f)
        state = torch.load(pt_path)
        a_hash = hashlib.sha256(state["A"].numpy().tobytes()).hexdigest()[:16]
        b_hash = hashlib.sha256(state["B"].numpy().tobytes()).hexdigest()[:16]
        enc = state.get("target_encoder", "pitot")
        payload_str = f"{enc}|{a_hash}|{b_hash}"
        signature = hmac.new(_ADAPTER_HMAC_KEY, payload_str.encode(), hashlib.sha256).hexdigest()
        state["hmac_hex"] = signature
        torch.save(state, pt_path)
        print(f"Resigned {pt_path}")
