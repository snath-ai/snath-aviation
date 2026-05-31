import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")))

import os
import json
import datetime
from dmn.health_monitor import AviationHealthMonitor
from dmn.adapter_router import AviationAdapterRouter

# 1. Setup a mocked adapter directory
os.makedirs("models/adapters_live", exist_ok=True)

# Create timestamps: One from today, one from 3 years ago
now = datetime.datetime.now(datetime.timezone.utc)
three_years_ago = (now - datetime.timedelta(days=365 * 3)).isoformat()

# Weather adapter (3 years old - should decay fast)
with open("models/adapters_live/adapter_weather_icing.json", "w") as f:
    json.dump({
        "type": "weather_icing",
        "centroid_v_a": [0.0, 0.0, 0.0],
        "failure_class": "weather_induced",
        "created_at": three_years_ago
    }, f)

# Hardware adapter (3 years old - should decay very slowly)
with open("models/adapters_live/adapter_hardware_bent.json", "w") as f:
    json.dump({
        "type": "hardware_bent",
        "centroid_v_a": [0.0, 0.0, 0.0],
        "failure_class": "hardware_struct",
        "created_at": three_years_ago
    }, f)
    
# Fresh adapter (Created today)
with open("models/adapters_live/adapter_gps_spoof.json", "w") as f:
    json.dump({
        "type": "gps_spoof",
        "centroid_v_a": [0.0, 0.0, 0.0],
        "failure_class": "default",
        "created_at": now.isoformat()
    }, f)

print("\n[1] Starting AviationHealthMonitor Temporal Test...")

class MockRouter:
    def divergence(self, a, b): return 0.8
    def route(self, a, b, d): return None

class MockEncoder:
    def __init__(self, name):
        self.modality = name
        self.lora_A = None
        self.lora_B = None

router = MockRouter()
adapter_router = AviationAdapterRouter(adapter_dir="models/adapters_live")
monitor = AviationHealthMonitor(router, adapter_router, adapter_dir="models/adapters_live", min_trust_weight=0.40, verbose=True)

# Register a mocked sensor
pitot = MockEncoder("pitot")
radar = MockEncoder("radar")
gps = MockEncoder("gps")

monitor.register(pitot)
monitor.register(radar)
monitor.register(gps)

# Simulate the system attempting to load these three LoRAs
print("\n[2] Simulating System 1 Cache Hits...")
monitor.notify_lora_loaded(pitot, "weather_icing")
monitor.notify_lora_loaded(radar, "hardware_bent")
monitor.notify_lora_loaded(gps, "gps_spoof")

# Run the temporal audit to show the math in action!
print("\n[3] Running Temporal Audit (W = exp(-λ * Δt))...")
monitor.temporal_audit()
