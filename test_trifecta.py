from demo_real_world import *
flight_data = fetch_live_flight()
radar = RadarEncoder()
pitot = PitotEncoder()

z_radar = radar.encode(flight_data)
# Artificially inject a divergence > 0.5
div = 0.99
decision = RouteDecision.TRIGGER_REPLAN
print(f"Divergence: {div:.2f}, Decision: {decision.name}")

if decision == RouteDecision.TRIGGER_REPLAN:
    kernel = AviationRoutingKernel()
    score = 0.9 # High safety score, normally triggers COMMIT
    
    final_route = kernel.route(score, divergence=div, jury_decision=None)
    print(f"Final Maneuver: {final_route}")
    
    if "STRUCTURAL_IMPASSE" in final_route:
        reporter = IncidentReporterNode(severity_threshold="LOW", incident_log_path="lar_logs/incidents.jsonl")
        incident_state = GraphState({
            "last_error": "RUNTIME_ERROR",
            "run_id": flight_data["callsign"]
        })
        print("🚨 STRUCTURAL_IMPASSE detected. Filing Art 73 Incident Report...")
        reporter.execute(incident_state)
