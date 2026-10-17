from server.analysis_server import AnalysisSession

s = AnalysisSession()
print("Evaluating empty session...")
s.evaluate()
print("Evaluating empty session probes...")
s.probes()

s.push_audio([0.0]*640, 100)
print("Evaluating session with audio...")
s.evaluate()
print("Evaluating session with audio probes...")
s.probes()
