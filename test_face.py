import json
from server.analysis_server import AnalysisSession

session = AnalysisSession()
session.push_face(1234, [0.5]*100)
print("Face OK")
