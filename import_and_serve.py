#!/usr/bin/env python3
"""
DEPRECATED: This file contains the old BentoML serving approach.

🚨 IMPORTANT: This approach has been replaced by retrain_and_serve.py

For the simplest way to serve your fraud detection model, use:

    python retrain_and_serve.py

This new approach:
- ✅ Retrains a compatible model with current sklearn version
- ✅ Starts a Flask REST API server
- ✅ Bypasses MLflow/BentoML version compatibility issues
- ✅ Provides real-time predictions at http://localhost:3000

Old BentoML code (kept for reference):
"""

# import bentoml
# from bentoml.io import JSON
# import mlflow

# # ... old BentoML implementation removed ...

print("🚨 DEPRECATED: Use 'python retrain_and_serve.py' instead")
print("📖 See README.md for the new serving approach")
