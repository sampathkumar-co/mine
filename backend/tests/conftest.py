import os

os.environ.setdefault("DIRECTOR_ENVIRONMENT", "test")
os.environ.setdefault("DIRECTOR_AUTH_REQUIRED", "false")
os.environ.setdefault("DIRECTOR_AUTH_SECRET", "test-auth-secret-change-me")
