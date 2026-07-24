import os

os.environ.setdefault("DIRECTOR_ENVIRONMENT", "test")
os.environ.setdefault("DIRECTOR_AUTH_REQUIRED", "false")
os.environ.setdefault("DIRECTOR_AUTH_SECRET", "test-auth-secret-change-me")
os.environ.setdefault("DIRECTOR_DATABASE_URL", "sqlite+pysqlite:///./test_director.db")
os.environ.setdefault("DIRECTOR_UPLOAD_DIR", "./.test-data/uploads")
os.environ.setdefault("DIRECTOR_OUTPUT_DIR", "./.test-data/outputs")
os.environ.setdefault("DIRECTOR_MAX_UPLOAD_BYTES", "1048576")
os.environ.setdefault("DIRECTOR_RESUMABLE_REQUEST_BYTES", "65536")
