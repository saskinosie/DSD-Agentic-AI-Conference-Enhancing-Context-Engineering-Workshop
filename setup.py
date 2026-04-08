"""
Pre-download script for the workshop.

Run this at home BEFORE the workshop to download all required data.
Conference Wi-Fi will not reliably support ~1 GB of downloads.

Usage:
    python setup.py
"""

import subprocess
import sys
import time


CRDB_IMAGE = "cockroachdb/cockroach:v25.4.6"
DATABASE_URL = "postgresql://root@localhost:26257/defaultdb?sslmode=disable"
WORKSHOP_URL = "postgresql://root@localhost:26257/workshop?sslmode=disable"


def check_docker():
    print("=" * 60)
    print("Step 1: Pulling CockroachDB Docker image (~400 MB)")
    print("=" * 60)
    try:
        subprocess.run(
            ["docker", "pull", CRDB_IMAGE],
            check=True,
        )
        print("Docker image pulled successfully.\n")
    except FileNotFoundError:
        print("ERROR: Docker not found. Please install Docker Desktop first.")
        print("  https://docs.docker.com/get-started/get-docker/\n")
    except subprocess.CalledProcessError:
        print("ERROR: Docker pull failed. Make sure Docker Desktop is running.\n")


def install_dependencies():
    print("=" * 60)
    print("Step 2: Installing Python dependencies (~400 MB)")
    print("=" * 60)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("Python dependencies installed successfully.\n")
    except subprocess.CalledProcessError:
        print("ERROR: pip install failed. Check the error above.\n")


def download_dataset():
    print("=" * 60)
    print("Step 3: Downloading HuggingFace dataset (~250 MB)")
    print("=" * 60)
    try:
        from datasets import load_dataset

        dataset = load_dataset("Qdrant/hm_ecommerce_products", split="train")
        print(f"Dataset downloaded: {len(dataset)} products")
        print("Dataset cached locally for offline use.\n")
    except Exception as e:
        print(f"ERROR: Dataset download failed: {e}\n")


def init_database():
    print("=" * 60)
    print("Step 4: Initializing CockroachDB database")
    print("=" * 60)

    # Start CockroachDB via docker compose
    try:
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            check=True,
        )
        print("CockroachDB container started.")
    except FileNotFoundError:
        print("ERROR: Docker not found. Please install Docker Desktop first.")
        return
    except subprocess.CalledProcessError:
        print("ERROR: docker compose up failed. Make sure Docker Desktop is running.")
        return

    # Wait for CockroachDB to be ready
    print("Waiting for CockroachDB to accept connections...", end="", flush=True)
    import psycopg2

    for attempt in range(30):
        try:
            conn = psycopg2.connect(DATABASE_URL)
            conn.autocommit = True
            conn.close()
            print(" ready!")
            break
        except Exception:
            print(".", end="", flush=True)
            time.sleep(1)
    else:
        print("\nERROR: CockroachDB did not become ready in time.\n")
        return

    # Create workshop database and enable vector indexes
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata "
                "WHERE catalog_name = 'workshop'"
            )
            if not cur.fetchone():
                cur.execute("CREATE DATABASE workshop")
                print("Created 'workshop' database.")
            else:
                print("Database 'workshop' already exists.")

            cur.execute(
                "SET CLUSTER SETTING feature.vector_index.enabled = true"
            )
            print("Enabled vector index support.")
        conn.close()
        print("Database initialization complete.\n")
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}\n")


def verify():
    print("=" * 60)
    print("Verification")
    print("=" * 60)

    errors = []

    # Check Docker image
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", CRDB_IMAGE],
            capture_output=True,
        )
        if result.returncode == 0:
            print("  Docker image:    OK")
        else:
            errors.append("Docker image not found")
            print("  Docker image:    MISSING")
    except FileNotFoundError:
        errors.append("Docker not installed")
        print("  Docker:          NOT INSTALLED")

    # Check CockroachDB is running and workshop database exists
    try:
        import psycopg2

        conn = psycopg2.connect(WORKSHOP_URL)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        print("  CockroachDB:     OK (workshop database accessible)")
    except Exception:
        errors.append("CockroachDB not running or workshop database missing")
        print("  CockroachDB:     NOT READY (run: docker compose up -d)")

    # Check Python packages
    try:
        import psycopg2
        print(f"  psycopg2:        OK (v{psycopg2.__version__})")
    except ImportError:
        errors.append("psycopg2 not installed")
        print("  psycopg2:        MISSING")

    try:
        import pydantic_ai
        print("  pydantic-ai:     OK")
    except ImportError:
        errors.append("pydantic-ai not installed")
        print("  pydantic-ai:     MISSING")

    try:
        import openai
        print("  openai:          OK")
    except ImportError:
        errors.append("openai not installed")
        print("  openai:          MISSING")

    try:
        import datasets
        print("  datasets:        OK")
    except ImportError:
        errors.append("datasets not installed")
        print("  datasets:        MISSING")

    # Check dataset cache
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "Qdrant/hm_ecommerce_products",
            split="train",
        )
        print(f"  HF dataset:      OK ({len(dataset)} products cached)")
    except Exception:
        errors.append("HuggingFace dataset not cached")
        print("  HF dataset:      NOT CACHED")

    # Check .env
    import os

    if os.path.exists(".env"):
        from dotenv import load_dotenv

        load_dotenv()
        if os.getenv("OPENAI_API_KEY"):
            print("  .env file:       OK (OpenAI key set)")
        else:
            errors.append("OPENAI_API_KEY not set in .env")
            print("  .env file:       MISSING OpenAI key")
    else:
        errors.append(".env file not found")
        print("  .env file:       NOT FOUND (run: cp .env.example .env)")

    print()
    if errors:
        print(f"Issues found ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print("\nPlease fix these before the workshop.")
    else:
        print("All good! You are ready for the workshop.")


if __name__ == "__main__":
    print()
    print("Data Science Dojo Workshop Setup")
    print("Enhancing Context Engineering with Agentic Integration")
    print("into Vector Database Queries")
    print()
    print("This script will download ~1 GB of data.")
    print("Run this at home before the workshop.\n")

    check_docker()
    install_dependencies()
    download_dataset()
    init_database()
    verify()
