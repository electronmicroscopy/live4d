"""Offline tooling: read/write `.dstream` capture files, replay them, and synthesize test data.

These are for demonstrating and testing the pipeline without a detector — none of it is on the
live path (the live path receives frames straight from the DCU via the Holoscan ingest).
"""
