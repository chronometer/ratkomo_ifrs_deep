from datetime import datetime

def validate_timestamp(timestamp: str) -> str:
    """Validate timestamp format"""
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S%z")
        return timestamp
    except ValueError:
        raise ValueError("Invalid timestamp format")
