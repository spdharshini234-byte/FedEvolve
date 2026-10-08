def format_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024


def summarize_communication(tracker):
    """Turns a CommunicationTracker summary into the structure shown on the Privacy page."""
    s = tracker.summary() if hasattr(tracker, "summary") else tracker
    s["total_readable"] = format_bytes(s["total_bytes"])
    return s
