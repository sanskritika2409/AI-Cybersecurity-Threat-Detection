def detect_threats(predictions):
    print("🚨 Detecting threats...")

    results = []

    for i, pred in enumerate(predictions):
        if pred == 1:
            results.append((i, "⚠️ Threat Detected"))
        else:
            results.append((i, "✅ Normal"))

    return results