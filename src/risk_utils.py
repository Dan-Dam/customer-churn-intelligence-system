RISK_COLORS = {

    "✅ Low Risk": "#10B981",

    "⚠️ Medium Risk": "#F59E0B",

    "🚨 High Risk": "#F97316",

    "🔥 Critical Risk": "#DC2626"
}


def classify_risk(probability):

    if probability >= 0.80:

        return {
            "label": "🔥 Critical Risk",
            "color": RISK_COLORS["🔥 Critical Risk"]
        }

    elif probability >= 0.60:

        return {
            "label": "🚨 High Risk",
            "color": RISK_COLORS["🚨 High Risk"]
        }

    elif probability >= 0.40:

        return {
            "label": "⚠️ Medium Risk",
            "color": RISK_COLORS["⚠️ Medium Risk"]
        }

    else:

        return {
            "label": "✅ Low Risk",
            "color": RISK_COLORS["✅ Low Risk"]
        }