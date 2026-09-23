NORMALIZED_METRICS = {
    "revenue": {
        "display_name": "Revenue",
        "primary_concept": "Revenues",
        "unit": "USD",
        "fact_type": "duration",
        "nonnegative_expected": True,
    },

    "gross_profit": {
        "display_name": "Gross Profit",
        "primary_concept": "GrossProfit",
        "unit": "USD",
        "fact_type": "duration",
        "nonnegative_expected": False,
    },

    "operating_income": {
        "display_name": "Operating Income",
        "primary_concept": "OperatingIncomeLoss",
        "unit": "USD",
        "fact_type": "duration",
        "nonnegative_expected": False,
    },

    "net_income": {
        "display_name": "Net Income",
        "primary_concept": "NetIncomeLoss",
        "unit": "USD",
        "fact_type": "duration",
        "nonnegative_expected": False,
    },

    "cash": {
        "display_name": "Cash and Cash Equivalents",
        "primary_concept": "CashAndCashEquivalentsAtCarryingValue",
        "unit": "USD",
        "fact_type": "instant",
        "nonnegative_expected": True,
    },
}