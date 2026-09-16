import json
from pathlib import Path

from src.ingestion.sec import SECClient


TESLA_CIK = "1318605"
OUTPUT_DIR = Path("data/raw")

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = SECClient()

    print("Downloading Tesla SEC Company Facts...")

    data = client.get_company_facts(TESLA_CIK)

    output_path = OUTPUT_DIR / "tsla_company_facts.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data, 
            file,
            indent=2, 
        )

    print(f"Saved to: {output_path}")

    print(f"Company: {data['entityName']}")

if __name__ == "__main__":
    main()