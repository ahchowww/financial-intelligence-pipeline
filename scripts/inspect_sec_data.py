import json
from pathlib import Path

DATA_PATH = Path("data/raw/tsla_company_facts.json")

def main() -> None:
    with DATA_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print("Company:", data['entityName'])
    print("CIK:", data["cik"])

    print("\nAvailable taxonomies:")
    print(data["facts"].keys())

    us_gaap = data["facts"]["us-gaap"]

    print("\nNumber of US-GAAP concepts:")
    print(len(us_gaap))
    
    print("\nFirst 20 concepts:")
    
    for concept in list(us_gaap.keys())[:20]:
        print(concept)

    """Revenue-related concept"""
    print("\nRevenue-related concepts:")

    for concept, details in us_gaap.items():
        if "revenue" in concept.lower():
            print(
                concept,
                "->",
                details.get("label")
            )

    """Inspect one concept"""
    concept_name = "RevenueFromContractWithCustomerExcludingAssessedTax"

    concept = us_gaap[concept_name]

    print("\nConcept:")
    print(concept_name)

    print("\nLabel:")
    print(concept["label"])

    print("\nDescription:")
    print(concept["description"])

    print("\nUnits:")
    print(concept["units"].keys())


    """Look at actual observations"""
    facts = concept["units"]["USD"]

    print("\nNumber of observations:")
    print(len(facts))

    print("\nFirst observation:")
    print(facts[0])



if __name__ == "__main__":
    main()