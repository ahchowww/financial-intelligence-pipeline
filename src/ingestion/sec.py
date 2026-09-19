import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

class SECClient:
    '''Simple client for retrieving public SEC EDGAR data'''

    BASE_URL = 'https://data.sec.gov'

    def __init__(self) -> None:
        user_agent =os.getenv('SEC_USER_AGENT')

        if not user_agent:
            raise ValueError(
                "SEC_USER_AGENT is missing. Add it to your .env file."
            )

        self.headers ={
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def get_company_facts(self, cik: str) -> dict[str, Any]:
        """
        Retrieve all XBRL company facts for one SEC registrant.
        Input: CIK
        Output: SEC company facts dictionary

        Parameters
        -----------------
        cik:
            SEC Central Index Key
        
        Returns
        -----------------
        dict
            Company Facts JSON response
        """

        cik = cik.zfill(10)

        url =(
            f"{self.BASE_URL}/api/xbrl/"
            f"companyfacts/CIK{cik}.json"
        )

        # API request
        response = requests.get(
            url,
            headers=self.headers,
            timeout=30,
        )

        response.raise_for_status()

        return response.json()