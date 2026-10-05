import os
import requests
import pandas as pd

from pathlib import Path

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def fetch_company_submissions(
        cik: str,
) -> dict:
    """
    Fetch SEC submissions metadata for one company.

    Parameters:
    cik:
        SEC CIK number.
    
    Returns:
    dict
        Raw SEC submissions JSON.
    """

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise ValueError(
            "SEC USER AGENT is not set."
        )

    padded_cik = str(cik).zfill(10)

    url = SEC_SUBMISSIONS_URL.format(
        cik=padded_cik
    )

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": (
            "gzip, deflate"
        ),
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def fetch_submission_file(
        file_name: str,
) -> dict:
    """
    Fetch one SEC historical submissions file.

    Example:
    CIK0001318605-submissions-001.json
    """

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT is not set."
        )

    url = (
        "https://data.sec.gov/submissions/" + file_name
    )

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": (
            "gzip, deflate"
        ),
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


def extract_historical_filings(
        submission_file: dict,
) -> pd.DataFrame:
    """
    Convert one historical SEC submissions JSON file into a DataFrame.
    """

    return pd.DataFrame(
        submission_file
    )


def extract_recent_filings(
        submissions: dict,
) -> pd.DataFrame:
    """
    Convert SEC recent filing metadata into a DataFrame.
    """

    recent = submissions["filings"]["recent"]

    df = pd.DataFrame(recent)

    return df


def filter_financial_filings(
        filings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep 10-Q, 10-K and amendments.
    """

    allowed_forms = {
        "10-Q",
        "10-K",
        "10-Q/A",
        "10-K/A",
    }

    filtered = (
        filings[
            filings[
                "form"
            ].isin(
                allowed_forms
            )
        ]
        .copy()
    )

    return filtered


def combine_filings(
        recent_filings: pd.DataFrame,
        historical_filings: list[pd.DataFrame],
) -> pd.DataFrame:
    """
    Combine recent and historical SEC filings.
    Duplicate accession numbers are removed.
    """

    frames = [
        recent_filings,
        *historical_filings,
    ]

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "accessionNumber"
        ],
        keep="first",
    ).sort_values(
        "filingDate",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return combined


def prepare_filing_metadata(
        filings: pd.DataFrame,
        cik: str,
) -> pd.DataFrame:
    """
    Prepare clean SEC filing metadata and construct the
    primary filing document URL.
    """

    required_columns = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "primaryDocument",
    ]

    missing_columns = (
        set(required_columns) - set(filings.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required filings columns: "
            f"{sorted(missing_columns)}"
        )

    result = filings[required_columns].copy()

    # Normalized dates
    result["filingDate"] = pd.to_datetime(
        result["filingDate"]
    )

    result["reportDate"] = pd.to_datetime(
        result["reportDate"]
    )

    # SEC archive URL components
    cik_no_leading_zero = str(int(cik))

    result["accession_no_dashes"] = (
        result["accessionNumber"].str.replace(
            "-",
            "",
            regex=False,
        )
    )

    result["filing_url"] = (
        "https://www.sec.gov/Archives/edgar/data/"
        + cik_no_leading_zero
        + "/"
        + result["accession_no_dashes"]
        + "/"
        + result["primaryDocument"]
    )

    # Rename columns
    result = result.rename(
        columns={
            "accessionNumber": "accession_number",
            "filingDate": "filed",
            "reportDate": "report_date",
            "primaryDocument": "primary_document",
        }
    )

    result = result[
        [
            "accession_number",
            "form",
            "report_date",
            "filed",
            "primary_document",
            "filing_url",
        ]
    ].sort_values(
        "filed"
    ).reset_index(
        drop=True
    )

    return result


def download_filing_document(
        filing_url: str,
        output_path: Path,
) -> Path:
    """
    Download one SEC filing HTML document.

    Parameters:
    filing_url:
        Full SEC filing document URL.
    
    output_path:
        local path where the HTML file should be saved.

    Return:
    Path:
        Saved file path.
    """

    user_agent = os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise ValueError(
            "SEC_USER_AGENT is not set."
        )

    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": (
            "gzip, deflate"
        ),
    }

    response = requests.get(
        filing_url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_bytes(
        response.content
    )

    return output_path