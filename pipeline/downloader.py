"""
Module for downloading and caching arXiv source files (LaTeX/TeX or PDF).
"""

import os
import time
import urllib.request
import urllib.error
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arxiv_cache")

def clean_arxiv_id(arxiv_id: str) -> str:
    """
    Cleans the arXiv ID, removing 'arXiv:' prefix and any versions.

    Parameters
    ----------
    arxiv_id : str
        The raw arXiv ID string, potentially containing a prefix or version suffix.

    Returns
    -------
    str
        The cleaned arXiv ID suitable for API queries.
    """
    clean_id = arxiv_id.strip()
    if clean_id.lower().startswith("arxiv:"):
        clean_id = clean_id[6:]
    # Split on 'v' to remove version if present, e.g., '2506.06618v1' -> '2506.06618'
    if "v" in clean_id:
        parts = clean_id.split("v")
        if parts[-1].isdigit():
            clean_id = "v".join(parts[:-1])
    return clean_id

def download_arxiv_pdf(clean_id: str, paper_dir: str, delay_seconds: float = 1.0) -> str:
    """
    Downloads the compiled PDF version of the paper from arXiv.

    Parameters
    ----------
    clean_id : str
        The cleaned arXiv ID of the paper.
    paper_dir : str
        The directory where the PDF should be saved.
    delay_seconds : float, optional
        The sleep delay before requesting the server (default is 1.0).

    Returns
    -------
    str
        The path to the downloaded PDF file, or empty string if failed.
    """
    pdf_path = os.path.join(paper_dir, f"{clean_id}.pdf")
    if os.path.exists(pdf_path):
        logger.info(f"Using cached PDF for arXiv:{clean_id} at {pdf_path}")
        return pdf_path

    logger.info(f"Sleeping for {delay_seconds}s before downloading PDF for arXiv:{clean_id}...")
    time.sleep(delay_seconds)

    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    logger.info(f"Downloading PDF from {url} to {pdf_path}...")

    headers = {
        'User-Agent': 'UniversityNLPProjectCrawler/1.0 (arpitha; OTH Amberg-Weiden NLP Course Project)'
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response, open(pdf_path, 'wb') as out_file:
            out_file.write(response.read())
        logger.info(f"Successfully downloaded PDF for arXiv:{clean_id}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error downloading PDF for arXiv:{clean_id}: {str(e)}")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return ""

def download_arxiv_html(clean_id: str, paper_dir: str, delay_seconds: float = 1.0) -> str:
    """
    Downloads the HTML version of the paper from arXiv.

    Parameters
    ----------
    clean_id : str
        The cleaned arXiv ID of the paper.
    paper_dir : str
        The directory where the HTML should be saved.
    delay_seconds : float, optional
        The sleep delay before requesting the server (default is 1.0).

    Returns
    -------
    str
        The path to the downloaded HTML file, or empty string if failed.
    """
    html_path = os.path.join(paper_dir, f"{clean_id}.html")
    if os.path.exists(html_path):
        logger.info(f"Using cached HTML for arXiv:{clean_id} at {html_path}")
        return html_path

    logger.info(f"Sleeping for {delay_seconds}s before downloading HTML for arXiv:{clean_id}...")
    time.sleep(delay_seconds)

    url = f"https://arxiv.org/html/{clean_id}"
    logger.info(f"Downloading HTML from {url} to {html_path}...")

    headers = {
        'User-Agent': 'UniversityNLPProjectCrawler/1.0 (arpitha; OTH Amberg-Weiden NLP Course Project)'
    }

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
            with open(html_path, 'w', encoding='utf-8') as out_file:
                out_file.write(html_content)
        logger.info(f"Successfully downloaded HTML for arXiv:{clean_id}")
        return html_path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info(f"HTML format not available for arXiv:{clean_id} (404 Not Found)")
        else:
            logger.error(f"HTTP Error downloading HTML for arXiv:{clean_id}: {e.code} {e.reason}")
        return ""
    except Exception as e:
        logger.error(f"Error downloading HTML for arXiv:{clean_id}: {str(e)}")
        if os.path.exists(html_path):
            os.remove(html_path)
        return ""

def download_arxiv_source(arxiv_id: str, delay_seconds: float = 3.0) -> str:
    """
    Downloads the PDF of the paper.
    No longer downloads the LaTeX source to comply with arXiv robots.txt limits.

    Parameters
    ----------
    arxiv_id : str
        The arXiv ID of the paper to download.
    delay_seconds : float, optional
        The sleep delay before requesting the server, to respect robots.txt (default is 3.0).

    Returns
    -------
    str
        The path to the directory containing the downloaded PDF.
    """
    clean_id = clean_arxiv_id(arxiv_id)
    paper_dir = os.path.join(CACHE_DIR, clean_id)
    
    os.makedirs(paper_dir, exist_ok=True)
    
    files = os.listdir(paper_dir)
    pdf_file = f"{clean_id}.pdf"
    html_file = f"{clean_id}.html"
    

    # Try to download HTML first, as it is much better for equation extraction and context parsing.
    # We check if it exists in the directory first to avoid redundant downloads.
    if html_file not in files:
        download_arxiv_html(clean_id, paper_dir, delay_seconds=delay_seconds)
    else:
        logger.info(f"Using cached HTML for arXiv:{clean_id} at {paper_dir}")

    # Always try to download PDF as fallback or for reference.
    # The PDF can be used for backup extraction (via Nougat/PyMuPDF) if HTML is unavailable.
    if pdf_file not in files and "paper.pdf" not in files:
        download_arxiv_pdf(clean_id, paper_dir, delay_seconds=delay_seconds)
    else:
        logger.info(f"Using cached PDF for arXiv:{clean_id} at {paper_dir}")
        
    return paper_dir
