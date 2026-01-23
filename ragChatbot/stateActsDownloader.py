import bs4 as _bs4
import requests
import os
import datetime
import argparse

# alias back to original name
BeautifulSoup = _bs4.BeautifulSoup
datetime = datetime.datetime


# -------------------- GLOBAL VARIABLES --------------------

CENTRAL_ACTS_LIST_URL= "https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&rpp=845"
INDIA_CODE_BASE_URL= "https://www.indiacode.nic.in"
FAILED_LOG_FILE= "failed_pdf.txt"

# -------------------- LOG FAILURE --------------------

def log_failure(log_file, idx, act_page_url, pdf_url, filename, error):
    with open(log_file, "a") as log:
        log.write(
            f"[{datetime.now().isoformat()}]\n"
            f"Index      : {idx}\n"
            f"Act Page   : {act_page_url}\n"
            f"PDF URL    : {pdf_url}\n"
            f"Filename   : {filename}\n"
            f"Error      : {error}\n"
            f"{'-'*60}\n"
        )



# -------------------- DOWNLOAD CENTRAL ACT PDFS --------------------

def download_central_acts_pdfs(pdf_dir: str, log_dir: str):
    # This is the central acts list page link.
    first_link = CENTRAL_ACTS_LIST_URL
    # Headers for our crawlers to work smoothly.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.indiacode.nic.in/"
    }

    base_url = INDIA_CODE_BASE_URL
    # Making the directory to save pdfs.
    os.makedirs(pdf_dir, exist_ok=True)

    # Making the log file.
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, FAILED_LOG_FILE)
    open(log_file, "w").close()

    session = requests.Session()
    soup = BeautifulSoup(
        session.get(first_link, headers=headers, timeout=30).text,
        "html.parser"
    )

    # Fetching all the Central Acts Pages link.
    links = soup.select("table tr td a")

    # Opened the log file, to keep saving
    for idx, a in enumerate(links, start=1):
        act_page_url = None
        pdf_url = None
        filename = None

        try:
            href = a.get("href")
            if not href:
                raise Exception("Empty href on act listing")

            act_page_url = base_url + href

            page_html = session.get(
                act_page_url, headers=headers, timeout=30
            ).text
            soup2 = BeautifulSoup(page_html, "html.parser")

            pdf_anchor = soup2.select_one("a[href$='.pdf']")
            if not pdf_anchor:
                raise Exception("PDF link not found on act page")

            pdf_url = base_url + pdf_anchor.get("href")
            filename = pdf_url.split("/")[-1]
            path = os.path.join(pdf_dir, filename)

            r = session.get(pdf_url, stream=True, timeout=30)
            if r.status_code != 200:
                raise Exception(f"PDF download failed (HTTP {r.status_code})")

            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            if os.path.getsize(path) < 5000:
                raise Exception("Downloaded file too small (likely blocked)")

        except Exception as e:
            log_failure(
                log_file=log_file,
                idx=idx,
                act_page_url=act_page_url,
                pdf_url=pdf_url,
                filename=filename,
                error=str(e)
            )


# -------------------- MAIN --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Central Acts PDFs from IndiaCode"
    )

    parser.add_argument(
        "--outputDir",
        required=True,
        type=str,
        help="Directory where PDFs will be stored"
    )

    args = parser.parse_args()

    pdf_dir = os.path.abspath(args.outputDir)
    log_dir = os.path.dirname(pdf_dir)

    download_central_acts_pdfs(
        pdf_dir=pdf_dir,
        log_dir=log_dir
    )




if __name__ == "__main__":
    main()