from bs4 import BeautifulSoup
import requests
import os

FirstLink = "https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&rpp=845"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.indiacode.nic.in/"
}

session = requests.Session()
html = session.get(FirstLink, headers=headers).text

soup = BeautifulSoup(html)

# print(soup.prettify())

firstAs= soup.select("table tr td a")

baseUrl= "https://www.indiacode.nic.in"
        
for a in firstAs:
    try:
        href = a.get("href")
        if not href:
            continue 

        fullUrl= baseUrl+href
        # print(fullUrl)   
        session = requests.Session()
        html = session.get(fullUrl, headers=headers).text

        soup2 = BeautifulSoup(html)
        # print(soup2.prettify())
        aa=soup2.select_one("a[href$='.pdf']")
        hreff = aa.get("href")

        pdf_url = baseUrl+hreff

        os.makedirs("pdfs", exist_ok=True)

        filename = pdf_url.split("/")[-1]
        path = f"pdfs/{filename}"

        r = session.get(pdf_url, stream=True, timeout=30)

        if r.status_code != 200:
            raise Exception(f"PDF HTTP {r.status_code}")

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(path) < 5000:
            raise Exception("Downloaded file is too small (likely blocked)")
    except Exception as e:
        continue




    