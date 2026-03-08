"""Book download step.

This module provides the BookDownloadStep which downloads books from
LibGen and other mirror sources.
"""

import re
import time
import urllib3
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from course_builder.domain.books import DownloadedBook, LibgenResult
from course_builder.pipeline.base import PipelineContext, PipelineStep

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _extract_md5(book: LibgenResult) -> str | None:
    """Extract MD5 hash from book's download links."""
    sources = [book.direct_download_link or ""] + (book.mirror_links or [])
    for url in sources:
        if "md5=" in url:
            return url.split("md5=")[1].split("&")[0]
        if "/md5/" in url:
            return url.split("/md5/")[-1].split("/")[0]
        if "/main/" in url:
            part = url.split("/main/")[-1]
            if "/" in part:
                return part.split("/")[0]
    return None


def _get_libgen_download_url(md5: str, session: requests.Session) -> str | None:
    """Get download URL from libgen.li with session for cookies."""
    try:
        # First get the ads page to get download key
        ads_url = f"https://libgen.li/ads.php?md5={md5}"
        print("    Fetching libgen.li ads page...")
        resp = session.get(ads_url, timeout=30, verify=False)
        if resp.status_code != 200:
            print(f"    libgen.li returned {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "get.php" in href and "md5=" in href:
                if not href.startswith("http"):
                    href = "https://libgen.li/" + href
                print(f"    Found: {href[:70]}...")
                return href
    except Exception as e:
        print(f"    libgen.li failed: {e}")
    return None


def _get_libgen_lc_url(md5: str, session: requests.Session) -> str | None:
    """Try libgen.lc mirror."""
    try:
        url = f"https://libgen.lc/ads.php?md5={md5}"
        print("    Trying libgen.lc...")
        resp = session.get(url, timeout=30, verify=False)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "get.php" in href or "cloudflare" in href.lower():
                if not href.startswith("http"):
                    href = "https://libgen.lc/" + href
                print(f"    Found: {href[:70]}...")
                return href
    except Exception as e:
        print(f"    libgen.lc failed: {e}")
    return None


def _get_annas_archive_urls(md5: str, session: requests.Session) -> list[str]:
    """Get download URLs from Anna's Archive - the most reliable source."""
    urls = []
    try:
        # Anna's Archive page for this MD5
        page_url = f"https://annas-archive.org/md5/{md5}"
        print("    Fetching Anna's Archive page...")
        resp = session.get(page_url, timeout=30, verify=False)
        if resp.status_code != 200:
            print(f"    Anna's Archive returned {resp.status_code}")
            return urls

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find all download links - they're usually in buttons or specific link patterns
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Look for direct download links
            if any(
                pattern in href
                for pattern in [
                    "/slow_download/",
                    "/fast_download/",
                    "library.lol",
                    "libgen.li/get",
                    "libgen.lc/get",
                    "/dl/",
                ]
            ):
                if not href.startswith("http"):
                    href = "https://annas-archive.org" + href
                if href not in urls:
                    urls.append(href)
                    print(f"    Found: {href[:70]}...")

        # Also try the slow download endpoint directly
        slow_url = f"https://annas-archive.org/slow_download/{md5}/0/2"
        urls.insert(0, slow_url)

    except Exception as e:
        print(f"    Anna's Archive failed: {e}")
    return urls


def _get_library_lol_url(md5: str, session: requests.Session) -> str | None:
    """Try library.lol mirror - commonly used LibGen mirror."""
    try:
        url = f"https://library.lol/main/{md5}"
        print("    Trying library.lol...")
        resp = session.get(url, timeout=30, verify=False)
        if resp.status_code != 200:
            print(f"    library.lol returned {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        # Look for the GET button/link
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text().strip().lower()
            if "get" in text or "download" in text or "cloudflare" in href.lower():
                if not href.startswith("http"):
                    href = "https://library.lol" + href
                print(f"    Found: {href[:70]}...")
                return href
            # Also check for direct file links
            if href.endswith(".pdf") or "/get.php" in href:
                if not href.startswith("http"):
                    href = "https://library.lol" + href
                print(f"    Found: {href[:70]}...")
                return href
    except Exception as e:
        print(f"    library.lol failed: {e}")
    return None


def _sanitize_filename(name: str, max_length: int = 100) -> str:
    """Remove characters that are unsafe for filenames and truncate."""
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name).strip(". ")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip(". ")
    return sanitized


def _download_file(
    url: str, dest: Path, session: requests.Session, timeout: int = 180, max_retries: int = 3
) -> None:
    """Download a file from url and save to dest with retry logic."""
    print(f"    Downloading from: {url[:80]}...")

    # Handle Anna's Archive slow_download - may redirect to actual file
    if "slow_download" in url:
        # First request may return HTML with a redirect or direct link
        resp = session.get(url, timeout=60, verify=False, allow_redirects=True)
        if resp.status_code == 200 and "text/html" in resp.headers.get("content-type", ""):
            # Parse HTML to find actual download link
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if any(ext in href.lower() for ext in [".pdf", "/get", "/download"]):
                    if not href.startswith("http"):
                        href = "https://annas-archive.org" + href
                    print(f"    Following redirect to: {href[:70]}...")
                    url = href
                    break

    last_error = None
    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = 5 * attempt
            print(f"    Retry {attempt}/{max_retries} after {wait_time}s...")
            time.sleep(wait_time)

        try:
            resp = session.get(url, timeout=timeout, stream=True, verify=False, allow_redirects=True)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                # Check if it's a small HTML error page
                content = resp.content
                if len(content) < 50000 and b"<html" in content.lower():
                    raise ValueError("Server returned HTML instead of file")

            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0

            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_done = downloaded / 1024 / 1024
                        mb_total = total_size / 1024 / 1024
                        print(
                            f"\r    Progress: {mb_done:.1f} MB / {mb_total:.1f} MB ({pct:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        mb_done = downloaded / 1024 / 1024
                        print(f"\r    Downloaded: {mb_done:.1f} MB", end="", flush=True)
            print()

            # Validate - check file size and magic bytes
            if dest.stat().st_size < 10000:
                dest.unlink()
                raise ValueError(f"File too small ({dest.stat().st_size} bytes)")

            with open(dest, "rb") as f:
                header = f.read(5)
            if header != b"%PDF-":
                dest.unlink()
                raise ValueError("Downloaded file is not a valid PDF")

            # Success!
            return

        except Exception as e:
            last_error = e
            if dest.exists():
                dest.unlink()
            print(f"    Attempt {attempt + 1} failed: {str(e)[:60]}")
            continue

    # All retries exhausted
    raise last_error or ValueError("Download failed after retries")


def _wait_for_manual_download(dest: Path, book_title: str, md5: str | None) -> bool:
    """Prompt user to manually download and wait for file."""
    import sys

    print("\n" + "=" * 60)
    print("AUTOMATIC DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print(f"\nBook: {book_title}")
    if md5:
        print(f"MD5: {md5}")
        print("\nTry these URLs in your browser:")
        print(f"  1. https://libgen.li/ads.php?md5={md5}")
        print(f"  2. https://libgen.lc/ads.php?md5={md5}")
        print(f"  3. https://annas-archive.org/md5/{md5}")
    print(f"\nSave the PDF to:")
    print(f"  {dest}")
    print("\n" + "-" * 60)

    # Check if we're in interactive mode
    if not sys.stdin.isatty():
        print("Non-interactive mode detected - cannot prompt for manual download.")
        print("Please run interactively or download the file manually and re-run.")
        return False

    while True:
        try:
            response = input("Press ENTER when file is ready, or 's' to skip this book: ").strip().lower()
        except EOFError:
            print("No interactive input available - skipping.")
            return False

        if response == "s":
            print("Skipping this book...")
            return False
        if dest.exists() and dest.stat().st_size > 10000:
            # Validate PDF
            with open(dest, "rb") as f:
                if f.read(5) == b"%PDF-":
                    print("File validated successfully!")
                    return True
            print("File exists but is not a valid PDF. Please try again.")
        else:
            print(f"File not found or too small. Expected at: {dest}")


class BookDownloadStep(PipelineStep):
    """Pipeline step that downloads books from LibGen mirrors."""

    def __init__(self, output_dir: str = "downloads", allow_manual: bool = True):
        """Initialize the download step.

        Args:
            output_dir: Directory to save downloaded books.
            allow_manual: Whether to prompt for manual download on failure.
        """
        self.output_dir = output_dir
        self.allow_manual = allow_manual

    def run(self, context: PipelineContext) -> PipelineContext:
        """Download all found books.

        Expects context["books_found"] and context["certification_name"].
        Populates context["books_downloaded"] and context["books_failed"].
        """
        books_found: list[LibgenResult] = context["books_found"]
        certification_name = context["certification_name"]

        safe_cert_name = _sanitize_filename(certification_name)
        dest_dir = Path(self.output_dir) / safe_cert_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        downloaded: list[DownloadedBook] = []
        failed: list[DownloadedBook] = []

        session = requests.Session()
        session.headers.update(HEADERS)

        for i, book in enumerate(books_found, 1):
            ext = book.extension or "pdf"
            filename = f"{_sanitize_filename(book.title)}.{ext}"
            file_path = dest_dir / filename

            print(f"\n[{i}/{len(books_found)}] {book.title[:70]}...")

            # Skip if file already exists and is valid
            if file_path.exists() and file_path.stat().st_size > 10000:
                with open(file_path, "rb") as f:
                    if f.read(5) == b"%PDF-":
                        print("  -> Already exists and valid, skipping")
                        downloaded.append(
                            DownloadedBook(
                                title=book.title,
                                author=book.author,
                                extension=ext,
                                file_path=str(file_path),
                                success=True,
                                error=None,
                            )
                        )
                        continue

            # Extract MD5
            md5 = _extract_md5(book)
            if md5:
                print(f"    MD5: {md5}")

            # Build list of URLs to try - prioritize most reliable sources
            urls_to_try = []

            if md5:
                # 1. Anna's Archive - most reliable aggregator
                annas_urls = _get_annas_archive_urls(md5, session)
                urls_to_try.extend(annas_urls)

                # 2. library.lol - popular direct mirror
                library_lol_url = _get_library_lol_url(md5, session)
                if library_lol_url:
                    urls_to_try.append(library_lol_url)

                # 3. libgen.li
                libgen_url = _get_libgen_download_url(md5, session)
                if libgen_url:
                    urls_to_try.append(libgen_url)

                # 4. libgen.lc
                libgen_lc_url = _get_libgen_lc_url(md5, session)
                if libgen_lc_url:
                    urls_to_try.append(libgen_lc_url)

            # 5. Original URLs from search as fallback
            if book.direct_download_link:
                urls_to_try.append(book.direct_download_link)
            urls_to_try.extend(book.mirror_links or [])

            # Try each URL with delays between attempts
            last_error = None
            success = False

            for idx, url in enumerate(urls_to_try):
                if idx > 0:
                    # Add delay between attempts to avoid rate limiting
                    print("    Waiting 2s before next attempt...")
                    time.sleep(2)
                try:
                    _download_file(url, file_path, session)
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    print(f"    Failed: {last_error[:60]}")
                    if file_path.exists():
                        file_path.unlink()

            # If automatic download failed, try manual
            if not success and self.allow_manual:
                success = _wait_for_manual_download(file_path, book.title, md5)

            result = DownloadedBook(
                title=book.title,
                author=book.author,
                extension=ext,
                file_path=str(file_path),
                success=success,
                error=None if success else (last_error or "Download failed"),
            )

            if success:
                downloaded.append(result)
                print(f"  -> Saved to {file_path}")
            else:
                failed.append(result)
                print(f"  -> Failed: {result.error}")

        print(f"\n{'=' * 60}")
        print(f"Downloads: {len(downloaded)} succeeded, {len(failed)} failed.")
        print(f"{'=' * 60}\n")

        context["books_downloaded"] = downloaded
        context["books_failed"] = failed
        return context
