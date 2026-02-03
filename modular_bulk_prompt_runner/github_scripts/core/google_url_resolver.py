# core/google_url_resolver.py
"""
Google URL Resolver - Post-processing utility for resolving Vertex redirect URLs.

This module provides a standalone resolver that processes batch output CSVs
to convert Google's Vertex redirect URLs to actual destination page URLs.

Usage:
    resolver = GoogleURLResolver()
    resolver.resolve_batch_output(detail_csv_path)

The resolver:
1. Reads the batch output CSV
2. Extracts unique Vertex redirect URLs
3. Resolves each URL with retry logic and delays
4. Creates a backup of the original CSV
5. Saves a resolved version with actual page URLs
"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd


class GoogleURLResolver:
    """Post-processing resolver for Google Vertex redirect URLs.

    Processes batch output CSVs to resolve redirect URLs to actual
    destination page URLs, with progress output at each stage.
    """

    VERTEX_PREFIX = "https://vertexaisearch.cloud.google.com/"

    def __init__(
        self,
        max_attempts: int = 4,
        base_delay_range: Tuple[float, float] = (3.0, 5.0),
        increment_range: Tuple[float, float] = (3.0, 5.0),
        request_timeout: float = 5.0,
    ):
        """Initialize the resolver.

        Args:
            max_attempts: Maximum retry attempts per URL (default: 4)
            base_delay_range: (min, max) seconds for base delay
            increment_range: (min, max) seconds to add per retry
            request_timeout: Timeout for each HEAD request in seconds
        """
        self.max_attempts = max_attempts
        self.base_delay_range = base_delay_range
        self.increment_range = increment_range
        self.request_timeout = request_timeout

        # Statistics
        self.stats = {
            "total_urls": 0,
            "resolved": 0,
            "failed": 0,
            "skipped": 0,
        }

    def resolve_batch_output(
        self,
        csv_path: str | Path,
        url_column: str = "citation_url",
        output_suffix: str = "_resolved",
        backup_suffix: str = "_original",
    ) -> Optional[Path]:
        """Resolve all Vertex URLs in a batch output CSV.

        Args:
            csv_path: Path to the detail CSV file
            url_column: Name of the column containing URLs
            output_suffix: Suffix for the resolved output file
            backup_suffix: Suffix for the backup file

        Returns:
            Path to the resolved CSV, or None if failed
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            print(f"Error: File not found: {csv_path}")
            return None

        print(f"\n{'='*60}")
        print(f"Google URL Resolver")
        print(f"{'='*60}")
        print(f"\nInput file: {csv_path.name}")

        # Step 1: Read the CSV
        print(f"\n[1/5] Reading batch output...")
        try:
            df = pd.read_csv(csv_path)
            print(f"      Loaded {len(df)} rows")
        except Exception as e:
            print(f"      Error reading CSV: {e}")
            return None

        if url_column not in df.columns:
            print(f"      Error: Column '{url_column}' not found")
            print(f"      Available columns: {list(df.columns)}")
            return None

        # Step 2: Extract unique Vertex URLs
        print(f"\n[2/5] Extracting unique redirect URLs...")
        vertex_urls = self._extract_vertex_urls(df, url_column)
        print(f"      Found {len(vertex_urls)} unique Vertex redirect URLs")

        if not vertex_urls:
            print(f"      No Vertex URLs to resolve. File may already be resolved.")
            return csv_path

        # Step 3: Resolve URLs
        print(f"\n[3/5] Resolving URLs (this may take a while)...")
        print(f"      Using {self.base_delay_range[0]}-{self.base_delay_range[1]}s delays between requests")
        print()

        url_mapping = self._resolve_urls(vertex_urls)

        print(f"\n      Resolution complete:")
        print(f"        Resolved: {self.stats['resolved']}")
        print(f"        Failed:   {self.stats['failed']}")

        # Step 4: Create backup
        print(f"\n[4/5] Creating backup of original...")
        backup_path = csv_path.with_stem(csv_path.stem + backup_suffix)
        try:
            shutil.copy2(csv_path, backup_path)
            print(f"      Backup saved: {backup_path.name}")
        except Exception as e:
            print(f"      Warning: Could not create backup: {e}")

        # Step 5: Apply mapping and save
        print(f"\n[5/5] Applying URL replacements...")
        resolved_df = self._apply_mapping(df, url_column, url_mapping)

        output_path = csv_path.with_stem(csv_path.stem + output_suffix)
        try:
            resolved_df.to_csv(output_path, index=False)
            print(f"      Resolved output saved: {output_path.name}")
        except Exception as e:
            print(f"      Error saving resolved CSV: {e}")
            return None

        # Print summary
        self._print_summary(url_mapping, output_path)

        return output_path

    def _extract_vertex_urls(self, df: pd.DataFrame, url_column: str) -> List[str]:
        """Extract unique Vertex redirect URLs from the DataFrame."""
        urls = df[url_column].dropna().unique()
        vertex_urls = [
            url for url in urls
            if isinstance(url, str) and url.startswith(self.VERTEX_PREFIX)
        ]
        return vertex_urls

    def _resolve_urls(self, urls: List[str]) -> Dict[str, Dict]:
        """Resolve a list of URLs and return mapping.

        Returns:
            Dict mapping redirect_url -> {resolved_url, domain, status}
        """
        try:
            import requests
        except ImportError:
            print("      Error: 'requests' library not installed")
            print("      Run: pip install requests")
            return {}

        mapping = {}
        total = len(urls)
        self.stats["total_urls"] = total

        for i, url in enumerate(urls, 1):
            # Progress indicator
            progress = f"[{i}/{total}]"

            resolved_url, status, domain = self._resolve_single_url(url, requests)

            if resolved_url:
                mapping[url] = {
                    "resolved_url": resolved_url,
                    "domain": domain,
                    "status": status,
                }
                self.stats["resolved"] += 1
                print(f"      {progress} OK: {domain}")
            else:
                # Extract domain from original URL title if available
                fallback_domain = self._extract_domain_from_vertex_url(url)
                mapping[url] = {
                    "resolved_url": None,
                    "domain": fallback_domain,
                    "status": status,
                }
                self.stats["failed"] += 1
                print(f"      {progress} FAILED ({status}): {fallback_domain or 'unknown'}")

            # Delay before next URL (not after the last one)
            if i < total:
                delay = random.uniform(*self.base_delay_range)
                time.sleep(delay)

        return mapping

    def _resolve_single_url(self, redirect_url: str, requests) -> Tuple[Optional[str], str, Optional[str]]:
        """Resolve a single redirect URL with progressive retry backoff.

        Returns:
            Tuple of (resolved_url or None, status, domain or None)
        """
        base_delay = random.uniform(*self.base_delay_range)
        increment = random.uniform(*self.increment_range)

        last_error = "unknown"

        for attempt in range(self.max_attempts):
            try:
                response = requests.head(
                    redirect_url,
                    allow_redirects=True,
                    timeout=self.request_timeout,
                )

                final_url = response.url
                is_resolved = not final_url.startswith(self.VERTEX_PREFIX)

                if is_resolved:
                    domain = self._extract_domain(final_url)
                    status = "success" if response.status_code == 200 else f"http_{response.status_code}"
                    return final_url, status, domain

                if response.status_code == 404:
                    return None, "expired", None

                last_error = f"http_{response.status_code}"

            except requests.exceptions.Timeout:
                last_error = "timeout"
            except requests.exceptions.ConnectionError:
                last_error = "connection_error"
            except requests.exceptions.TooManyRedirects:
                last_error = "too_many_redirects"
            except requests.exceptions.RequestException as e:
                last_error = str(e)[:30]

            # Progressive backoff before retry
            if attempt < self.max_attempts - 1:
                delay = base_delay + (attempt * increment)
                delay = delay + random.uniform(-0.5, 0.5)
                delay = max(0.5, delay)
                time.sleep(delay)

        return None, last_error, None

    def _extract_domain(self, url: str) -> str:
        """Extract domain from a URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain or "unknown"
        except Exception:
            return "unknown"

    def _extract_domain_from_vertex_url(self, url: str) -> Optional[str]:
        """Try to extract a hint about the domain from Vertex URL (limited)."""
        # Vertex URLs don't contain the domain in a parseable way
        # This is a placeholder - in practice we rely on the google_domain
        # field from the original extraction
        return None

    def _apply_mapping(
        self,
        df: pd.DataFrame,
        url_column: str,
        mapping: Dict[str, Dict],
    ) -> pd.DataFrame:
        """Apply the URL mapping to the DataFrame."""
        df = df.copy()

        # Add new columns for resolved data
        df["resolved_url"] = None
        df["resolved_domain"] = None
        df["resolution_status"] = None

        for idx, row in df.iterrows():
            url = row[url_column]
            if pd.isna(url) or not isinstance(url, str):
                continue

            if url in mapping:
                info = mapping[url]
                df.at[idx, "resolved_url"] = info["resolved_url"]
                df.at[idx, "resolved_domain"] = info["domain"]
                df.at[idx, "resolution_status"] = info["status"]

                # Update the main URL column with resolved URL (if available)
                if info["resolved_url"]:
                    df.at[idx, url_column] = info["resolved_url"]

        return df

    def _print_summary(self, mapping: Dict[str, Dict], output_path: Path):
        """Print a summary of the resolution results."""
        print(f"\n{'='*60}")
        print(f"Resolution Summary")
        print(f"{'='*60}")

        # Count by status
        status_counts = {}
        domain_counts = {}

        for url, info in mapping.items():
            status = info["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

            domain = info["domain"]
            if domain:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        print(f"\nBy status:")
        for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
            print(f"  {status}: {count}")

        if domain_counts:
            print(f"\nTop domains:")
            for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {domain}: {count}")

        print(f"\nOutput file: {output_path}")
        print(f"{'='*60}\n")


def resolve_google_urls(csv_path: str | Path, **kwargs) -> Optional[Path]:
    """Convenience function to resolve URLs in a CSV file.

    Args:
        csv_path: Path to the detail CSV file
        **kwargs: Additional arguments passed to GoogleURLResolver

    Returns:
        Path to the resolved CSV, or None if failed
    """
    resolver = GoogleURLResolver(**kwargs)
    return resolver.resolve_batch_output(csv_path)
