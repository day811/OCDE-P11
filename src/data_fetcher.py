"""
Data Fetcher Module
Retrieves cultural events from Open Agenda API and creates immutable snapshots
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAgendaFetcher:
    """Fetch events from Open Agenda API"""
    
    # Open Agenda API endpoint (free, no auth required)
    BASE_URL = "https://www.openagenda.com/api/events"
    
    # API parameters
    TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    
    def __init__(self, region: str = "Occitanie", days_back: int = 365):
        """
        Initialize Open Agenda fetcher.
        
        Args:
            region: Geographic region (e.g., "Occitanie", "Toulouse")
            days_back: Historical period to fetch (days)
        """
        self.region = region
        self.days_back = days_back
        self.events = []
    
    def _get_date_range(self) -> tuple:
        """
        Calculate date range for API query.
        
        Returns:
            Tuple of (start_date, end_date) in YYYY-MM-DD format
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)
        
        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
    
    def _build_api_params(self, page: int = 1, limit: int = 100) -> Dict[str, Any]:
        """
        Build API parameters for Open Agenda query.
        
        Args:
            page: Page number for pagination
            limit: Events per page
            
        Returns:
            Dictionary of API parameters
        """
        start_date, end_date = self._get_date_range()
        
        params = {
            "search": self.region,           # Geographic search
            "limit": limit,                  # Items per page
            "offset": (page - 1) * limit,   # Pagination offset
            "range": f"{start_date},{end_date}",  # Date range
            "omitDescription": False,        # Include descriptions
            "sort": "-updated_at",           # Sort by most recent updates
        }
        
        return params
    
    def fetch_page(self, page: int = 1, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch single page of events from API.
        
        Args:
            page: Page number
            limit: Events per page (max 100)
            
        Returns:
            List of event dictionaries
        """
        params = self._build_api_params(page=page, limit=limit)
        
        for attempt in range(self.MAX_RETRIES):
            try:
                logger.info(f"Fetching page {page} from Open Agenda API...")
                
                response = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=self.TIMEOUT
                )
                response.raise_for_status()
                
                data = response.json()
                events = data.get("data", [])
                
                logger.info(f"✅ Page {page}: Got {len(events)} events")
                return events
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️  Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {str(e)}")
                
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"❌ Failed to fetch page {page} after {self.MAX_RETRIES} retries")
                    raise
                
                # Exponential backoff
                import time
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        return []
    
    def fetch_all(self, limit: int = 100, max_pages = None) -> List[Dict[str, Any]]:
        """
        Fetch all events within date range.
        
        Args:
            limit: Events per page (max 100)
            max_pages: Maximum pages to fetch (None = all)
            
        Returns:
            List of all events
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"FETCHING EVENTS FROM OPEN AGENDA")
        logger.info(f"{'='*70}")
        logger.info(f"Region:     {self.region}")
        logger.info(f"Period:     {self.days_back} days")
        logger.info(f"Limit:      {limit} events/page")
        
        all_events = []
        page = 1
        
        while True:
            # Fetch page
            events = self.fetch_page(page=page, limit=limit)
            
            if not events:
                logger.info(f"No more events to fetch")
                break
            
            all_events.extend(events)
            
            # Check if we should continue
            if max_pages and page >= max_pages:
                logger.info(f"Reached maximum pages ({max_pages})")
                break
            
            page += 1
            
            # Don't hammer the API
            import time
            time.sleep(0.5)
        
        logger.info(f"{'='*70}")
        logger.info(f"✅ TOTAL EVENTS FETCHED: {len(all_events)}")
        logger.info(f"{'='*70}\n")
        
        self.events = all_events
        return all_events
    
    def validate_events(self) -> int:
        """
        Validate fetched events and filter out invalid ones.
        
        Returns:
            Number of valid events
        """
        logger.info(f"Validating {len(self.events)} events...")
        
        valid_events = []
        invalid_count = 0
        
        for event in self.events:
            # Check required fields
            required_fields = ["id", "title", "description", "dates"]
            
            if all(field in event for field in required_fields):
                valid_events.append(event)
            else:
                invalid_count += 1
                logger.debug(f"Skipping invalid event: missing required fields")
        
        self.events = valid_events
        logger.info(f"✅ {len(self.events)} valid events (skipped {invalid_count})")
        
        return len(self.events)
    
    def save_snapshot(self, output_path) -> str:
        """
        Save events as immutable JSON snapshot.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Path to saved snapshot
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        snapshot_data = {
            "metadata": {
                "fetch_date": datetime.now().isoformat(),
                "region": self.region,
                "days_back": self.days_back,
                "total_events": len(self.events),
                "api_version": "openagenda_v3"
            },
            "events": self.events
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Snapshot saved: {output_path}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        
        return str(output_path)


def fetch_events_snapshot(
    region: str = "Occitanie",
    days_back: int = 365,
    snapshot_date = None,
    output_dir = None
) -> str:
    """
    High-level function to fetch and save events snapshot.
    
    Args:
        region: Geographic region
        days_back: Historical period (days)
        snapshot_date: Force specific date (YYYY-MM-DD). Default: today
        output_dir: Directory for snapshot file
        
    Returns:
        Path to saved snapshot JSON
    """
    from config import SnapshotConfig
    
    if snapshot_date is None:
        snapshot_date = datetime.now().strftime("%Y-%m-%d")
    
    if output_dir is None:
        output_path = SnapshotConfig.get_raw_snapshot_path(snapshot_date)
    else:
        output_path = Path(output_dir) / f"raw_snapshot_{snapshot_date}.json"
    
    # Fetch events
    fetcher = OpenAgendaFetcher(region=region, days_back=days_back)
    fetcher.fetch_all()
    fetcher.validate_events()
    
    # Save snapshot
    snapshot_path = fetcher.save_snapshot(str(output_path))
    
    return snapshot_path


if __name__ == "__main__":
    # Example usage
    from config import SnapshotConfig
    
    SnapshotConfig.print_config()
    
    snapshot_path = fetch_events_snapshot(
        region=SnapshotConfig.REGION,
        days_back=SnapshotConfig.DAYS_BACK
    )
    
    print(f"\n✅ Done! Snapshot: {snapshot_path}")
