"""
Data Preprocessing Module
Cleans, validates, and structures events for vectorization
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventPreprocessor:
    """Clean and preprocess event data from snapshots"""
    
    def __init__(self, snapshot_path: str):
        """
        Initialize preprocessor with snapshot data.
        
        Args:
            snapshot_path: Path to raw snapshot JSON
        """
        self.snapshot_path = snapshot_path
        self.df = pd.DataFrame()
        self.metadata = None
        self.stats = {}
        
        self._load_snapshot()
    
    def _load_snapshot(self) -> None:
        """Load and parse JSON snapshot"""
        logger.info(f"Loading snapshot: {self.snapshot_path}")
        
        with open(self.snapshot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data.get("metadata", {})
        events = data.get("events", [])
        
        # Convert to DataFrame
        self.df = pd.DataFrame(events)
        
        logger.info(f"Loaded {len(self.df)} events")
        self.stats['loaded'] = len(self.df)
    
    def _extract_event_date(self, dates_field: list) -> Optional[datetime]:
        """
        Extract primary event date from dates field.
        
        Args:
            dates_field: List of date objects from Open Agenda
            
        Returns:
            Datetime object or None
        """
        if not dates_field or not isinstance(dates_field, list):
            return None
        
        try:
            # Get first date from list
            first_date = dates_field[0]
            
            if isinstance(first_date, dict):
                # Format: {"start": "2026-01-25T19:00:00+02:00"}
                date_str = first_date.get("start") or first_date.get("begin")
            else:
                date_str = first_date
            
            # Parse ISO format
            if date_str:
                return pd.to_datetime(date_str)
        
        except Exception as e:
            logger.debug(f"Error parsing date: {str(e)}")
        
        return None
    
    def _extract_location(self, location_field: dict) -> str:
        """
        Extract readable location from Open Agenda location object.
        
        Args:
            location_field: Location dictionary from API
            
        Returns:
            Formatted location string
        """
        if not location_field:
            return "Unknown"
        
        if isinstance(location_field, str):
            return location_field
        
        if isinstance(location_field, dict):
            # Try common location field names
            return (
                location_field.get("label") or
                location_field.get("name") or
                location_field.get("city") or
                str(location_field)
            )
        
        return "Unknown"
    
    def standardize_columns(self) -> None:
        """
        Standardize and extract relevant columns from raw data.
        Creates consistent DataFrame structure.
        """
        logger.info("Standardizing columns...")
        
        # Create standardized DataFrame
        standardized = []
        
        for idx, row in self.df.iterrows():
            try:
                event_date = self._extract_event_date(row.get("dates", []))
                location = self._extract_location(row.get("location", {}))
                
                standardized_row = {
                    "id": row.get("id", f"unknown_{idx}"),
                    "title": row.get("title", "No title").strip(),
                    "description": row.get("description", "").strip(),
                    "event_date": event_date,
                    "location": location,
                    "url": row.get("url", ""),
                    "uid": row.get("uid", ""),
                }
                
                standardized.append(standardized_row)
            
            except Exception as e:
                logger.debug(f"Error standardizing row {idx}: {str(e)}")
                continue
        
        self.df = pd.DataFrame(standardized)
        logger.info(f"✅ Standardized to {len(self.df)} records")
    
    def drop_duplicates(self) -> None:
        """Remove duplicate events by ID"""
        initial_count = len(self.df)
        
        self.df = self.df.drop_duplicates(subset=['id'], keep='first')
        
        removed = initial_count - len(self.df)
        logger.info(f"Removed {removed} duplicates")
        self.stats['duplicates_removed'] = removed
    
    def handle_missing_values(self) -> None:
        """Handle missing values appropriately"""
        logger.info("Handling missing values...")
        
        # Fill missing descriptions
        self.df['description'] = self.df['description'].fillna('')
        
        # Fill missing URLs
        self.df['url'] = self.df['url'].fillna('')
        
        # Drop rows with missing date (cannot filter without it)
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=['event_date'])
        dropped = before_drop - len(self.df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} events with missing dates")
            self.stats['missing_dates_dropped'] = dropped
        
        # Drop rows with missing location
        before_drop = len(self.df)
        self.df = self.df[self.df['location'] != 'Unknown']
        dropped = before_drop - len(self.df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} events with unknown location")
            self.stats['unknown_location_dropped'] = dropped
    
    def filter_by_date_range(self, days_back: int = 365) -> None:
        """
        Filter events within specified date range.
        Keeps events from (now - days_back) to (now + 1 year in future).
        
        Args:
            days_back: How many days back to keep (default: 365)
        """
        logger.info(f"Filtering by date range ({days_back} days back)...")
        
        now = datetime.now()
        cutoff_past = now - timedelta(days=days_back)
        cutoff_future = now + timedelta(days=365)  # Keep 1 year in future
        
        before = len(self.df)
        
        self.df = self.df[
            (self.df['event_date'] >= cutoff_past) &
            (self.df['event_date'] <= cutoff_future)
        ]
        
        removed = before - len(self.df)
        logger.info(f"Filtered by date: {removed} events removed")
        logger.info(f"Date range: {cutoff_past.date()} to {cutoff_future.date()}")
        self.stats['date_filtered'] = removed
    
    def validate_text_content(self, min_length: int = 10) -> None:
        """
        Validate that events have meaningful text content.
        Removes events with very short descriptions.
        
        Args:
            min_length: Minimum combined text length
        """
        logger.info(f"Validating text content (min {min_length} chars)...")
        
        before = len(self.df)
        
        # Combine title and description, check length
        self.df['combined_text'] = self.df['title'].fillna('').str.cat(self.df['description'].fillna(''), sep='')
        self.df = self.df[self.df['combined_text'].str.len() >= min_length]
        self.df = self.df.drop(columns=['combined_text'])
        
        removed = before - len(self.df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} events with insufficient text")
            self.stats['insufficient_text_removed'] = removed
    
    def sort_and_reset_index(self) -> None:
        """Sort by event date and reset index"""
        self.df = self.df.sort_values('event_date').reset_index(drop=True)
        logger.info("Sorted by event date")
    
    def run_full_pipeline(self, days_back: int = 365) -> None:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            days_back: Historical period (days)
        """
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING PIPELINE")
        logger.info("="*70 + "\n")
        
        self.standardize_columns()
        self.drop_duplicates()
        self.handle_missing_values()
        self.filter_by_date_range(days_back=days_back)
        self.validate_text_content()
        self.sort_and_reset_index()
        
        logger.info("\n" + "="*70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*70)
        self.print_stats()
    
    def print_stats(self) -> None:
        """Print preprocessing statistics"""
        logger.info("\nStatistics:")
        logger.info(f"  Initial events:        {self.stats.get('loaded', 0)}")
        logger.info(f"  Duplicates removed:    {self.stats.get('duplicates_removed', 0)}")
        logger.info(f"  Missing dates:         {self.stats.get('missing_dates_dropped', 0)}")
        logger.info(f"  Unknown location:      {self.stats.get('unknown_location_dropped', 0)}")
        logger.info(f"  Date filtered:         {self.stats.get('date_filtered', 0)}")
        logger.info(f"  Insufficient text:     {self.stats.get('insufficient_text_removed', 0)}")
        logger.info(f"  Final events:          {len(self.df)}\n")
    
    def save_processed_data(self, output_path) -> str:
        """
        Save processed events as CSV.
        
        Args:
            output_path: Path to save CSV file
            
        Returns:
            Path to saved CSV
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert datetime to string for CSV serialization
        df_export = self.df.copy()
        df_export['event_date'] = df_export['event_date'].astype(str)
        
        df_export.to_csv(output_path, index=False, encoding='utf-8')
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Processed data saved: {output_path}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Rows: {len(df_export)}")
        
        return str(output_path)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get processed DataFrame"""
        return self.df


def preprocess_snapshot(
    snapshot_path: str,
    days_back: int = 365,
    output_path = None
) -> str:
    """
    High-level function to preprocess snapshot.
    
    Args:
        snapshot_path: Path to raw snapshot JSON
        days_back: Historical period (days)
        output_path: Path to save processed CSV
        
    Returns:
        Path to processed CSV file
    """
    from config import SnapshotConfig
    
    # Initialize preprocessor
    preprocessor = EventPreprocessor(snapshot_path)
    
    # Run pipeline
    preprocessor.run_full_pipeline(days_back=days_back)
    
    # Determine output path
    if output_path is None:
        # Extract snapshot date from input path
        snapshot_date = Path(snapshot_path).stem.replace("raw_snapshot_", "")
        output_path = SnapshotConfig.get_processed_snapshot_path(snapshot_date)
    
    # Save processed data
    processed_path = preprocessor.save_processed_data(output_path)
    
    return processed_path


if __name__ == "__main__":
    # Example usage
    from config import SnapshotConfig
    
    SnapshotConfig.print_config()
    
    # Assuming snapshot already exists
    snapshot_path = SnapshotConfig.get_raw_snapshot_path("2026-01-25")
    
    if Path(snapshot_path).exists():
        processed_path = preprocess_snapshot(
            snapshot_path=snapshot_path,
            days_back=SnapshotConfig.DAYS_BACK
        )
        print(f"\n✅ Done! Processed file: {processed_path}")
    else:
        print(f"Snapshot not found: {snapshot_path}")
        print("Run data_fetcher.py first")
