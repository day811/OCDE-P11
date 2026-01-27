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
    
    def __init__(self, snapshot_path: str, days_back: int = 365):
        """
        Initialize preprocessor with snapshot data.
        
        Args:
            snapshot_path: Path to raw snapshot JSON
        """
        self.snapshot_path = snapshot_path
        # Extract snapshot date from filename (raw_snapshot_YYYY-MM-DD.json)
        snapshot_date_str = Path(snapshot_path).stem.replace("raw_snapshot_", "")
        snapshot_date = datetime.fromisoformat(snapshot_date_str)
        self.cutoff_date = (snapshot_date - timedelta(days=days_back)).isoformat()

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
    
    def _extract_event_date(self, dates_field) -> list:
        """
        Extract primary event date from dates field.
        
        Args:
            dates_field: List of date objects from Open Agenda
            
        Returns:
            Datetime object or None
        """
        if not dates_field or not isinstance(dates_field, str):
            return []
        filtered_dates= []
        try:
            # Convert "json" to json
            dates_field = json.loads(dates_field)
            for date_field in dates_field:
                # Filter dates that not in correct range
                if date_field['begin'] >= self.cutoff_date:
                    filtered_dates.append(date_field)    
            return filtered_dates
                
        except Exception as e:
            logger.debug(f"Error parsing date: {str(e)}")
        
        return []
    
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
                timings = self._extract_event_date(row.get("timings", []))
                if len(timings) > 0 : 
                    standardized_row = {
                        "uid": row.get("uid", ""),
                        "title_fr": str(row.get("title_fr", "No title")).strip(),
                        "description_fr": str(row.get("description", "")).strip(),
                        "longdescription_fr": str(row.get("longdescription_fr", "")).strip(),
                        "timings": timings,
                        "location_name": row.get("location_name", ""),
                        "location_city": row.get("location_city", ""),
                        "location_address": row.get("location_address", ""),
                        "conditions_fr": row.get("conditions_fr", ""),
                        "canonicalurl": row.get("canonicalurl", ""),
                        "location_lat": row.get("location_coordinates", {}).get("lat"),
                        "location_lon": row.get("location_coordinates", {}).get("lon"),
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
        
        self.df = self.df.drop_duplicates(subset=['uid'], keep='first')
        
        removed = initial_count - len(self.df)
        logger.info(f"Removed {removed} duplicates")
        self.stats['duplicates_removed'] = removed
    
    def handle_missing_values(self) -> None:
        """Handle missing values appropriately"""
        logger.info("Handling missing values...")
        
        # Fill missing title
        self.df['title_fr'] = self.df['title_fr'].fillna('')
        
        # Fill missing descriptions
        self.df['description_fr'] = self.df['description_fr'].fillna('')
        
        # Fill missing longdescription_fr
        self.df['longdescription_fr'] = self.df['longdescription_fr'].fillna('')
        
        # Fill missing URLs
        self.df['canonicalurl'] = self.df['canonicalurl'].fillna('')
        
        # Drop rows with missing date (cannot filter without it)
        before_drop = len(self.df)
        self.df = self.df.dropna(subset=['timings'])
        dropped = before_drop - len(self.df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} events with missing dates")
            self.stats['missing_dates_dropped'] = dropped
        
        # Drop rows with missing location
        before_drop = len(self.df)
        self.df = self.df[self.df['location_city'] != 'Unknown']
        dropped = before_drop - len(self.df)
        
        if dropped > 0:
            logger.warning(f"Dropped {dropped} events with unknown location")
            self.stats['unknown_location_dropped'] = dropped
    

    
    def validate_text_content(self, min_length: int = 10) -> None:
        """
        Validate that events have meaningful text content.
        Removes events with very short descriptions.
        
        Args:
            min_length: Minimum combined text length
        """
        logger.info(f"Validating text content (min {min_length} chars)...")
        
        before = len(self.df)
        
        # Combine title, description and longdescription, check length
        self.df['text_length'] = (
        self.df['title_fr'].str.len() + 
        self.df['conditions_fr'].str.len() + 
        self.df['description_fr'].str.len()+
        self.df['longdescription_fr'].str.len()
        )
    
        # Keep only events with sufficient text
        self.df = self.df[self.df['text_length'] >= min_length]
        self.df = self.df.drop(columns=['text_length'])
       
        removed = before - len(self.df)
        
        if removed > 0:
            logger.warning(f"Removed {removed} events with insufficient text")
            self.stats['insufficient_text_removed'] = removed
    

    def sort_and_reset_index(self) -> None:
        """Sort by event date and reset index"""
        self.df = self.df.sort_values('uid').reset_index(drop=True)
        logger.info("Sorted by event uid")
    

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
        Save processed events as JSON.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Path to saved JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df_export = self.df.copy()
        
        # Save as JSON (preserve complex types)
        df_export.to_json(output_path, orient='records', indent=2, force_ascii=False)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Processed data saved: {output_path}")
        logger.info(f"   Size: {file_size_mb:.2f} MB")
        logger.info(f"   Rows: {len(df_export)}")
        
        return str(output_path)
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get processed DataFrame"""
        return self.df


def preprocess_snapshot(snapshot_path: str, days_back: int = 365, output_path = None) -> str:
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
    preprocessor = EventPreprocessor(snapshot_path, days_back = SnapshotConfig.DAYS_BACK)
    
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
