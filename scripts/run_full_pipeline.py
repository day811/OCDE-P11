#!/usr/bin/env python3
"""
Full RAG Data Pipeline Orchestration
Executes: Fetch → Preprocess → Vectorize → Index
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from src.vector.data_fetcher import fetch_events_snapshot
from src.vector.preprocessing import preprocess_snapshot
from src.vector.vectorization import EventVectorizer


class PipelineOrchestrator:
    """Orchestrate complete data pipeline"""
    
    def __init__(self, region: str, days_back: int, max_pages: int, snapshot_date = None):
        """
        Initialize orchestrator.
        
        Args:
            region: Geographic region
            days_back: Historical period (days)
            snapshot_date: Force specific snapshot date (YYYY-MM-DD)
        """
        self.region = region
        self.max_pages = max_pages
        self.days_back = days_back
        
        if snapshot_date is None:
            self.snapshot_date = datetime.now().strftime("%Y-%m-%d")
        else:
            self.snapshot_date = snapshot_date
        
        self.raw_snapshot_path = None
        self.processed_path = ""
    
    def print_header(self) -> None:
        """Print pipeline header"""
        print("\n" + "="*70)
        print("RAG DATA PIPELINE - FULL EXECUTION")
        print("="*70)
        print(f"Snapshot Date:  {self.snapshot_date}")
        print(f"Region:         {self.region}")
        print(f"Historical:     {self.days_back} days")
        print("="*70 + "\n")
    
    def step_1_fetch(self) -> bool:
        """
        Step 1: Fetch events from Open Agenda.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n[1/2] FETCHING from Open Agenda...")
            print("-" * 70)
            
            self.raw_snapshot_path = fetch_events_snapshot(
                region=self.region,
                days_back=self.days_back,
                max_pages=self.max_pages,
                snapshot_date=self.snapshot_date
            )
            
            print(f"✅ Fetch complete: {self.raw_snapshot_path}\n")
            return True
        
        except Exception as e:
            print(f"\n❌ Fetch failed: {str(e)}\n")
            return False
    
    def step_2_preprocess(self) -> bool:
        """
        Step 2: Preprocess and clean events.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("\n[2/2] PREPROCESSING data...")
            print("-" * 70)
            
            if not self.raw_snapshot_path:
                raise ValueError("Snapshot path not set. Run step 1 first.")
            
            self.processed_path = preprocess_snapshot(
                snapshot_path=self.raw_snapshot_path,
                days_back=self.days_back
            )
            
            print(f"✅ Preprocessing complete: {self.processed_path}\n")
            return True
        
        except Exception as e:
            print(f"\n❌ Preprocessing failed: {str(e)}\n")
            return False
    
    def step3_vectorize_and_index(self) -> bool:
        """NOUVEAU : Vectoriser + Indexer dans Faiss"""
        print("\n[STEP 3] Vectorizing and indexing with Faiss...")
        try:
            vectorizer = EventVectorizer(
                model_name="mistral-embed",
                api_key=Config.MISTRAL_API_KEY
            )
            index_path, metadata_path = vectorizer.run_full_vectorization_pipeline(
                processed_path= self.processed_path,
                snapshot_date=self.snapshot_date,
                chunk_size=500
            )
            return True        
        except Exception as e:
            print(f"❌ Vectorization failed: {e}")
            return False


    def print_summary(self, success: bool) -> int:
        """Print execution summary"""
        print("\n" + "="*70)
        
        if success:
            print("✅ PIPELINE COMPLETE!")
            print("="*70)
            print(f"Snapshot:       {self.snapshot_date}")
            print(f"Raw data:       {self.raw_snapshot_path}")
            print(f"Processed:      {self.processed_path}")
            
            # Check if this is development snapshot
            if self.snapshot_date == Config.DEV_SNAPSHOT_DATE:
                print(f"\n🔒 This is your DEVELOPMENT snapshot - FROZEN for testing")
            else:
                print(f"\n⭐ This is a LIVE snapshot - will be updated on next pipeline run")
            
            print("="*70 + "\n")
            
            return 0
        else:
            print("❌ PIPELINE FAILED!")
            print("="*70 + "\n")
            return 1
    
    def run(self) -> int:
        """
        Execute complete pipeline.
        
        Returns:
            0 if successful, 1 if failed
        """
        self.print_header()
        Config.print_config()
        
        # Step 1: Fetch
        if not self.step_1_fetch():
            return self.print_summary(False)
        
        # Step 2: Preprocess
        if not self.step_2_preprocess():
            return self.print_summary(False)
        
        # Step 3: Preprocess
        if not self.step3_vectorize_and_index():
            return self.print_summary(False)
 
         # Success
        return self.print_summary(True)


def main():
    """Parse arguments and run pipeline"""
    parser = argparse.ArgumentParser(
        description="RAG Data Pipeline - Fetch, Preprocess, Vectorize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch today's data for Occitanie
  python scripts/run_full_pipeline.py

  # Fetch for specific region and period
  python scripts/run_full_pipeline.py --region "Toulouse" --days 180

  # Force specific snapshot date (useful for reproducible development)
  python scripts/run_full_pipeline.py --snapshot-date 2026-01-25

  # List available snapshots
  python scripts/run_full_pipeline.py --list-indexes
        """
    )
    
    parser.add_argument(
        "--environment",
        default=Config.ENVIRONMENT,
        help=f"Execution mode (default: {Config.ENVIRONMENT})"
    )
    
    parser.add_argument(
        "--region",
        default=Config.REGION,
        help=f"Geographic region (default: {Config.REGION})"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=Config.DAYS_BACK,
        help=f"Historical period in days (default: {Config.DAYS_BACK})"
    )
    
    parser.add_argument(
        "--snapshot-date",
        help="Force specific snapshot date (YYYY-MM-DD). Default: today"
    )
    
    parser.add_argument(
        "--list-indexes",
        action="store_true",
        help="List all available snapshots and exit"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=Config.MAX_PAGES,
        help="Number of pages to collect"
    )

    parser.add_argument(
        '--vectorize-only',
        action='store_true',
        help="Run only Step 3 (vectorization & indexing) using an existing processed snapshot"
    )

    args = parser.parse_args()
    
    # Handle list-indexes command
    if args.list_indexes:
        Config.print_available_indexes()
        return 0
    
    # Run pipeline
    orchestrator = PipelineOrchestrator(
        region=args.region,
        days_back=args.days,
        max_pages = args.max_pages,
        snapshot_date=args.snapshot_date,
    )
    
    # Handle vectorize-only command: run vectorization only using existing processed snapshot
    if args.vectorize_only:
        # On a besoin d'une date de snapshot explicite
        if not args.snapshot_date:
            print("❌ --vectorize-only requires --snapshot-date YYYY-MM-DD")
            return 1
        
        # Construire le chemin du snapshot pré‑traité
        processed_path = Config.get_processed_snapshot_path(args.snapshot_date)
        if not Path(processed_path).exists():
            print(f"❌ Processed snapshot not found: {processed_path}")
            print(f"   Run steps 1 & 2 first for this date.")
            return 1
        orchestrator.processed_path = processed_path
        ok = orchestrator.step3_vectorize_and_index()
        return orchestrator.print_summary(ok)
    else:
        return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
