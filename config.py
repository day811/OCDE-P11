"""
RAG System Configuration
Manages environment, snapshot versions, and paths
"""

from enum import Enum
from pathlib import Path
import os
from dotenv import load_dotenv


load_dotenv()


class Environment(Enum):
    """Execution environment"""
    DEVELOPMENT = "development"  # Phase 2-5: Fixed snapshot for testing
    DEMO = "demo"                # Soutenance: Latest snapshot
    PRODUCTION = "production"    # After delivery


class SnapshotConfig:
    """Manage snapshot versions and index selection"""
    
    # ============= PATHS =============
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    INDEXES_DIR = DATA_DIR / "indexes"
    
    # Create directories if they don't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============= SNAPSHOT CONFIGURATION =============
    # FROZEN snapshot for reproducible testing (Phase 2-5)
    DEVELOPMENT_SNAPSHOT_DATE = "2026-01-26"
    
    # Current execution environment
    ENVIRONMENT = Environment[os.getenv("RAG_ENVIRONMENT", "development").upper()]
    
    # ============= API CONFIGURATION =============
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
    
    # ============= DATA PARAMETERS =============
    # Geographic region for filtering
    REGION = os.getenv("RAG_REGION", "Occitanie")
    
    # Historical period (days) for data collection
    DAYS_BACK = int(os.getenv("RAG_DAYS_BACK", "365"))

    # Max pages limit to fetch
    MAX_PAGES = os.getenv("MAX_PAGES", None)
    if MAX_PAGES : MAX_PAGES=int(MAX_PAGES)
    
    # ============= METHODS =============
    
    @classmethod
    def set_environment(cls, env: Environment) -> None:
        """
        Change execution environment.
        
        Args:
            env: Target environment (DEVELOPMENT, DEMO, or PRODUCTION)
        """
        cls.ENVIRONMENT = env
        print(f"📌 Environment set to: {env.value.upper()}")
    
    @classmethod
    def get_active_index_path(cls) :
        """
        Get the appropriate Faiss index based on environment.
        
        Returns:
            Path to Faiss index file
            
        Raises:
            FileNotFoundError: If expected index doesn't exist
        """
        if cls.ENVIRONMENT == Environment.DEVELOPMENT:
            # Always use FROZEN development snapshot
            index_path = cls.INDEXES_DIR / f"faiss_index_{cls.DEVELOPMENT_SNAPSHOT_DATE}.bin"
            
            if not index_path.exists():
                raise FileNotFoundError(
                    f"\n❌ Development index not found: {index_path}\n"
                    f"Run: python scripts/run_full_pipeline.py "
                    f"--snapshot-date {cls.DEVELOPMENT_SNAPSHOT_DATE}\n"
                )
            
            print(f"🔒 [DEVELOPMENT] Loading index from {cls.DEVELOPMENT_SNAPSHOT_DATE}")
            return str(index_path)
        
        elif cls.ENVIRONMENT == Environment.DEMO:
            # Use LATEST available snapshot
            index_files = sorted(cls.INDEXES_DIR.glob("faiss_index_*.bin"))
            
            if not index_files:
                raise FileNotFoundError(
                    f"\n❌ No index files found in {cls.INDEXES_DIR}\n"
                    f"Run: python scripts/run_full_pipeline.py\n"
                )
            
            latest_index = index_files[-1]
            latest_date = latest_index.stem.replace("faiss_index_", "")
            print(f"⭐ [DEMO] Loading LATEST index from {latest_date}")
            return str(latest_index)
        
        elif cls.ENVIRONMENT == Environment.PRODUCTION:
            raise NotImplementedError("Production mode not implemented for POC")
    
    @classmethod
    def get_active_metadata_path(cls) -> str:
        """
        Get the appropriate metadata file based on active index.
        
        Returns:
            Path to metadata JSON file
        """
        index_path = cls.get_active_index_path()
        metadata_path = index_path.replace("faiss_index_", "metadata_").replace(".bin", ".json")
        
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        return metadata_path
    
    @classmethod
    def print_available_indexes(cls) -> None:
        """Display all available snapshots with their properties"""
        print("\n" + "="*70)
        print("📚 AVAILABLE SNAPSHOTS")
        print("="*70)
        
        index_files = sorted(cls.INDEXES_DIR.glob("faiss_index_*.bin"))
        
        if not index_files:
            print("  ❌ No indexes found. Run pipeline first.")
            print("="*70 + "\n")
            return
        
        for idx, fpath in enumerate(index_files, 1):
            snapshot_date = fpath.stem.replace("faiss_index_", "")
            file_size_mb = fpath.stat().st_size / (1024 * 1024)
            
            # Check if this is development or latest
            markers = []
            if snapshot_date == cls.DEVELOPMENT_SNAPSHOT_DATE:
                markers.append("🔒 DEVELOPMENT (FROZEN)")
            if idx == len(index_files):
                markers.append("⭐ LATEST")
            
            marker_str = " | ".join(markers) if markers else ""
            print(f"  {idx}. {snapshot_date}  ({file_size_mb:.1f} MB)  {marker_str}")
        
        print("="*70 + "\n")
    
    @classmethod
    def get_raw_snapshot_path(cls, snapshot_date: str = "") -> str:
        """Get path for raw snapshot JSON"""
        if snapshot_date == "":
            from datetime import datetime
            snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        return str(cls.RAW_DATA_DIR / f"raw_snapshot_{snapshot_date}.json")
    
    @classmethod
    def get_processed_snapshot_path(cls, snapshot_date: str = "") -> str:
        """Get path for processed snapshot JSON"""
        if snapshot_date == "":
            from datetime import datetime
            snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        return str(cls.PROCESSED_DATA_DIR / f"processed_events_{snapshot_date}.json")
    
    @classmethod
    def get_index_path(cls, snapshot_date: str = "") -> str:
        """Get path for Faiss index"""
        if snapshot_date == "":
            from datetime import datetime
            snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        return str(cls.INDEXES_DIR / f"faiss_index_{snapshot_date}.bin")
    
    @classmethod
    def get_metadata_path(cls, snapshot_date: str = "") -> str:
        """Get path for metadata JSON"""
        if snapshot_date == "":
            from datetime import datetime
            snapshot_date = datetime.now().strftime("%Y-%m-%d")
        
        return str(cls.INDEXES_DIR / f"metadata_{snapshot_date}.json")
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration"""
        print("\n" + "="*70)
        print("⚙️  CONFIGURATION")
        print("="*70)
        print(f"Environment:          {cls.ENVIRONMENT.value.upper()}")
        print(f"Region:               {cls.REGION}")
        print(f"Historical period:    {cls.DAYS_BACK} days")
        print(f"Dev snapshot date:    {cls.DEVELOPMENT_SNAPSHOT_DATE}")
        print(f"Data directory:       {cls.DATA_DIR}")
        print(f"Indexes directory:    {cls.INDEXES_DIR}")
        print("="*70 + "\n")
