"""
Token usage tracking and accounting
Logs and reports token consumption from Mistral API calls
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenAccounting:
    """Track and report token usage across RAG operations"""
    
    def __init__(self, log_dir: Path = Path("data/token_logs")):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"tokens_{self.session_id}.jsonl"
    
    def log_vectorization(self, chunks_count: int, tokens_used: int, 
                         operation: str = "vectorization"):
        """Log vectorization operation"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'chunks': chunks_count,
            'tokens': tokens_used,
            'tokens_per_chunk': tokens_used / chunks_count if chunks_count > 0 else 0
        }
        self._write_record(record)
        logger.info(f"{operation}: {tokens_used} tokens for {chunks_count} chunks "
                   f"({record['tokens_per_chunk']:.2f} tok/chunk)")
    
    def log_search(self, query_tokens: int, context_tokens: int = 0, 
                  llm_tokens: int = 0, operation: str = "search"):
        """Log search operation (embedding + optional LLM)"""
        total = query_tokens + context_tokens + llm_tokens
        record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'query_tokens': query_tokens,
            'context_tokens': context_tokens,
            'llm_tokens': llm_tokens,
            'total_tokens': total
        }
        self._write_record(record)
        logger.info(f"{operation}: query={query_tokens}, context={context_tokens}, "
                   f"llm={llm_tokens}, total={total}")
    
    def get_session_report(self) -> Dict:
        """Generate report for current session"""
        logs = self._load_session_logs()
        
        vectorization = [l for l in logs if l['operation'] in ['vectorization', 'indexing']]
        searches = [l for l in logs if l['operation'] in ['search', 'chat']]
        
        report = {
            'session_id': self.session_id,
            'session_start': logs[0]['timestamp'] if logs else None,
            'session_end': logs[-1]['timestamp'] if logs else None,
            'vectorization': {
                'operations': len(vectorization),
                'total_chunks': sum(l.get('chunks', 0) for l in vectorization),
                'total_tokens': sum(l.get('tokens', 0) for l in vectorization),
                'avg_tokens_per_chunk': (
                    sum(l.get('tokens', 0) for l in vectorization) / 
                    sum(l.get('chunks', 0) for l in vectorization)
                    if sum(l.get('chunks', 0) for l in vectorization) > 0 else 0
                )
            },
            'searches': {
                'operations': len(searches),
                'total_tokens': sum(l.get('total_tokens', 0) for l in searches),
                'avg_tokens_per_search': (
                    sum(l.get('total_tokens', 0) for l in searches) / len(searches)
                    if len(searches) > 0 else 0
                )
            }
        }
        
        report['total_tokens'] = (
            report['vectorization']['total_tokens'] + 
            report['searches']['total_tokens']
        )
        
        return report
    
    def print_report(self):
        """Print formatted report"""
        report = self.get_session_report()
        
        print("\n" + "="*80)
        print("TOKEN USAGE REPORT")
        print("="*80)
        print(f"Session ID: {report['session_id']}")
        print(f"Duration: {report['session_start']} → {report['session_end']}")
        print()
        
        print("VECTORIZATION:")
        print(f"  Operations: {report['vectorization']['operations']}")
        print(f"  Total chunks: {report['vectorization']['total_chunks']}")
        print(f"  Total tokens: {report['vectorization']['total_tokens']}")
        print(f"  Avg tokens/chunk: {report['vectorization']['avg_tokens_per_chunk']:.2f}")
        print()
        
        print("SEARCHES:")
        print(f"  Operations: {report['searches']['operations']}")
        print(f"  Total tokens: {report['searches']['total_tokens']}")
        print(f"  Avg tokens/search: {report['searches']['avg_tokens_per_search']:.2f}")
        print()
        
        print(f"GRAND TOTAL: {report['total_tokens']} tokens")
        print("="*80 + "\n")
    
    def _write_record(self, record: Dict):
        """Write record to session log file"""
        try:
            with open(self.session_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error writing token log: {e}")
    
    def _load_session_logs(self) -> List[Dict]:
        """Load all records from session log"""
        logs = []
        try:
            if self.session_log_file.exists():
                with open(self.session_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading token logs: {e}")
        return logs


# Global accounting instance
_accounting = None

def get_accounting(log_dir = None) -> TokenAccounting:
    """Get or initialize global accounting instance"""
    global _accounting
    if _accounting is None:
        _accounting = TokenAccounting(log_dir=log_dir or Path("data/token_logs"))
    return _accounting


def reset_accounting():
    """Reset accounting instance (for testing)"""
    global _accounting
    _accounting = None