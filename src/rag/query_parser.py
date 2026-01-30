# src/rag/query_parser.py
import re
from datetime import datetime, timedelta
from typing import Optional, Dict

class QueryParser:
    """Extract date and city constraints from user questions"""
    
    CITIES = [
        'toulouse', 'montpellier', 'nîmes', 'perpignan', 'albi',
        'rodez', 'cahors', 'figeac', 'auch', 'tarbes',
        'pau', 'biarritz', 'bayonne'
    ]
    
    @staticmethod
    def parse_date(query: str) -> Optional[datetime]:
        """Extract date from query"""
        query_lower = query.lower()
        
        # Today/tonight
        if re.search(r'\b(ce soir|aujourd\'hui|today|ce jour|ce matin|cet après-midi)\b', query_lower):
            return datetime.today()
        
        # Tomorrow
        if re.search(r'\b(demain|tomorrow)\b', query_lower):
            return datetime.today() + timedelta(days=1)
        
        # Relative days: "dans X jours"
        match = re.search(r'\bdans\s+(\d+)\s+(jour|jours|day|days)\b', query_lower)
        if match:
            days = int(match.group(1))
            return datetime.today() + timedelta(days=days)
        
        # This weekend
        if re.search(r'\b(ce weekend|ce week-end|this weekend)\b', query_lower):
            today = datetime.today()
            days_until_saturday = (5 - today.weekday()) % 7
            if days_until_saturday == 0:
                days_until_saturday = 7
            return today + timedelta(days=days_until_saturday)
        
        # Next week
        if re.search(r'\b(semaine prochaine|next week)\b', query_lower):
            return datetime.today() + timedelta(weeks=1)
        
        return None
    
    @staticmethod
    def parse_city(query: str) -> Optional[str]:
        """Extract city from query"""
        query_lower = query.lower()
        
        for city in QueryParser.CITIES:
            if re.search(rf'\b(à|in|de|from)\s+{city}\b', query_lower):
                return city.capitalize()
            if re.search(rf'\b{city}\b', query_lower):
                return city.capitalize()
        
        return None
    
    @staticmethod
    def parse_constraints(query: str) -> Dict:
        """Parse date and city from query"""
        return {
            'date': QueryParser.parse_date(query),
            'city': QueryParser.parse_city(query)
        }
