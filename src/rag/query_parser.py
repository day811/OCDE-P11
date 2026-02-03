# src/rag/query_parser.py
import re
from datetime import datetime, timedelta
from typing import Optional, Dict

class QueryParser:
    """Extract date and city constraints from user questions"""
    
    CITIES = [
        'toulouse', 'montpellier', 'nîmes', 'perpignan', 'albi',
        'rodez', 'cahors', 'figeac', 'auch', 'lunel',
        'sete', 'millau', 'narbonne', "carcassonne", "foix", "montauban", 
    ]
    DEPARTMENTS = ['ariege', 'aude', 'aveyron', 'gard', 'haute-garonne', 'herault', 'lot',
                   'lozere', 'hautes-pyrenees', 'pyrenees-orientales', 'tarn', 'tarn-et-garonne']
    
    @staticmethod
    def parse_date(query: str) -> Optional[tuple[datetime,int]]:
        """Extract date from query"""
        query_lower = query.lower()
        
        #On a precise date
        match = re.search(r'\ble\s+(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[0,1,2])[\/\-\s](?:20|)([234][0-9]|)', query_lower +" ")
        if match and match.lastindex >= 2: # pyright: ignore[reportOptionalOperand]
            dd = int(match[1])
            mm = int(match[2])
            aa = int("20" + match[3] if match[3] else datetime.now().strftime('%y'))
            return (datetime(aa,mm,dd),0)

        # Today/tonight
        if re.search(r'\b(ce soir|aujourd\'hui|ce jour|ce matin|cet après-midi)\b', query_lower):
            return (datetime.today(),0)
        
        # Tomorrow
        if re.search(r'\b(demain)\b', query_lower):
            return (datetime.today()+ timedelta(days=1), 0)
        
        # Relative days: "dans X jours"
        match = re.search(r'\bdans\s+(\d+)\s+(jour|jours)\b', query_lower)
        if match:
            days = int(match.group(1))
            return (datetime.today() + timedelta(days=days), 0)
        
        # This weekend
        if re.search(r'\b(ce weekend|ce week-end)\b', query_lower):
            today = datetime.today()
            days_until_saturday = (5 - today.weekday()) % 7 or 7
            
            return (today + timedelta(days=days_until_saturday) ,1)
        
        # Next week
        if re.search(r'\b(semaine prochaine)\b', query_lower):
            today = datetime.today()
            days_until_monday = (- today.weekday()) % 7 or 7
             
            return (today + timedelta(days=days_until_monday) ,6)
        
        return None
    
    @staticmethod
    def parse_city(query: str) -> Optional[str]:
        """Extract city from query"""
        query_lower = query.lower()
        
        for city in QueryParser.CITIES:
            if re.search(rf'\b(à|dans|de|près)\s+{city}\b', query_lower):
                return city.capitalize()
            if re.search(rf'\b{city}\b', query_lower):
                return city.capitalize()
        
        return None
    
    @staticmethod
    def parse_department(query: str) -> Optional[str]:
        """Extract department from query"""
        query_lower = query.lower()
        
        for city in QueryParser.DEPARTMENTS:
            if re.search(rf'\b(à|dans|de|près)\s+{city}\b', query_lower):
                return city.capitalize()
            if re.search(rf'\b{city}\b', query_lower):
                return city.capitalize()
        
        return None

    @staticmethod
    def parse_constraints(query: str) -> Dict:
        """Parse date and city from query"""
        return {
            'date': QueryParser.parse_date(query),
            'city': QueryParser.parse_city(query),
            'dept': QueryParser.parse_department(query)
        }
