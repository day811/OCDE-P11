from typing import Optional
from datetime import datetime, timedelta

def clean_location(location:str) -> str:
    location = location.strip().lower()
    """ Remove accents from text """
    accents = { 'a': ['à', 'ã', 'á', 'â'],
                'e': ['é', 'è', 'ê', 'ë'],
                'i': ['î', 'ï'],
                'u': ['ù', 'ü', 'û'],
                'o': ['ô', 'ö'],
                ' ': ['-','/'] 
                }
    for (char, accented_chars) in accents.items():
        for accented_char in accented_chars:
            location = location.replace(accented_char, char)
    return location    

def flat_date_constraints(
    self,
    date_constraint: Optional[tuple[datetime,int]] = None,
):

    result = ""
    if date_constraint and len(date_constraint):
        begin  = date_constraint[0]
        end = datetime = date_constraint[0] + timedelta(date_constraint[1])
        if not date_constraint[1]:
            result = f"\nDate : le {begin.date().strftime("%d-%m-%y")}"    
        else:
            result = f"\nDate : du {begin.date().strftime("%d-%m-%y")} au {end.date().strftime("%d-%m-%y")}"  
    return result

