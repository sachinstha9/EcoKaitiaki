import pandas as pd
from rapidfuzz import fuzz

df = pd.read_csv("assets/nz.csv")
cities = df['city']

def get_matched_cities(terms, sentence, threshold=75):
    sentence_lower = sentence.lower()
    matches = []
    
    for term in terms:
        term_lower = term.lower()
        if ' ' not in term:
            for word in sentence_lower.split():
                if fuzz.ratio(term_lower, word) >= threshold:
                    matches.append(term)
                    break
        else:
            term_words = term_lower.split()
            window_size = len(term_words)
            words = sentence_lower.split()
            
            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i + window_size])
                if fuzz.ratio(term_lower, window) >= threshold:
                    matches.append(term)
                    break
    return matches

def get_cords(query):
    matched_cities = get_matched_cities(cities, query)
    with_cords = df[df['city'].isin(matched_cities)]
    return  df[df["city"] == "auckland"] if with_cords.empty else with_cords