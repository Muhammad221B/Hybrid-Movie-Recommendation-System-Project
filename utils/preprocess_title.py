import re

def preprocess_title(title):
    title = re.sub(r'[.,]', '', title)
    title = title.strip()
    title = title.lower()
    return title