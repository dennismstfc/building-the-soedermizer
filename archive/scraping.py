import requests
from bs4 import BeautifulSoup
import json

from standard_values import DATA_PATH, RAW_DATA_PATH

def scrape_gendered_words(url: str = "https://geschicktgendern.de/") -> None:
    # Fetch content
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to fetch the page")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    rows = soup.find_all("tr", class_=lambda x: x and x.startswith("row-"))

    non_gendered_words = []
    gendered_words = []

    for row in rows:
        # Non-gendered words 
        columns = row.find_all("td", class_="column-1")
        for column in columns:
            non_gendered_words.append(column.text.strip())

        # Gendered words
        columns = row.find_all("td", class_="column-2")
        for column in columns:
            gendered_words.append(column.text.strip())

    word_data = {}

    for non_gendered, gendered in zip(non_gendered_words, gendered_words):
        first_letter = non_gendered[0].upper()
        # Initialize the dictionary for the first letter if not already present
        if first_letter not in word_data:
            word_data[first_letter] = {}

        # Store the non-gendered word
        word_data[first_letter][non_gendered] = {"non_gendered": non_gendered, "gendered": gendered.split(";")}

    if not DATA_PATH.exists():
        DATA_PATH.mkdir()

    # Save the data to a JSON file
    with open(RAW_DATA_PATH, 'w', encoding='utf-8') as json_file:
        json.dump(word_data, json_file, ensure_ascii=False, indent=4)

    print("Scraping completed and saved as json")
   
if __name__ == "__main__":
    scrape_gendered_words()