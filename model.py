from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast, BertForTokenClassification, pipeline
import torch
import re
from dateutil.parser import parse
import spacy
from datetime import datetime, timedelta

id2label = {
    0: 'O',        # Outside of a named entity
    1: 'B-PER',    # Beginning of a person's name right after another person's name
    2: 'I-PER',    # Person's name
    3: 'B-LOC',    # Beginning of a location right after another location
    4: 'I-LOC',    # Location
    5: 'B-ORG',    # Beginning of an organization right after another organization
    6: 'I-ORG',    # Organization
    7: 'B-MISC',   # Beginning of a miscellaneous entity right after another miscellaneous entity
    8: 'I-MISC',   # Miscellaneous entity
}

# Model and tokenizer for date and time
tokenizer = AutoTokenizer.from_pretrained("tokenizer_date")
model = AutoModelForTokenClassification.from_pretrained("model_date")


#Model and tokenizer for Name and Location
model_ = BertForTokenClassification.from_pretrained("model_loc")
tokenizer_ = BertTokenizerFast.from_pretrained("tokenizer_loc")

extract_date_and_time = pipeline("token-classification", model=model, tokenizer=tokenizer)

def merge_entities_date_time(sentence):
    merged_entities = {'date': [], 'time': []}
    current_date = []
    current_time = []
    
    entities = extract_date_and_time(sentence)
    for entity in entities:
        if entity['entity'].startswith('B-DATE'):
            if current_date:
                merged_entities['date'].append(' '.join(current_date))
            current_date = [entity['word']]
        elif entity['entity'].startswith('I-DATE'):
            current_date.append(entity['word'])
        elif entity['entity'].startswith('B-TIME'):
            if current_time:
                merged_entities['time'].append(''.join(current_time).replace('##', ''))
            current_time = [entity['word'].replace('##', '')]
        elif entity['entity'].startswith('I-TIME'):
            current_time.append(entity['word'].replace('##', ''))
    
    if current_date:
        merged_entities['date'].append(''.join(current_date))
    if current_time:
        merged_entities['time'].append(''.join(current_time))
    
    
    return merged_entities

def predict_entities(sentence, model, tokenizer):
    # Tokenize the input sentence and convert to tensor
    sentence = sentence.lower()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)

    # Move input tensors to the same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    predicted_labels = [id2label[pred] for pred in predictions]

    token_label_pairs = list(zip(tokens, predicted_labels))
    token_label_pairs = [(token, label) for token, label in token_label_pairs if token not in (tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token)]

    return token_label_pairs

def get_name_loc(sentence, model, tokenizer):
    token_label_pairs = predict_entities(sentence, model, tokenizer)

    # for token, label in token_label_pairs:
    #     print(f"{token}: {label}")

    person = []
    location = []

    for token, label in token_label_pairs:
        if label == "B-PER":
            if("##" in token):
                person.append(token.replace("##", ""))
            else:
                person.append(token)
        elif label == "I-PER":
            person.append(token.replace("##", ""))
        elif label == "B-ORG":
            location = [token.replace("##", "")]
        elif label == "I-ORG":
            location.append(token.replace("##", ""))

    output = {
        "person": "".join(person),
        "location": " ".join(location)
    }

    return output

def get_all_entities(sentence):

    date_time = merge_entities_date_time(sentence)
    loc_name = get_name_loc(sentence, model_, tokenizer_)

    return {**date_time, **loc_name}

class DateExtract:
  def __init__(self):
    self.date_formats = r'\b\d{1,2}/\d{1,2}/\d{2}\b|\b\d{1,2}(?:st|nd|rd|th)(?:January|February|March|April|May|June|July|August|September|October|November|December)\d{2}\b|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2},? \d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)(?:January|February|March|April|May|June|July|August|September|October|November|December)\d{2,4}\b|\b\d{1,2}(?:st|nd|rd|th)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b'
    self.nlp = spacy.load("en_core_web_sm")

  def month_to_number(self, month_name):
    date = datetime.strptime(month_name, "%B")
    return date.month

  def extract_dates(self, sentence):

    # Find all matches of date formats in the input string
    matches = re.findall(self.date_formats, sentence, re.IGNORECASE)

    # Parse the matched dates using dateutil.parser.parse
    parsed_dates = []
    for match in matches:
        try:
            parsed_date = parse(match, fuzzy=True)
            parsed_dates.append(parsed_date.strftime('%Y-%m-%d'))  # Convert parsed date to YYYY-MM-DD format
        except ValueError:
            pass  # Ignore if parsing fails

    return parsed_dates

  # Function to convert relative dates to YYYY-MM-DD format
  def get_relative_date(self, sentence):
      def helper(relative_text):
        relative_text = relative_text.lower()
        today = datetime.now().date()

        if "tomorrow" in relative_text:
            return (today + timedelta(days=1)).isoformat()
        elif "day after tomorrow"in relative_text:
            return (today + timedelta(days=2)).isoformat()
        elif "yesterday" in relative_text:
            return (today - timedelta(days=1)).isoformat()
        elif "next" in relative_text:
            next_day = today + timedelta(days=7 - today.weekday())
            if "monday" in relative_text:
                next_monday = next_day + timedelta(days=(0 - next_day.weekday()) % 7)
                return next_monday.isoformat()
            elif "tuesday" in relative_text:
                return (next_day + timedelta(days=1)).isoformat()
            elif "wednesday" in relative_text:
                return (next_day + timedelta(days=2)).isoformat()
            elif "thursday" in relative_text:
                return (next_day + timedelta(days=3)).isoformat()
            elif "friday" in relative_text:
                return (next_day + timedelta(days=4)).isoformat()
            elif "saturday" in relative_text:
                return (next_day + timedelta(days=5)).isoformat()
            elif "sunday" in relative_text:
                return (next_day + timedelta(days=6)).isoformat()
        elif "last" in relative_text:
            last_day = today - timedelta(days=today.weekday() + 7)
            if "monday" in relative_text:
                return (last_day + timedelta(days=1)).isoformat()
            elif "tuesday" in relative_text:
                return (last_day + timedelta(days=2)).isoformat()
            elif "wednesday" in relative_text:
                return (last_day + timedelta(days=3)).isoformat()
            elif "thursday" in relative_text:
                return (last_day + timedelta(days=4)).isoformat()
            elif "friday" in relative_text:
                return (last_day + timedelta(days=5)).isoformat()
            elif "saturday" in relative_text:
                return (last_day + timedelta(days=6)).isoformat()
            elif "sunday" in relative_text:
                return (last_day + timedelta(days=7)).isoformat()
        elif "two weeks" in relative_text:
            return (today + timedelta(weeks=2)).isoformat()

        return None

      doc = self.nlp(sentence)
      relative_dates = []
      for ent in doc.ents:
          if ent.label_ == "DATE":
              relative_date = helper(ent.text)
              if relative_date:
                  relative_dates.append(relative_date)
      return relative_dates

  def get_range_dates(self, end_date, start_datee=datetime.now().date()):
    start_date = start_datee
    end_date_str = end_date
    doc = self.nlp(end_date_str)
    print(f"{doc.ents=}")

    # Extract the end date from the sentence
    for ent in doc.ents:
        if ent.label_ == "DATE":
            end_date_str = ent.text

    # Extract day and month from the end date string
    day, month_name = end_date_str.split()
    end_day = int(day)
    end_month = self.month_to_number(month_name)

    # Define the end date
    end_date = datetime(datetime.now().year, end_month, end_day).date()

    # Generate list of dates between start and end date
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.isoformat())
        current_date += timedelta(days=1)
    return date_list

sentence = input("Enter the sentence : ")

output = get_all_entities(sentence)


print(output)

