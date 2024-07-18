from dateutil.parser import parse
import spacy
def parse_date_time_from_text(text):
    doc = nlp(text)
    entities_dict = {'TIME': None, 'DATE': None}
    for ent in doc.ents:
        if ent.label_ in entities_dict:
            entities_dict[ent.label_] = ent.text
    return entities_dict


entities_dict = parse_date_time_from_text("I have meeting on 28th April")

print(entities_dict)