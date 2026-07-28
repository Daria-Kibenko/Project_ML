from natasha import (
    Segmenter, NewsEmbedding, NewsNERTagger,
    NewsMorphTagger, Doc
)

_segmenter = Segmenter()
_emb = NewsEmbedding()
_ner_tagger = NewsNERTagger(_emb)
_morph_tagger = NewsMorphTagger(_emb)

INCIDENT_KEYWORDS = {
    "разлив": "экологический инцидент",
    "выброс": "экологический инцидент",
    "загрязнение": "экологический инцидент",
    "авария": "промышленная авария",
    "пожар": "пожар",
    "взрыв": "промышленная авария",
    "забастовка": "трудовой конфликт",
    "увольнение": "трудовой конфликт",
    "штраф": "регуляторный риск",
    "проверка": "регуляторный риск",
}


def extract_entities(text: str) -> dict:
    doc = Doc(text)
    doc.segment(_segmenter)
    doc.tag_morph(_morph_tagger)
    doc.tag_ner(_ner_tagger)

    organizations = [span.text for span in doc.spans if span.type == "ORG"]
    locations = [span.text for span in doc.spans if span.type == "LOC"]
    persons = [span.text for span in doc.spans if span.type == "PER"]

    incident_type = "не определён"
    text_lower = text.lower()
    for keyword, label in INCIDENT_KEYWORDS.items():
        if keyword in text_lower:
            incident_type = label
            break

    return {
        "organizations": list(set(organizations)),
        "locations": list(set(locations)),
        "persons": list(set(persons)),
        "incident_type": incident_type,
    }


if __name__ == "__main__":
    sample = "На заводе НЛМК в Липецке произошёл разлив химикатов, пострадавших нет"
    print(extract_entities(sample))
