"""
Hindi spelling and character counting tasks.

Adapted from nanochat's SpellingBee for Devanagari.

Two tasks:
1. HindiSpelling: Spell Hindi words character by character
   "शब्द लिखें: भारत" → "भ,ा,र,त"

2. HindiLetterCount: Count occurrences of a character in a Hindi word
   "शब्द 'आकाश' में 'आ' कितनी बार है?" → counting + answer

These help the model learn Devanagari character-level understanding,
which is hard for tokenizer-based models.
"""

import random
from tasks.common import Task

# Common Hindi words for spelling tasks
# Mix of everyday words, names, places, abstract concepts
HINDI_WORDS = [
    "भारत", "हिंदी", "दिल्ली", "मुंबई", "कोलकाता", "चेन्नई", "बेंगलुरु",
    "राजस्थान", "महाराष्ट्र", "गुजरात", "उत्तर", "प्रदेश", "बिहार", "पंजाब",
    "नमस्ते", "धन्यवाद", "शुभकामनाएं", "स्वागत", "विदाई", "अभिनंदन",
    "परिवार", "माता", "पिता", "भाई", "बहन", "दादा", "दादी", "नाना", "नानी",
    "विद्यालय", "विश्वविद्यालय", "पुस्तकालय", "अस्पताल", "दवाखाना",
    "सरकार", "प्रधानमंत्री", "राष्ट्रपति", "संसद", "लोकतंत्र", "संविधान",
    "अर्थव्यवस्था", "व्यापार", "उद्योग", "कृषि", "प्रौद्योगिकी",
    "शिक्षा", "ज्ञान", "विज्ञान", "गणित", "इतिहास", "भूगोल", "साहित्य",
    "संगीत", "नृत्य", "चित्रकला", "खेल", "क्रिकेट", "हॉकी", "फुटबॉल",
    "पानी", "खाना", "रोटी", "चावल", "दाल", "सब्जी", "फल", "दूध", "चाय",
    "सूरज", "चांद", "तारे", "बादल", "बारिश", "हवा", "आकाश", "पृथ्वी",
    "समय", "घड़ी", "कल", "आज", "कल", "सुबह", "शाम", "रात", "दोपहर",
    "स्वास्थ्य", "दवाई", "चिकित्सक", "उपचार", "बीमारी", "स्वस्थ",
    "कंप्यूटर", "इंटरनेट", "मोबाइल", "सॉफ्टवेयर", "हार्डवेयर",
    "पुस्तक", "कहानी", "कविता", "उपन्यास", "लेखक", "पत्रकार",
    "रेलगाड़ी", "हवाईजहाज", "बस", "कार", "साइकिल", "मोटरसाइकिल",
    "अध्यापक", "विद्यार्थी", "परीक्षा", "अंक", "उत्तीर्ण", "पाठ्यक्रम",
    "महात्मा", "गांधी", "स्वतंत्रता", "आजादी", "क्रांति", "देशभक्ति",
    "अभिनेता", "अभिनेत्री", "निर्देशक", "फिल्म", "सिनेमा", "नाटक",
    "मंदिर", "मस्जिद", "गुरुद्वारा", "चर्च", "धर्म", "आस्था", "प्रार्थना",
    "नदी", "पहाड़", "समुद्र", "जंगल", "रेगिस्तान", "झील", "झरना",
    "त्योहार", "दीवाली", "होली", "ईद", "क्रिसमस", "बैसाखी", "पोंगल",
    "बाजार", "दुकान", "खरीदारी", "कीमत", "रुपया", "पैसा", "बचत",
    "मित्र", "दोस्त", "प्रेम", "विश्वास", "सम्मान", "सहायता", "दया",
    "आसमान", "धरती", "अग्नि", "जल", "वायु", "प्रकृति", "पर्यावरण",
    "संस्कृति", "परंपरा", "विरासत", "कला", "शिल्प", "हस्तकला",
    "न्यायालय", "न्यायाधीश", "वकील", "कानून", "अधिकार", "कर्तव्य",
    "सेना", "सैनिक", "रक्षा", "सुरक्षा", "शांति", "युद्ध",
    "अनुसंधान", "आविष्कार", "खोज", "प्रयोग", "परिणाम", "सिद्धांत",
]

# Hindi user message templates for letter counting
HINDI_TEMPLATES = [
    "शब्द '{word}' में '{letter}' कितनी बार आता है?",
    "'{word}' में '{letter}' की गिनती बताओ",
    "'{word}' में कितने '{letter}' हैं?",
    "बताओ '{word}' में '{letter}' कितनी बार है",
    "'{word}' शब्द में '{letter}' की संख्या क्या है?",
    "गिनो '{word}' में '{letter}' कितने बार आता है",
    "'{word}' में '{letter}' अक्षर कितनी बार दिखता है?",
    "'{word}' में '{letter}' कुल कितने हैं?",
    "'{word}' शब्द में '{letter}' अक्षर गिनें",
    "बताइए '{word}' में '{letter}' की कुल संख्या",
]


class HindiSpelling(Task):
    """Spell Hindi words character by character."""

    def __init__(self, size=10000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.size = size
        self.split = split
        self.words = HINDI_WORDS

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else 10_000_000 + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        chars = list(word)
        spelled = ",".join(chars)

        messages = [
            {"role": "user", "content": f"शब्द लिखें: {word}"},
            {"role": "assistant", "content": f"{word}:{spelled}"},
        ]
        return {"messages": messages}


class HindiLetterCount(Task):
    """Count occurrences of a character in a Hindi word."""

    def __init__(self, size=10000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"]
        self.size = size
        self.split = split
        self.words = HINDI_WORDS

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else 10_000_000 + index
        rng = random.Random(seed)

        word = rng.choice(self.words)
        chars = list(word)

        # 80% pick a char from the word, 20% pick a random Hindi char
        if rng.random() < 0.8:
            letter = rng.choice(chars)
        else:
            all_chars = set()
            for w in self.words:
                all_chars.update(list(w))
            letter = rng.choice(list(all_chars))

        count = chars.count(letter)

        template = rng.choice(HINDI_TEMPLATES)
        user_msg = template.format(word=word, letter=letter)

        # Build counting response
        spelled = ",".join(chars)
        response = f"शब्द '{word}' के अक्षर गिनते हैं:\n{word}:{spelled}\n\n"

        running_count = 0
        for i, char in enumerate(chars, 1):
            if char == letter:
                running_count += 1
                response += f"{i}:{char} ← मिला! गिनती={running_count}\n"
            else:
                response += f"{i}:{char}\n"

        response += f"\nकुल '{letter}' की संख्या: {count}\n\n#### {count}"

        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """Check if the count after #### matches."""
        import re
        ref_match = re.search(r"#### (\d+)", conversation['messages'][-1]['content'])
        pred_match = re.search(r"#### (\d+)", assistant_response)
        if ref_match and pred_match:
            return int(ref_match.group(1) == pred_match.group(1))
        return 0


if __name__ == "__main__":
    print("=== HindiSpelling ===")
    task = HindiSpelling(size=5)
    for i in range(5):
        ex = task[i]
        print(f"  {ex['messages'][0]['content']} → {ex['messages'][1]['content']}")

    print("\n=== HindiLetterCount ===")
    task = HindiLetterCount(size=3)
    for i in range(3):
        ex = task[i]
        print("=" * 60)
        print(f"Q: {ex['messages'][0]['content']}")
        print(f"A: {ex['messages'][1]['content'][:200]}...")
