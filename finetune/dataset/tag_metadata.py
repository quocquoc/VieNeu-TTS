import re

# All English vocabulary words used in generation (sorted longest first to match multi-word phrases first)
EN_VOCAB = [
    # Multi-word phrases first
    "good morning", "good night", "good afternoon", "see you later",
    "thank you", "fire truck", "ice cream",
    # Animals
    "dog", "cat", "bird", "fish", "rabbit", "elephant", "tiger", "lion",
    "monkey", "bear", "horse", "pig", "duck", "frog", "butterfly", "bee",
    "snake", "turtle", "sheep", "goat", "deer", "fox", "wolf", "zebra",
    "giraffe", "penguin", "dolphin", "whale", "octopus", "parrot",
    "hamster", "squirrel", "kangaroo", "panda", "koala", "flamingo",
    "peacock", "owl", "eagle", "shark", "jellyfish", "starfish", "crab",
    "lobster", "shrimp", "snail", "caterpillar", "ladybug", "ant",
    "grasshopper", "dragonfly",
    # Food & drink
    "apple", "banana", "orange", "grape", "strawberry", "watermelon",
    "mango", "pineapple", "lemon", "peach", "bread", "rice", "milk",
    "egg", "cake", "cookie", "pizza", "soup", "noodle", "candy",
    "chocolate", "juice", "water", "tea", "coffee", "butter", "cheese",
    "yogurt", "sandwich", "salad", "sushi", "pasta", "cereal", "honey",
    "jam", "popcorn", "chips",
    # Colors
    "red", "blue", "green", "yellow", "pink", "purple", "white",
    "black", "brown", "gray", "gold", "silver", "beige",
    # Body parts
    "eye", "nose", "mouth", "ear", "hand", "foot", "head", "hair",
    "finger", "tooth", "arm", "leg", "shoulder", "knee", "elbow",
    "chin", "forehead", "cheek", "tongue", "lip", "neck", "back",
    "stomach", "chest",
    # School items
    "book", "pen", "pencil", "ruler", "eraser", "notebook", "scissors",
    "crayon", "desk", "chair", "backpack", "glue", "calculator",
    "compass", "stapler", "folder", "tape", "marker",
    # Weather
    "sun", "rain", "cloud", "wind", "snow", "rainbow", "storm", "fog",
    "thunder", "lightning", "hail", "frost",
    # Transport
    "car", "bus", "train", "plane", "bike", "boat", "truck", "taxi",
    "motorcycle", "helicopter", "ship", "rocket", "submarine",
    "ambulance", "tractor",
    # Clothes
    "shirt", "pants", "dress", "hat", "shoe", "sock", "jacket", "skirt",
    "coat", "glove", "scarf", "boot", "sandal", "tie",
    # Family
    "mother", "father", "sister", "brother", "grandmother", "grandfather",
    "baby", "friend", "aunt", "uncle", "cousin", "family",
    # Actions
    "run", "jump", "swim", "fly", "sing", "dance", "read", "write",
    "draw", "play", "sleep", "eat", "drink", "walk", "climb", "laugh",
    "cry", "think", "listen", "speak", "cook", "clean", "study",
    "teach", "help", "share", "love", "hug",
    # Numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "twenty", "hundred",
    # Fruits & vegetables
    "tomato", "carrot", "potato", "corn", "cabbage", "onion", "garlic",
    "mushroom", "pumpkin", "cucumber", "broccoli", "spinach", "pepper",
    "eggplant", "lettuce", "celery", "avocado", "coconut", "papaya",
    "guava",
    # Household
    "table", "bed", "sofa", "lamp", "clock", "mirror", "window",
    "floor", "wall", "roof", "door", "stairs", "kitchen", "bathroom",
    "bedroom", "refrigerator", "television", "telephone", "computer",
    "fan", "pillow", "blanket",
    # Adjectives
    "big", "small", "tall", "short", "fast", "slow", "hot", "cold",
    "happy", "sad", "beautiful", "strong", "smart", "kind", "brave",
    "funny", "dirty", "new", "old", "young", "loud", "quiet",
    "soft", "hard", "sweet", "sour", "salty",
    # Greetings
    "hello", "goodbye", "sorry", "please", "welcome",
    # Nature
    "tree", "flower", "grass", "leaf", "mountain", "river", "sea",
    "lake", "forest", "beach", "island", "sky", "star", "moon",
    "rock", "sand", "soil", "air", "fire",
    # Sports
    "football", "basketball", "swimming", "running", "tennis",
    "volleyball", "baseball", "cycling", "gymnastics", "boxing",
    "skating", "skiing",
    # Music
    "music", "song", "drum", "guitar", "piano", "violin", "flute",
    "trumpet", "harp", "saxophone",
    # Professions
    "doctor", "teacher", "police", "farmer", "chef", "pilot", "nurse",
    "engineer", "artist", "singer", "actor", "writer", "scientist",
    "soldier", "firefighter", "dentist",
    # Places
    "school", "hospital", "market", "park", "zoo", "library", "museum",
    "restaurant", "hotel", "airport", "station", "bank", "church",
    "temple", "stadium", "supermarket", "cinema", "factory",
    # Time
    "morning", "afternoon", "evening", "night", "today", "tomorrow",
    "yesterday", "week", "month", "year",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October",
    # Technology
    "phone", "laptop", "tablet", "camera", "robot", "internet",
    "game", "video", "photo", "message",
    # Shapes
    "circle", "square", "triangle", "rectangle", "heart", "diamond", "oval",
    # From original lines
    "chicken", "cow", "crocodile",
]

# Sort by length descending so multi-word phrases are matched before single words
EN_VOCAB_SORTED = sorted(EN_VOCAB, key=len, reverse=True)

PLACEHOLDER_RE = re.compile(r'\x00EN(\d+)\x00')


def tag_sentence(text):
    """Add [VI] and [EN_SLOWLY] tags to a mixed Vietnamese-English sentence."""
    placeholders = {}
    counter = [0]

    def replace_en(match):
        word = match.group(0)
        key = f"\x00EN{counter[0]}\x00"
        placeholders[key] = f"[EN_SLOWLY]{word}[/EN_SLOWLY]"
        counter[0] += 1
        return key

    # Replace all English vocab words with placeholders
    tagged = text
    for vocab_word in EN_VOCAB_SORTED:
        pattern = r'\b' + re.escape(vocab_word) + r'\b'
        tagged = re.sub(pattern, replace_en, tagged, flags=re.IGNORECASE)

    # Split by placeholders
    parts = re.split(r'(\x00EN\d+\x00)', tagged)

    result_parts = []
    for part in parts:
        if part in placeholders:
            result_parts.append(placeholders[part])
            continue

        stripped = part.strip()
        if not stripped:
            continue

        has_alpha = any(c.isalpha() for c in stripped)
        if not has_alpha:
            # Pure punctuation/spaces — keep outside tags
            result_parts.append(stripped)
        else:
            # Check for leading punctuation (e.g. ", quả táo..." → separate the comma)
            lead_match = re.match(r'^([^\w\u00C0-\u024F\u1EA0-\u1EFF]+)', stripped)
            if lead_match:
                lead_punct = lead_match.group(1)
                vi_content = stripped[len(lead_punct):]
                result_parts.append(lead_punct)
                if vi_content.strip():
                    result_parts.append(f"[VI]{vi_content.strip()}[/VI]")
            else:
                result_parts.append(f"[VI]{stripped}[/VI]")

    # Join with spaces, then fix space-before-punctuation
    output = ' '.join(p for p in result_parts if p)
    output = re.sub(r' +([.,!?;:])', r'\1', output)
    output = re.sub(r' {2,}', ' ', output)
    return output.strip()


def process_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    tagged_count = 0
    skipped_count = 0

    for line in lines:
        line = line.rstrip('\n')
        if not line:
            out_lines.append('')
            continue

        parts = line.split('|', 1)
        if len(parts) != 2:
            out_lines.append(line)
            continue

        filename, text = parts[0], parts[1]

        # Skip lines already tagged
        if '[VI]' in text or '[EN_SLOWLY]' in text:
            out_lines.append(line)
            skipped_count += 1
            continue

        tagged_text = tag_sentence(text)
        out_lines.append(f"{filename}|{tagged_text}")
        tagged_count += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
        if out_lines:
            f.write('\n')

    print(f"Processed: {tagged_count} tagged, {skipped_count} already tagged (skipped)")
    return tagged_count


if __name__ == '__main__':
    import sys

    input_path = "/Users/quoctruong/Documents/My_projects/TTSModel_VieNeuTTS/FineTune-VieNeuTTS/VieNeu-TTS/finetune/dataset/metadata_2 copy.csv"
    output_path = input_path  # overwrite in place

    # Quick preview on first 5 untagged lines
    print("=== PREVIEW (first 5 untagged lines) ===")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    count = 0
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        parts = line.split('|', 1)
        if len(parts) != 2:
            continue
        filename, text = parts[0], parts[1]
        if '[VI]' in text or '[EN_SLOWLY]' in text:
            continue
        tagged = tag_sentence(text)
        print(f"  IN : {filename}|{text}")
        print(f"  OUT: {filename}|{tagged}")
        print()
        count += 1
        if count >= 5:
            break

    confirm = input("Proceed with tagging all lines? [y/N]: ").strip().lower()
    if confirm == 'y':
        process_file(input_path, output_path)
        print("Done.")
    else:
        print("Aborted.")
