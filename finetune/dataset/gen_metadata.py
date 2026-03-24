import random

random.seed(42)

vocab = [
    # Animals
    ("dog", "con chó"), ("cat", "con mèo"), ("bird", "con chim"), ("fish", "con cá"),
    ("rabbit", "con thỏ"), ("elephant", "con voi"), ("tiger", "con hổ"), ("lion", "con sư tử"),
    ("monkey", "con khỉ"), ("bear", "con gấu"), ("horse", "con ngựa"), ("pig", "con lợn"),
    ("duck", "con vịt"), ("frog", "con ếch"), ("butterfly", "con bướm"), ("bee", "con ong"),
    ("snake", "con rắn"), ("turtle", "con rùa"), ("sheep", "con cừu"), ("goat", "con dê"),
    ("deer", "con hươu"), ("fox", "con cáo"), ("wolf", "con sói"), ("zebra", "con ngựa vằn"),
    ("giraffe", "con hươu cao cổ"), ("penguin", "chim cánh cụt"), ("dolphin", "con cá heo"),
    ("whale", "con cá voi"), ("octopus", "con bạch tuộc"), ("parrot", "con vẹt"),
    ("hamster", "con chuột hamster"), ("squirrel", "con sóc"), ("kangaroo", "con chuột túi"),
    ("panda", "con gấu trúc"), ("koala", "con gấu túi"), ("flamingo", "chim hồng hạc"),
    ("peacock", "con công"), ("owl", "con cú"), ("eagle", "con đại bàng"), ("shark", "con cá mập"),
    ("jellyfish", "con sứa"), ("starfish", "con sao biển"), ("crab", "con cua"),
    ("lobster", "con tôm hùm"), ("shrimp", "con tôm"), ("snail", "con ốc sên"),
    ("caterpillar", "con sâu bướm"), ("ladybug", "con bọ rùa"), ("ant", "con kiến"),
    ("grasshopper", "con châu chấu"), ("dragonfly", "con chuồn chuồn"),
    # Food
    ("apple", "quả táo"), ("banana", "quả chuối"), ("orange", "quả cam"), ("grape", "quả nho"),
    ("strawberry", "quả dâu tây"), ("watermelon", "quả dưa hấu"), ("mango", "quả xoài"),
    ("pineapple", "quả dứa"), ("lemon", "quả chanh"), ("peach", "quả đào"),
    ("bread", "bánh mì"), ("rice", "cơm"), ("milk", "sữa"), ("egg", "quả trứng"),
    ("cake", "bánh kem"), ("cookie", "bánh quy"), ("pizza", "bánh pizza"),
    ("soup", "canh"), ("noodle", "mì"), ("candy", "kẹo"), ("chocolate", "sô cô la"),
    ("juice", "nước ép"), ("water", "nước"), ("tea", "trà"), ("coffee", "cà phê"),
    ("butter", "bơ"), ("cheese", "phô mai"), ("yogurt", "sữa chua"), ("ice cream", "kem"),
    ("sandwich", "bánh sandwich"), ("salad", "rau trộn"), ("sushi", "sushi"),
    ("pasta", "mì pasta"), ("cereal", "ngũ cốc"), ("honey", "mật ong"),
    ("jam", "mứt"), ("popcorn", "bỏng ngô"), ("chips", "khoai tây chiên"),
    # Colors
    ("red", "màu đỏ"), ("blue", "màu xanh dương"), ("green", "màu xanh lá"),
    ("yellow", "màu vàng"), ("pink", "màu hồng"), ("purple", "màu tím"),
    ("white", "màu trắng"), ("black", "màu đen"), ("brown", "màu nâu"),
    ("gray", "màu xám"), ("orange", "màu cam"), ("gold", "màu vàng kim"),
    ("silver", "màu bạc"), ("beige", "màu be"),
    # Body parts
    ("eye", "mắt"), ("nose", "mũi"), ("mouth", "miệng"), ("ear", "tai"),
    ("hand", "bàn tay"), ("foot", "bàn chân"), ("head", "cái đầu"), ("hair", "tóc"),
    ("finger", "ngón tay"), ("tooth", "cái răng"), ("arm", "cánh tay"), ("leg", "chân"),
    ("shoulder", "vai"), ("knee", "đầu gối"), ("elbow", "khuỷu tay"), ("chin", "cằm"),
    ("forehead", "trán"), ("cheek", "má"), ("tongue", "lưỡi"), ("lip", "môi"),
    ("neck", "cổ"), ("back", "lưng"), ("stomach", "bụng"), ("chest", "ngực"),
    # School items
    ("book", "cuốn sách"), ("pen", "cây bút"), ("pencil", "bút chì"),
    ("ruler", "cái thước"), ("eraser", "cục tẩy"), ("notebook", "quyển vở"),
    ("scissors", "cái kéo"), ("crayon", "bút màu"), ("desk", "cái bàn"),
    ("chair", "cái ghế"), ("backpack", "ba lô"), ("glue", "keo dán"),
    ("calculator", "máy tính"), ("compass", "compa"), ("stapler", "cái dập ghim"),
    ("folder", "bìa hồ sơ"), ("tape", "băng dính"), ("marker", "bút lông"),
    # Weather
    ("sun", "mặt trời"), ("rain", "mưa"), ("cloud", "đám mây"), ("wind", "gió"),
    ("snow", "tuyết"), ("rainbow", "cầu vồng"), ("storm", "bão"), ("fog", "sương mù"),
    ("thunder", "sấm"), ("lightning", "sét"), ("hail", "mưa đá"), ("frost", "sương giá"),
    # Transport
    ("car", "xe ô tô"), ("bus", "xe buýt"), ("train", "tàu hỏa"), ("plane", "máy bay"),
    ("bike", "xe đạp"), ("boat", "con thuyền"), ("truck", "xe tải"), ("taxi", "xe taxi"),
    ("motorcycle", "xe máy"), ("helicopter", "máy bay trực thăng"), ("ship", "con tàu"),
    ("rocket", "tên lửa"), ("submarine", "tàu ngầm"), ("ambulance", "xe cấp cứu"),
    ("fire truck", "xe cứu hỏa"), ("tractor", "máy kéo"),
    # Clothes
    ("shirt", "áo sơ mi"), ("pants", "quần dài"), ("dress", "váy"), ("hat", "cái mũ"),
    ("shoe", "đôi giày"), ("sock", "đôi tất"), ("jacket", "áo khoác"), ("skirt", "váy ngắn"),
    ("coat", "áo choàng"), ("glove", "găng tay"), ("scarf", "khăn quàng"),
    ("boot", "ủng"), ("sandal", "dép xăng đan"), ("tie", "cà vạt"),
    # Family
    ("mother", "mẹ"), ("father", "bố"), ("sister", "chị gái"), ("brother", "anh trai"),
    ("grandmother", "bà"), ("grandfather", "ông"), ("baby", "em bé"), ("friend", "bạn bè"),
    ("aunt", "cô"), ("uncle", "chú"), ("cousin", "anh chị em họ"), ("family", "gia đình"),
    # Actions
    ("run", "chạy"), ("jump", "nhảy"), ("swim", "bơi"), ("fly", "bay"),
    ("sing", "hát"), ("dance", "nhảy múa"), ("read", "đọc"), ("write", "viết"),
    ("draw", "vẽ"), ("play", "chơi"), ("sleep", "ngủ"), ("eat", "ăn"),
    ("drink", "uống"), ("walk", "đi bộ"), ("climb", "leo trèo"), ("laugh", "cười"),
    ("cry", "khóc"), ("think", "suy nghĩ"), ("listen", "lắng nghe"), ("speak", "nói"),
    ("cook", "nấu ăn"), ("clean", "dọn dẹp"), ("study", "học bài"), ("teach", "dạy học"),
    ("help", "giúp đỡ"), ("share", "chia sẻ"), ("love", "yêu thương"), ("hug", "ôm"),
    # Numbers
    ("one", "số một"), ("two", "số hai"), ("three", "số ba"), ("four", "số bốn"),
    ("five", "số năm"), ("six", "số sáu"), ("seven", "số bảy"), ("eight", "số tám"),
    ("nine", "số chín"), ("ten", "số mười"), ("eleven", "số mười một"),
    ("twelve", "số mười hai"), ("twenty", "số hai mươi"), ("hundred", "một trăm"),
    # Fruits/vegetables
    ("tomato", "quả cà chua"), ("carrot", "củ cà rốt"), ("potato", "củ khoai tây"),
    ("corn", "ngô"), ("cabbage", "bắp cải"), ("onion", "củ hành"), ("garlic", "tỏi"),
    ("mushroom", "nấm"), ("pumpkin", "bí ngô"), ("cucumber", "dưa chuột"),
    ("broccoli", "súp lơ xanh"), ("spinach", "rau bina"), ("pepper", "ớt chuông"),
    ("eggplant", "cà tím"), ("lettuce", "xà lách"), ("celery", "cần tây"),
    ("avocado", "quả bơ"), ("coconut", "quả dừa"), ("papaya", "đu đủ"), ("guava", "ổi"),
    # Household
    ("table", "cái bàn"), ("bed", "cái giường"), ("sofa", "ghế sofa"), ("lamp", "đèn"),
    ("clock", "đồng hồ"), ("mirror", "gương"), ("window", "cửa sổ"), ("floor", "sàn nhà"),
    ("wall", "bức tường"), ("roof", "mái nhà"), ("door", "cái cửa"), ("stairs", "cầu thang"),
    ("kitchen", "nhà bếp"), ("bathroom", "phòng tắm"), ("bedroom", "phòng ngủ"),
    ("refrigerator", "tủ lạnh"), ("television", "ti vi"), ("telephone", "điện thoại"),
    ("computer", "máy tính"), ("fan", "quạt"), ("pillow", "cái gối"), ("blanket", "chăn"),
    # Adjectives
    ("big", "to lớn"), ("small", "nhỏ bé"), ("tall", "cao"), ("short", "thấp"),
    ("fast", "nhanh"), ("slow", "chậm"), ("hot", "nóng"), ("cold", "lạnh"),
    ("happy", "vui vẻ"), ("sad", "buồn"), ("beautiful", "xinh đẹp"), ("strong", "mạnh mẽ"),
    ("smart", "thông minh"), ("kind", "tốt bụng"), ("brave", "dũng cảm"), ("funny", "hài hước"),
    ("clean", "sạch sẽ"), ("dirty", "bẩn"), ("new", "mới"), ("old", "cũ"),
    ("young", "trẻ"), ("loud", "ồn ào"), ("quiet", "yên tĩnh"), ("soft", "mềm"),
    ("hard", "cứng"), ("sweet", "ngọt"), ("sour", "chua"), ("salty", "mặn"),
    # Greetings
    ("hello", "xin chào"), ("goodbye", "tạm biệt"), ("thank you", "cảm ơn"),
    ("sorry", "xin lỗi"), ("please", "làm ơn"), ("welcome", "chào mừng"),
    ("good morning", "chào buổi sáng"), ("good night", "chúc ngủ ngon"),
    ("good afternoon", "chào buổi chiều"), ("see you later", "hẹn gặp lại"),
    # Nature
    ("tree", "cái cây"), ("flower", "bông hoa"), ("grass", "cỏ"), ("leaf", "chiếc lá"),
    ("mountain", "ngọn núi"), ("river", "dòng sông"), ("sea", "biển"), ("lake", "cái hồ"),
    ("forest", "khu rừng"), ("beach", "bãi biển"), ("island", "hòn đảo"), ("sky", "bầu trời"),
    ("star", "ngôi sao"), ("moon", "mặt trăng"), ("rock", "tảng đá"), ("sand", "cát"),
    ("soil", "đất"), ("air", "không khí"), ("fire", "lửa"), ("water", "nước"),
    # Sports
    ("football", "bóng đá"), ("basketball", "bóng rổ"), ("swimming", "bơi lội"),
    ("running", "chạy bộ"), ("tennis", "quần vợt"), ("volleyball", "bóng chuyền"),
    ("baseball", "bóng chày"), ("cycling", "đạp xe"), ("gymnastics", "thể dục dụng cụ"),
    ("boxing", "quyền anh"), ("skating", "trượt băng"), ("skiing", "trượt tuyết"),
    # Music
    ("music", "âm nhạc"), ("song", "bài hát"), ("drum", "trống"), ("guitar", "đàn guitar"),
    ("piano", "đàn piano"), ("violin", "đàn violin"), ("flute", "sáo"),
    ("trumpet", "kèn trumpet"), ("harp", "đàn hạc"), ("saxophone", "kèn saxophone"),
    # Professions
    ("doctor", "bác sĩ"), ("teacher", "giáo viên"), ("police", "cảnh sát"),
    ("farmer", "nông dân"), ("chef", "đầu bếp"), ("pilot", "phi công"),
    ("nurse", "y tá"), ("engineer", "kỹ sư"), ("artist", "họa sĩ"), ("singer", "ca sĩ"),
    ("actor", "diễn viên"), ("writer", "nhà văn"), ("scientist", "nhà khoa học"),
    ("soldier", "chiến sĩ"), ("firefighter", "lính cứu hỏa"), ("dentist", "nha sĩ"),
    # Places
    ("school", "trường học"), ("hospital", "bệnh viện"), ("market", "chợ"),
    ("park", "công viên"), ("zoo", "vườn thú"), ("library", "thư viện"),
    ("museum", "bảo tàng"), ("restaurant", "nhà hàng"), ("hotel", "khách sạn"),
    ("airport", "sân bay"), ("station", "ga tàu"), ("bank", "ngân hàng"),
    ("church", "nhà thờ"), ("temple", "ngôi đền"), ("stadium", "sân vận động"),
    ("supermarket", "siêu thị"), ("cinema", "rạp chiếu phim"), ("factory", "nhà máy"),
    # Time
    ("morning", "buổi sáng"), ("afternoon", "buổi chiều"), ("evening", "buổi tối"),
    ("night", "ban đêm"), ("today", "hôm nay"), ("tomorrow", "ngày mai"),
    ("yesterday", "hôm qua"), ("week", "tuần"), ("month", "tháng"), ("year", "năm"),
    ("Monday", "thứ Hai"), ("Tuesday", "thứ Ba"), ("Wednesday", "thứ Tư"),
    ("Thursday", "thứ Năm"), ("Friday", "thứ Sáu"), ("Saturday", "thứ Bảy"),
    ("Sunday", "Chủ nhật"), ("January", "tháng Một"), ("February", "tháng Hai"),
    ("March", "tháng Ba"), ("April", "tháng Tư"), ("May", "tháng Năm"),
    ("June", "tháng Sáu"), ("July", "tháng Bảy"), ("August", "tháng Tám"),
    ("September", "tháng Chín"), ("October", "tháng Mười"),
    # Technology
    ("phone", "điện thoại"), ("laptop", "máy tính xách tay"), ("tablet", "máy tính bảng"),
    ("camera", "máy ảnh"), ("robot", "người máy"), ("internet", "mạng internet"),
    ("game", "trò chơi"), ("video", "video"), ("photo", "ảnh"), ("message", "tin nhắn"),
    # Shapes
    ("circle", "hình tròn"), ("square", "hình vuông"), ("triangle", "hình tam giác"),
    ("rectangle", "hình chữ nhật"), ("star", "hình ngôi sao"), ("heart", "hình trái tim"),
    ("diamond", "hình thoi"), ("oval", "hình bầu dục"),
]

short_templates = [
    "Từ {word} trong tiếng Anh có nghĩa là {vn}, cùng đọc nào, {word}.",
    "{word} nghĩa là {vn} trong tiếng Việt, các con đọc to nhé, {word}.",
    "Hôm nay học từ {word}, nghĩa là {vn}, các con đọc to nào, {word}.",
    "{word} là từ tiếng Anh chỉ {vn}, nào cùng đọc to theo cô, {word}.",
    "Từ {word} có nghĩa là {vn}, rất dễ đọc phải không, {word}.",
    "Các con ơi, {word} nghĩa là {vn}, cùng đọc to nào, {word}.",
    "{word} là từ tiếng Anh có nghĩa là {vn}, đọc to nào.",
    "Học từ mới nhé, {word} nghĩa là {vn}, nào cùng đọc, {word}.",
    "Từ {word} dịch sang tiếng Việt là {vn}, đọc to theo cô, {word}.",
    "Cô dạy từ mới, {word} có nghĩa là {vn}, nào đọc to nào, {word}.",
    "Chú ý nhé, {word} nghĩa là {vn}, các con đọc to theo cô, {word}.",
    "{word}, từ này có nghĩa là {vn} nhé, cùng đọc to nào, {word}.",
    "Từ tiếng Anh {word} nghĩa là {vn}, hãy đọc to nào, {word}.",
    "Cùng học từ {word} nhé, {word} có nghĩa là {vn} trong tiếng Việt.",
    "Đây là từ {word}, nghĩa là {vn}, các con nhớ nhé, {word}.",
    "Hãy đọc từ {word} thật to nhé, {word} có nghĩa là {vn}.",
    "{word} trong tiếng Việt là {vn}, nào đọc to theo cô, {word}.",
    "Từ mới hôm nay là {word}, nghĩa là {vn}, đọc to nhé, {word}.",
    "Lớp mình học từ {word} nhé, {word} có nghĩa là {vn} đó.",
    "Các bạn ơi, {word} là {vn} trong tiếng Việt, đọc to nào, {word}.",
]

long_templates = [
    "Từ {word} trong tiếng Anh có nghĩa là {vn} trong tiếng Việt, đây là từ rất hay, cùng đọc to theo cô nhé, {word}.",
    "{word} có nghĩa là {vn}, đây là từ rất thú vị và quan trọng, các con hãy đọc to theo cô nào, {word}.",
    "Hôm nay chúng ta học từ {word} nhé, {word} có nghĩa là {vn} trong tiếng Việt, các con đọc to nào.",
    "Đây là từ {word}, nghĩa là {vn} trong tiếng Việt, rất thú vị phải không, cùng đọc to theo cô nhé, {word}.",
    "Các con ơi, từ {word} trong tiếng Anh nghĩa là {vn}, đây là từ rất dễ nhớ, cùng đọc to nhé, {word}.",
    "Chúng ta cùng học từ mới nhé, {word} có nghĩa là {vn} trong tiếng Việt, nào cùng đọc to theo cô, {word}.",
    "Từ tiếng Anh {word} có nghĩa là {vn}, đây là từ rất hay và thú vị, các con đọc to theo cô nào, {word}.",
    "Hãy cùng nhau học từ {word} nhé, {word} trong tiếng Việt có nghĩa là {vn}, cùng đọc thật to nào.",
    "Cô dạy các con từ mới hôm nay là {word}, {word} có nghĩa là {vn}, hãy đọc thật to theo cô nhé, {word}.",
    "Các con hãy chú ý nhé, từ {word} trong tiếng Anh có nghĩa là {vn} trong tiếng Việt, nào đọc to nào, {word}.",
    "Học tiếng Anh thật vui phải không, hôm nay học từ {word} nghĩa là {vn}, cùng đọc to theo cô nhé, {word}.",
    "Từ {word} rất hay và thú vị, {word} có nghĩa là {vn} trong tiếng Việt, các con đọc to theo cô nhé.",
    "Cùng nhau học tiếng Anh nào, từ {word} có nghĩa là {vn}, đây là từ rất dễ nhớ, hãy đọc to nhé, {word}.",
    "Hôm nay cô dạy từ {word}, trong tiếng Việt {word} có nghĩa là {vn}, các con đọc to theo cô nào, {word}.",
    "Các con có biết từ {word} nghĩa là gì không, {word} có nghĩa là {vn} đó, cùng đọc to nào, {word}.",
    "Từ {word} trong tiếng Anh rất hay, nó có nghĩa là {vn} trong tiếng Việt, nào cùng đọc to nhé, {word}.",
    "Chúng ta hãy cùng nhau đọc từ {word} thật to nhé, {word} có nghĩa là {vn} trong tiếng Việt đó.",
    "Học tiếng Anh rất thú vị, từ {word} có nghĩa là {vn}, hãy cùng nhau đọc to theo cô nhé, {word}.",
    "Cô mời các con đọc to từ {word}, {word} trong tiếng Việt có nghĩa là {vn}, nào đọc to nào.",
    "Hôm nay chúng ta học từ mới {word}, {word} có nghĩa là {vn} nhé, cùng đọc to theo cô nào, {word}.",
]

def make_entry(word, vn, template, idx):
    text = template.format(word=word, vn=vn)
    return f"mix_{idx}.wav|{text}"

entries = []
idx = 3001
total = 3000
half = total // 2

# Generate short entries (60-90 chars)
short_count = 0
long_count = 0
vocab_cycle = list(vocab)
random.shuffle(vocab_cycle)

all_pairs = []
while len(all_pairs) < total:
    shuffled = list(vocab)
    random.shuffle(shuffled)
    all_pairs.extend(shuffled)

all_pairs = all_pairs[:total]

for i, (word, vn) in enumerate(all_pairs):
    if i < half:
        # target 60-90 chars
        candidates = []
        for t in short_templates:
            text = t.format(word=word, vn=vn)
            if 60 <= len(text) <= 90:
                candidates.append(text)
        if candidates:
            text = random.choice(candidates)
        else:
            # try long templates too
            for t in long_templates:
                text = t.format(word=word, vn=vn)
                if 60 <= len(text) <= 90:
                    candidates.append(text)
            if candidates:
                text = random.choice(candidates)
            else:
                # pick shortest short template
                texts = [t.format(word=word, vn=vn) for t in short_templates]
                text = min(texts, key=lambda x: abs(len(x) - 75))
    else:
        # target 90-120 chars
        candidates = []
        for t in long_templates:
            text = t.format(word=word, vn=vn)
            if 90 <= len(text) <= 120:
                candidates.append(text)
        if candidates:
            text = random.choice(candidates)
        else:
            for t in short_templates:
                text = t.format(word=word, vn=vn)
                if 90 <= len(text) <= 120:
                    candidates.append(text)
            if candidates:
                text = random.choice(candidates)
            else:
                texts = [t.format(word=word, vn=vn) for t in long_templates]
                text = min(texts, key=lambda x: abs(len(x) - 105))
    entries.append(f"mix_{idx}.wav|{text}")
    idx += 1

output_path = "/Users/quoctruong/Documents/My_projects/TTSModel_VieNeuTTS/FineTune-VieNeuTTS/VieNeu-TTS/finetune/dataset/metadata_2.csv"
with open(output_path, "a", encoding="utf-8") as f:
    for entry in entries:
        f.write(entry + "\n")

print(f"Added {len(entries)} entries")

# Stats
lengths = [len(e.split("|", 1)[1]) for e in entries]
short_ok = sum(1 for l in lengths if 60 <= l <= 90)
long_ok = sum(1 for l in lengths if 90 < l <= 120)
out_range = sum(1 for l in lengths if l < 60 or l > 120)
print(f"60-90 chars: {short_ok}")
print(f"90-120 chars: {long_ok}")
print(f"Out of range: {out_range}")
print(f"Min: {min(lengths)}, Max: {max(lengths)}")
