import logging
import os.path
import re
from typing import Set, List

import numpy as np
from nltk.corpus.reader import Synset
from sentence_transformers.util import cos_sim
from vwsd.common import InputSample
import fasttext

ENGLISH_STOPWORDS = {
    "a", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards", "again",
    "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway",
    "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't", "around", "as", "a's",
    "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides",
    "best", "better", "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant", "can't",
    "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com", "come", "comes",
    "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding",
    "could", "couldn't", "course", "c's", "currently", "definitely", "described", "despite", "did", "didn't",
    "different", "do", "does", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "during", "each",
    "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even",
    "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "far",
    "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four",
    "from", "further", "furthermore", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got",
    "gotten", "greetings", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he",
    "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "here's", "hereupon",
    "hers", "herself", "he's", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however",
    "how's", "i", "i'd", "ie", "if", "ignored", "i'll", "i'm", "immediate", "in", "inasmuch", "inc", "indeed",
    "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself", "i've",
    "just", "keep", "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly",
    "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd",
    "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most",
    "mostly", "much", "must", "mustn't", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
    "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone",
    "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often", "oh", "ok",
    "okay", "old", "on", "once", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours",
    "ourselves", "out", "outside", "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed",
    "please", "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather", "rd", "re",
    "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said",
    "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming",
    "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "since", "six", "so", "some", "somebody",
    "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified",
    "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "take", "taken", "tell", "tends", "th",
    "than", "thank", "thanks", "thanx", "that", "thats", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "theres", "there's", "thereupon",
    "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
    "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
    "towards", "tried", "tries", "truly", "try", "trying", "t's", "twice", "two", "un", "under", "unfortunately",
    "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually",
    "value", "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome",
    "well", "we'll", "went", "were", "we're", "weren't", "we've", "what", "whatever", "what's", "when", "whence",
    "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "where's", "whereupon", "wherever",
    "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "who's", "whose", "why", "why's",
    "will", "willing", "wish", "with", "within", "without", "wonder", "won't", "would", "wouldn't", "yes", "yet",
    "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've"
}

ITALIAN_STOPWORDS = {
    "a", "abbastanza", "abbia", "abbiamo", "abbiano", "abbiate", "accidenti", "ad", "adesso", "affinché", "agl", "agli",
    "ahime", "ahimè", "ai", "al", "alcuna", "alcuni", "alcuno", "all", "alla", "alle", "allo", "allora", "altre",
    "altri", "altrimenti", "altro", "altrove", "altrui", "anche", "ancora", "anni", "anno", "ansa", "anticipo",
    "assai", "attesa", "attraverso", "avanti", "avemmo", "avendo", "avente", "aver", "avere", "averlo", "avesse",
    "avessero", "avessi", "avessimo", "aveste", "avesti", "avete", "aveva", "avevamo", "avevano", "avevate", "avevi",
    "avevo", "avrai", "avranno", "avrebbe", "avrebbero", "avrei", "avremmo", "avremo", "avreste", "avresti", "avrete",
    "avrà", "avrò", "avuta", "avute", "avuti", "avuto", "basta", "ben", "bene", "benissimo", "brava", "bravo", "buono",
    "c", "caso", "cento", "certa", "certe", "certi", "certo", "che", "chi", "chicchessia", "chiunque", "ci",
    "ciascuna", "ciascuno", "cima", "cinque", "cio", "cioe", "cioè", "circa", "citta", "città", "ciò", "co",
    "codesta", "codesti", "codesto", "cogli", "coi", "col", "colei", "coll", "coloro", "colui", "come", "cominci",
    "comprare", "comunque", "con", "concernente", "conclusione", "consecutivi", "consecutivo", "consiglio", "contro",
    "cortesia", "cos", "cosa", "cosi", "così", "cui", "d", "da", "dagl", "dagli", "dai", "dal", "dall", "dalla",
    "dalle", "dallo", "dappertutto", "davanti", "degl", "degli", "dei", "del", "dell", "della", "delle", "dello",
    "dentro", "detto", "deve", "devo", "di", "dice", "dietro", "dire", "dirimpetto", "diventa", "diventare",
    "diventato", "dopo", "doppio", "dov", "dove", "dovra", "dovrà", "dovunque", "due", "dunque", "durante", "e",
    "ebbe", "ebbero", "ebbi", "ecc", "ecco", "ed", "effettivamente", "egli", "ella", "entrambi", "eppure", "era",
    "erano", "eravamo", "eravate", "eri", "ero", "esempio", "esse", "essendo", "esser", "essere", "essi", "ex", "fa",
    "faccia", "facciamo", "facciano", "facciate", "faccio", "facemmo", "facendo", "facesse", "facessero", "facessi",
    "facessimo", "faceste", "facesti", "faceva", "facevamo", "facevano", "facevate", "facevi", "facevo", "fai", "fanno",
    "farai", "faranno", "fare", "farebbe", "farebbero", "farei", "faremmo", "faremo", "fareste", "faresti", "farete",
    "farà", "farò", "fatto", "favore", "fece", "fecero", "feci", "fin", "finalmente", "finche", "fine", "fino", "forse",
    "forza", "fosse", "fossero", "fossi", "fossimo", "foste", "fosti", "fra", "frattempo", "fu", "fui", "fummo",
    "fuori", "furono", "futuro", "generale", "gente", "gia", "giacche", "giorni", "giorno", "giu", "già", "gli",
    "gliela", "gliele", "glieli", "glielo", "gliene", "grande", "grazie", "gruppo", "ha", "haha", "hai", "hanno",
    "ho", "i", "ie", "ieri", "il", "improvviso", "in", "inc", "indietro", "infatti", "inoltre", "insieme", "intanto",
    "intorno", "invece", "io", "l", "la", "lasciato", "lato", "le", "lei", "li", "lo", "lontano", "loro", "lui",
    "lungo", "luogo", "là", "ma", "macche", "magari", "maggior", "mai", "male", "malgrado", "malissimo", "me",
    "medesimo", "mediante", "meglio", "meno", "mentre", "mesi", "mezzo", "mi", "mia", "mie", "miei", "mila",
    "miliardi", "milioni", "minimi", "mio", "modo", "molta", "molti", "moltissimo", "molto", "momento", "mondo", "ne",
    "negl", "negli", "nei", "nel", "nell", "nella", "nelle", "nello", "nemmeno", "neppure", "nessun", "nessuna",
    "nessuno", "niente", "no", "noi", "nome", "non", "nondimeno", "nonostante", "nonsia", "nostra", "nostre", "nostri",
    "nostro", "novanta", "nove", "nulla", "nuovi", "nuovo", "o", "od", "oggi", "ogni", "ognuna", "ognuno", "oltre",
    "oppure", "ora", "ore", "osi", "ossia", "ottanta", "otto", "paese", "parecchi", "parecchie", "parecchio", "parte",
    "partendo", "peccato", "peggio", "per", "perche", "perchè", "perché", "percio", "perciò", "perfino", "pero",
    "persino", "persone", "però", "piedi", "pieno", "piglia", "piu", "piuttosto", "più", "po", "pochissimo", "poco",
    "poi", "poiche", "possa", "possedere", "posteriore", "posto", "potrebbe", "preferibilmente", "presa", "press",
    "prima", "primo", "principalmente", "probabilmente", "promesso", "proprio", "puo", "pure", "purtroppo", "può",
    "qua", "qualche", "qualcosa", "qualcuna", "qualcuno", "quale", "quali", "qualunque", "quando", "quanta", "quante",
    "quanti", "quanto", "quantunque", "quarto", "quasi", "quattro", "quel", "quella", "quelle", "quelli", "quello",
    "quest", "questa", "queste", "questi", "questo", "qui", "quindi", "quinto", "realmente", "recente", "recentemente",
    "registrazione", "relativo", "riecco", "rispetto", "salvo", "sara", "sarai", "saranno", "sarebbe", "sarebbero",
    "sarei", "saremmo", "saremo", "sareste", "saresti", "sarete", "sarà", "sarò", "scola", "scopo", "scorso", "se",
    "secondo", "seguente", "seguito", "sei", "sembra", "sembrare", "sembrato", "sembrava", "sembri", "sempre", "senza",
    "sette", "si", "sia", "siamo", "siano", "siate", "siete", "sig", "solito", "solo", "soltanto", "sono", "sopra",
    "soprattutto", "sotto", "spesso", "sta", "stai", "stando", "stanno", "starai", "staranno", "starebbe", "starebbero",
    "starei", "staremmo", "staremo", "stareste", "staresti", "starete", "starà", "starò", "stata", "state", "stati",
    "stato", "stava", "stavamo", "stavano", "stavate", "stavi", "stavo", "stemmo", "stessa", "stesse", "stessero",
    "stessi", "stessimo", "stesso", "steste", "stesti", "stette", "stettero", "stetti", "stia", "stiamo", "stiano",
    "stiate", "sto", "su", "sua", "subito", "successivamente", "successivo", "sue", "sugl", "sugli", "sui", "sul",
    "sull", "sulla", "sulle", "sullo", "suo", "suoi", "tale", "tali", "talvolta", "tanto", "te", "tempo", "terzo",
    "th", "ti", "titolo", "tra", "tranne", "tre", "trenta", "triplo", "troppo", "trovato", "tu", "tua", "tue", "tuo",
    "tuoi", "tutta", "tuttavia", "tutte", "tutti", "tutto", "uguali", "ulteriore", "ultimo", "un", "una", "uno",
    "uomo", "va", "vai", "vale", "vari", "varia", "varie", "vario", "verso", "vi", "vicino", "visto", "vita", "voi",
    "volta", "volte", "vostra", "vostre", "vostri", "vostro", "è"
}

PERSIAN_STOPWORDS = {
    "آباد", "آخ", "آخر", "آخرها", "آخه", "آدمهاست", "آرام", "آرام آرام", "آره", "آری", "آزادانه", "آسان", "آسیب پذیرند",
    "آشنایند", "آشکارا", "آقا", "آقای", "آقایان", "آمد", "آمدن", "آمده", "آمرانه", "آن", "آن گاه", "آنان", "آنانی",
    "آنجا", "آنرا", "آنطور", "آنقدر", "آنها", "آنهاست", "آنچنان", "آنچنان که", "اونجور", "اونجوری", "اونجوری که", "آنچه"
    , "آنکه", "آنگاه", "آن‌ها", "آهان", "آهای", "آور", "آورد", "آوردن", "آورده", "آوه", "آی", "آیا", "آید", "آیند", "ا",
    "اتفاقا", "اثرِ", "اجراست", "احتراما", "احتمالا", "احیاناً", "اخیر", "اخیراً", "اری", "از", "از آن پس", "از بس که",
    "از جمله", "ازاین رو", "ازجمله", "ازش", "اساسا", "اساساً", "است", "استفاد", "استفاده", "اسلامی اند", "اش", "اشتباها",
    "اشکارا", "اصلا", "اصلاً", "اصولا", "اصولاً", "اعلام", "اغلب", "افزود", "افسوس", "اقل", "اقلیت", "الا", "الان", "البته", "البتّه"
    , "الهی", "الی", "ام", "اما", "امروز", "امروزه", "امسال", "امشب", "امور", "امیدوارم", "امیدوارند", "امیدواریم",
    "ان", "ان شاأالله", "انشالا", "انتها", "انجام", "اند", "اندکی", "انشاالله", "انصافا", "انطور", "انقدر", "انها",
    "انچنان", "انکه", "انگار", "او", "اوست", "اول", "اولا", "اولاً", "اولین", "اون", "اکثر", "اکثرا", "اکثراً", "اکثریت",
    "اکنون", "اگر", "اگر چه", "اگرچه", "اگه", "ای", "ایا", "اید", "ایشان", "ایم", "این", "این جوری", "این قدر",
    "این گونه", "اینان", "اینجا", "اینجاست", "ایند", "اینطور", "اینقدر", "اینها", "اینهاست", "اینو", "اینچنین", "اینک",
    "اینکه", "اینگونه", "ب ", "با", "بااین حال", "بااین وجود", "باد", "بار", "بارة", "باره", "بارها", "باز", "باز هم",
    "بازهم", "بازی کنان", "بازیگوشانه", "باش", "باشد", "باشم", "باشند", "باشی", "باشید", "باشیم", "بالا", "بالاخره",
    "بالاخص", "بالاست", "بالای", "بالایِ", "بالطبع", "بالعکس", "باوجودی که", "باورند", "باید", "بتدریج", "بتوان", "بتواند",
    "بتوانی", "بتوانیم", "بجز", "بخش", "بخشه", "بخشی", "بخصوص", "بخواه", "بخواهد", "بخواهم", "بخواهند", "بخواهی",
    "بخواهید", "بخواهیم", "بخوبی", "بد", "بدان", "بدانجا", "بدانها", "بدهید", "بدون", "بدین", "بدین ترتیب", "بدینجا",
    "بر", "برآنند", "برا", "برابر", "برابرِ", "براحتی", "براساس", "براستی", "برای", "برایت", "برایش", "برایشان",
    "برایم", "برایمان", "برایِ", "برخوردار", "برخوردارند", "برخی", "برداری", "برعکس", "برنامه سازهاست", "بروز",
    "بروشنی", "بزرگ", "بزودی", "بس", "بسا", "بسادگی", "بسختی", "بسوی", "بسی", "بسیار", "بسیاری", "بشدت", "بطور",
    "بطوری که", "بعد", "بعد از این که", "بعدا", "بعدازظهر", "بعداً", "بعدها", "بعری", "بعضا", "بعضی", "بعضی شان",
    "بعضیهایشان", "بعضی‌ها", "بعلاوه", "بعید", "بفهمی نفهمی", "بلافاصله", "بله", "بلکه", "بلی", "بماند", "بنابراین",
    "بندی", "به", "به آسانی", "به تازگی", "به تدریج", "به تمامی", "به جای", "به جز", "به خوبی", "به درشتی", "به دلخواه",
    "به راستی", "به رغم", "به روشنی", "به زودی", "به سادگی", "به سرعت", "به شان", "به شدت", "به طور کلی", "به طوری که",
    "به علاوه", "به قدری", "به مراتب", "به ناچار", "به هرحال", "به هیچ وجه", "به وضوح", "به ویژه", "به کرات", "به گرمی",
    "بهت", "بهتر", "بهترین", "بهش", "بود", "بودم", "بودن", "بودند", "بوده", "بودی", "بودید", "بودیم", "بویژه", "بپا",
    "بکار", "بکن", "بکند", "بکنم", "بکنند", "بکنی", "بکنید", "بکنیم", "بگذاریم", "بگو", "بگوید", "بگویم", "بگویند",
    "بگویی", "بگویید", "بگوییم", "بگیر", "بگیرد", "بگیرم", "بگیرند", "بگیری", "بگیرید", "بگیریم", "بی", "بی آنکه",
    "بی اطلاعند", "بی تردید", "بی تفاوتند", "بی نیازمندانه", "بی هدف", "بیا", "بیاب", "بیابد", "بیابم", "بیابند",
    "بیابی", "بیابید", "بیابیم", "بیاور", "بیاورد", "بیاورم", "بیاورند", "بیاوری", "بیاورید", "بیاوریم", "بیاید",
    "بیایم", "بیایند", "بیایی", "بیایید", "بیاییم", "بیرون", "بیرونِ", "بیست", "بیش", "بیشتر", "بیشتری", "بین",
    "بیگمان", "ت", "تا", "تازه", "تان", "تاکنون", "تحت", "تحریم هاست", "تر", "تر براساس", "تریلیارد", "تریلیون",
    "ترین", "تصریحاً", "تعدادی", "تعمدا", "تقریبا", "تقریباً", "تلویحا", "تلویحاً", "تمام", "تمام قد", "تماما",
    "تمامشان", "تمامی", "تند تند", "تنها", "تو", "توؤماً", "توان", "تواند", "توانست", "توانستم", "توانستن", "توانستند",
    "توانسته", "توانستی", "توانستیم", "توانم", "توانند", "توانی", "توانید", "توانیم", "توسط", "تولِ", "توی", "تویِ",
    "تک تک", "ث", "ثالثاً", "ثانیا", "ثانیاً", "ج", "جا", "جای", "جایی", "جدا", "جداً", "جداگانه", "جدید", "جدیدا",
    "جرمزاست", "جریان", "جز", "جلو", "جلوگیری", "جلوی", "جلویِ", "جمع اند", "جمعا", "جمعی", "جنابعالی", "جناح",
    "جنس اند", "جهت", "جور", "ح", "حاشیه‌ای", "حاضر", "حاضرم", "حال", "حالا", "حاکیست", "حتما", "حتماً", "حتی", "حداقل",
    "حداکثر", "حدود", "حدودا", "حدودِ", "حسابگرانه", "حضرتعالی", "حق", "حقیرانه", "حقیقتا", "حول", "حکماً", "خ", "خارجِ",
    "خالصانه", "خب", "خداحافظ", "خداست", "خدمات", "خسته‌ای", "خصوصا", "خصوصاً", "خلاصه", "خواست", "خواستم", "خواستن",
    "خواستند", "خواسته", "خواستی", "خواستید", "خواستیم", "خواه", "خواهد", "خواهم", "خواهند", "خواهی", "خواهید",
    "خواهیم", "خوب", "خود", "خود به خود", "خودبه خودی", "خودت", "خودتان", "خودتو", "خودش", "خودشان", "خودم", "خودمان",
    "خودمو", "خوش", "خوشبختانه", "خویش", "خویشتن", "خویشتنم", "خیاه", "خیر", "خیره", "خیلی", "د", "دا", "داام",
    "دااما", "داخل", "داد", "دادم", "دادن", "دادند", "داده", "دادی", "دادید", "دادیم", "دار", "داراست", "دارد", "دارم",
    "دارند", "داری", "دارید", "داریم", "داشت", "داشتم", "داشتن", "داشتند", "داشته", "داشتی", "داشتید", "داشتیم",
    "دامم", "دانست", "دانند", "دایم", "دایما", "در", "در باره", "در بارهٌ", "در ثانی", "در مجموع", "در نهایت", "در واقع",
    "در کل", "در کنار", "دراین میان", "درباره", "درحالی که", "درحالیکه", "درست", "درست و حسابی", "درسته", "درصورتی که",
    "درعین حال", "درمجموع", "درواقع", "درون", "دریغ", "دریغا", "درین", "دسته دسته", "دشمنیم", "دقیقا", "دم", "دنبالِ",
    "ده", "دهد", "دهم", "دهند", "دهی", "دهید", "دهیم", "دو", "دو روزه", "دوباره", "دوم", "دیده", "دیر", "دیرت", "دیرم",
    "دیروز", "دیشب", "دیوانه‌ای", "دیوی", "دیگر", "دیگران", "دیگری", "دیگه", "ذ", "ذاتاً", "ر", "را", "راجع به", "راحت",
    "راسا", "راست", "راستی", "راه", "رسما", "رسید", "رسیده", "رشته", "رفت", "رفتارهاست", "رفته", "رنجند", "رهگشاست",
    "رو", "رواست", "روب", "روبروست", "روز", "روز به روز", "روزانه", "روزه ایم", "روزه ست", "روزه م", "روزهای", "روزه‌ای",
    "روش", "روی", "رویش", "رویِ", "ریزی", "ز", "زشتکارانند", "زمان", "زمانی", "زمینه", "زنند", "زهی", "زود", "زودتر",
    "زیاد", "زیاده", "زیر", "زیرا", "زیرِ", "زیرچشمی", "س", "سابق", "ساخته", "ساده اند", "سازی", "سالانه", "سالته",
    "سالم‌تر", "سالهاست", "سالیانه", "ساکنند", "سایر", "سخت", "سخته", "سر", "سراسر", "سرانجام", "سراپا", "سری", "سریع",
    "سریعا", "سریعاً", "سریِ", "سعی", "سمتِ", "سه باره", "سهواً", "سوم", "سوی", "سویِ", "سپس", "سیاه چاله هاست", "سیخ", "ش",
    "شان", "شاهدند", "شاهدیم", "شاید", "شبهاست", "شخصا", "شخصاً", "شد", "شدم", "شدن", "شدند", "شده", "شدی", "شدید",
    "شدیدا", "شدیداً", "شدیم", "شش", "شش نداشته", "شما", "شماری", "شماست", "شمایند", "شناسی", "شو", "شود", "شوراست",
    "شوقم", "شوم", "شوند", "شونده", "شوی", "شوید", "شویم", "شیرین", "شیرینه", "شیک", "ص", "صد", "صددرصد", "صرفا",
    "صرفاً", "صریحاً", "صندوق هاست", "صورت", "ض", "ضدِّ", "ضدِّ", "ضمن", "ضمناً", "ط", "طبعا", "طبعاً", "طبقِ", "طبیعتا", "طرف",
    "طریق", "طلبکارانه", "طور", "طی", "ظ", "ظاهرا", "ظاهراً", "ع", "عاجزانه", "عاقبت", "عبارتند", "عجب", "عجولانه",
    "عدم", "عرفانی", "عقب", "عقبِ", "علاوه بر", "علاوه بر آن", "علاوه برآن", "علناً", "علّتِ", "علی الظاهر", "علی رغم",
    "علیرغم", "علیه", "عمدا", "عمداً", "عمدتا", "عمدتاً", "عمده", "عمل", "عملا", "عملاً", "عملی اند", "عموم", "عموما",
    "عموماً", "عنقریب", "عنوان", "عنوانِ", "عیناً", "غ", "غالبا", "غزالان", "غیر", "غیرقانونی", "ف", "فاقد", "فبها", "فر",
    "فردا", "فعلا", "فعلاً", "فقط", "فلان", "فلذا", "فوق", "فکر", "ق", "قاالند", "قابل", "قاطبه", "قاطعانه", "قاعدتاً",
    "قانوناً", "قبل", "قبلا", "قبلاً", "قبلند", "قدر", "قدری", "قصدِ", "قضایاست", "قطعا", "قطعاً", "ل", "لااقل", "لاجرم",
    "لب", "لذا", "لزوماً", "لطفا", "لطفاً", "لیکن", "م", "ما", "مادامی", "ماست", "مامان مامان گویان", "مان", "مانند",
    "مانندِ", "مبادا", "متؤسفانه", "متاسفانه", "متعاقبا", "متفاوتند", "مثل", "مثلا", "مثلِ", "مجانی", "مجبورند", "مجددا",
    "مجدداً", "مجموعا", "مجموعاً", "محتاجند", "محکم", "محکم‌تر", "مخالفند", "مختلف", "مخصوصاً", "مدام", "مدت", "مدتهاست",
    "مدّتی", "مذهبی اند", "مرا", "مرتب", "مردانه", "مردم", "مردم اند", "مرسی", "مستحضرید", "مستقیما", "مستند", "مسلما",
    "مشت", "مشترکاً", "مشغولند", "مطمانا", "مطمانم", "مطمینا", "مع الاسف", "مع ذلک", "معتقدم", "معتقدند", "معتقدیم",
    "معدود", "معذوریم", "معلومه", "معمولا", "معمولاً", "معمولی", "مغرضانه", "مفیدند", "مقابل", "مقدار", "مقصرند", "مقصری",
    "ملیارد", "ملیون", "ممکن", "ممیزیهاست", "من", "منتهی", "منطقی", "منی", "مواجهند", "موارد", "موجودند", "مورد",
    "موقتا", "مکرر", "مکرراً", "مگر", "مگر آن که", "مگر این که", "مگو", "می", "میان", "میزان", "میلیارد", "میلیون",
    "میکند", "میکنم", "میکنند", "میکنی", "میکنید", "میکنیم", "می‌تواند", "می‌خواهیم", "می‌داند", "می‌رسد", "می‌رود",
    "می‌شود", "می‌کنم", "می‌کنند", "می‌کنیم", "ن", "ناامید", "ناخواسته", "ناراضی اند", "ناشی", "نام", "ناگاه", "ناگزیر",
    "ناگهان", "ناگهانی", "نباید", "نبش", "نبود", "نخست", "نخستین", "نخواهد", "نخواهم", "نخواهند", "نخواهی", "نخواهید",
    "نخواهیم", "نخودی", "ندارد", "ندارم", "ندارند", "نداری", "ندارید", "نداریم", "نداشت", "نداشتم", "نداشتند",
    "نداشته", "نداشتی", "نداشتید", "نداشتیم", "نزد", "نزدِ", "نزدیک", "نزدیکِ", "نسبتا", "نشان", "نشده", "نظیر", "نفرند",
    "نماید", "نموده", "نمی", "نمی‌شود", "نمی‌کند", "نه", "نه تنها", "نهایتا", "نهایتاً", "نوع", "نوعاً", "نوعی", "نکرده",
    "نکن", "نکند", "نکنم", "نکنند", "نکنی", "نکنید", "نکنیم", "نگاه", "نگو", "نیازمندند", "نیز", "نیست", "نیستم",
    "نیستند", "نیستیم", "نیمی", "ه", "ها", "های", "هایی", "هبچ", "هر", "هر از گاهی", "هر چند", "هر چند که", "هر چه",
    "هرچند", "هرچه", "هرکس", "هرگاه", "هرگز", "هزار", "هست", "هستم", "هستند", "هستی", "هستید", "هستیم", "هفت",
    "هق هق کنان", "هم", "هم اکنون", "هم اینک", "همان", "همان طور که", "همان گونه که", "همانا", "همانند", "همانها",
    "همدیگر", "همزمان", "همه", "همه روزه", "همه ساله", "همه شان", "همهٌ", "همه‌اش", "همواره", "همچنان", "همچنان که",
    "همچنین", "همچون", "همچین", "همگان", "همگی", "همیشه", "همین", "همین که", "هنوز", "هنگام", "هنگامِ", "هنگامی",
    "هنگامی که", "هوی", "هی", "هیچ", "هیچ گاه", "هیچکدام", "هیچکس", "هیچگاه", "هیچگونه", "هیچی", "و", "و لا غیر",
    "وابسته اند", "واقعا", "واقعاً", "واقعی", "واقفند", "واما", "وای", "وجود", "وحشت زده", "وسطِ", "وضع", "وقتی",
    "وقتی که", "وقتیکه", "ولی", "وگرنه", "وگو", "وی", "ویا", "ویژه", "ّه", "٪", "پ", "پارسال", "پارسایانه", "پاره‌ای",
    "پاعینِ", "پایین ترند", "پدرانه", "پرسان", "پروردگارا", "پریروز", "پس", "پس از", "پس فردا", "پشت", "پشتوانه اند",
    "پشیمونی", "پنج", "پهن شده", "پی", "پی درپی", "پیدا", "پیداست", "پیرامون", "پیش", "پیشاپیش", "پیشتر", "پیشِ",
    "پیوسته", "چ", "چاپلوسانه", "چت", "چته", "چرا", "چرا که", "چشم بسته", "چطور", "چقدر", "چنان", "چنانچه", "چنانکه",
    "چند", "چند روزه", "چندان", "چنده", "چندین", "چنین", "چه", "چه بسا", "چه طور", "چهار", "چو", "چون", "چکار",
    "چگونه", "چی", "چیز", "چیزی", "چیزیست", "چیست", "چیه", "ژ", "ک", "کارند", "کاش", "کاشکی", "کامل", "کاملا", "کاملاً",
    "کتبا", "کجا", "کجاست", "کدام", "کرد", "کردم", "کردن", "کردند", "کرده", "کردی", "کردید", "کردیم", "کس", "کسانی",
    "کسی", "کل", "کلا", "کلی", "کلیه", "کم", "کم کم", "کمااینکه", "کماکان", "کمتر", "کمتره", "کمتری", "کمی", "کن",
    "کنار", "کنارش", "کنارِ", "کنایه‌ای", "کند", "کنم", "کنند", "کننده", "کنون", "کنونی", "کنی", "کنید", "کنیم", "که",
    "کو", "کَی", "کی", "گ", "گاه", "گاهی", "گذاری", "گذاشته", "گذشته", "گردد", "گردند", "گرفت", "گرفتارند", "گرفتم",
    "گرفتن", "گرفتند", "گرفته", "گرفتی", "گرفتید", "گرفتیم", "گروهی", "گرچه", "گفت", "گفتم", "گفتن", "گفتند", "گفته",
    "گفتی", "گفتید", "گفتیم", "گه", "گهگاه", "گو", "گونه", "گوی", "گویا", "گوید", "گویم", "گویند", "گویی", "گویید",
    "گوییم", "گیر", "گیرد", "گیرم", "گیرند", "گیری", "گیرید", "گیریم", "ی", "یا", "یاب", "یابد", "یابم", "یابند",
    "یابی", "یابید", "یابیم", "یارب", "یافت", "یافتم", "یافتن", "یافته", "یافتی", "یافتید", "یافتیم", "یعنی", "یقینا",
    "یقیناً", "یه", "یواش یواش", "یک", "یک جوری", "یک کم", "یک کمی", "یکدیگر", "یکریز", "یکسال", "یکهزار", "یکی",
    "۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹", "…", "﻿و", "‏‏‏علاقه مند", "میخونم", "میخوانم",
    "می خوانم", "میخونید", "میخوانید", "می خوانید", "در آینده", "بشم", "بشی", "بشید", "بشین", "یک چیزی", "بهتون",
    "اینم", "بیفته", "محض رضای خدا", "هیچوقت", "دونستن", "میفرستین", "میفرستی", "میفرستم", "عه", "هستش", "همه‌مون",
    "همه مون", "جدی", "بدجور", "بد جور", "خداروشکر", "شی", "وجدانا", "روم", "بگین", "هیچ جور", "هیچجور", "هیچ‌جور",
    "مثل اینکه", "دوهزاری", "هستا", "شون", "هامو", "هام رو", "مارو", "ما رو", "رو", "داره", "این دفعه", "دفعه"
}


class WordnetSense:
    FASTTEXT_MODELS = {}
    STOPWORDS = {"en": ENGLISH_STOPWORDS, "it": ITALIAN_STOPWORDS, "fa": PERSIAN_STOPWORDS}

    def __init__(self, wn, wn_lang: str, sense: Synset, lang: str):
        self.wn = wn
        self.sense = sense
        self.wn_lang = wn_lang
        self.lang = lang
        self.fasttext = None
        self.stopwords = WordnetSense.STOPWORDS[lang]

    def _load_fasttext(self, lang: str):
        if lang in WordnetSense.FASTTEXT_MODELS:
            return WordnetSense.FASTTEXT_MODELS[lang]
        else:
            model_name = f"cc.{lang}.300.bin"
            logging.info("Loading fasttext model %s", model_name)
            model = fasttext.load_model(os.path.join("embeddings", model_name))
            WordnetSense.FASTTEXT_MODELS[lang] = model
            return model

    @property
    def id(self):
        return f"{self.wn_lang}:{self.sense.name()}"

    def score_sense(self, context_words: Set[str]) -> float:
        sense_words = self._extract_sense_words()
        score = 0.0
        words = [word for word in context_words if word not in self.stopwords]
        if len(words) == 0:
            words = context_words
        for word in words:
            if word in sense_words:
                score += 1.0 / len(sense_words)
            else:
                return 0.0
        return score

    def score_sense_vectors(self, context_words: Set[str]) -> float:
        if self.fasttext is None: self.fasttext = self._load_fasttext(self.lang)
        context_words = list(word for word in context_words if word not in self.stopwords)
        sense_words = list(word for word in self._extract_sense_words() if word not in self.stopwords)
        if len(context_words) == 0 or len(sense_words) == 0: return 0.0
        context_emb = np.vstack([self.vec(word) for word in context_words])
        sense_emb = np.vstack([self.vec(word) for word in sense_words])
        scores = cos_sim(context_emb, sense_emb).numpy()
        res = float(scores.max())
        min_proba = {"en": 0.7, "it": 0.5, "fa": 0.3}
        return res if res > min_proba[self.lang] else 0.0

    def vec(self, word: str):
        if word in self.fasttext:
            return self.fasttext[word]
        else:
            return np.zeros(300, dtype=np.float32)

    def _extract_sense_words(self, hypernyms=True, meronyms=False, examples=True):
        res = self._sense_to_words(self.sense)
        if hypernyms:
            for hypernym in self.sense.hypernyms():
                res.update(self._sense_to_words(hypernym))
            for hypernym in self.sense.instance_hypernyms():
                res.update(self._sense_to_words(hypernym))
        if meronyms:
            for meronym in self.sense.member_meronyms():
                res.update(self._sense_to_words(meronym))
            for meronym in self.sense.substance_meronyms():
                res.update(self._sense_to_words(meronym))
        if examples:
            examples = self.sense.examples(lang=self.wn_lang)
            if examples is not None and len(examples) > 0:
                text = re.sub(r"[\W_]+", " ", " ".join(examples), flags=re.UNICODE)
                res.update(set(text.lower().split()))
        return res

    def _sense_to_words(self, sense: Synset):
        texts = (sense.definition(lang=self.wn_lang), *sense.lemma_names(lang=self.wn_lang))
        texts = [v for v in texts if v is not None]
        text = " ".join([" ".join(v) if isinstance(v, list) else v for v in texts])
        text = re.sub(r"[\W_]+", " ", text, flags=re.UNICODE)
        words = set(text.lower().split())
        return words

    def extract_context(self, sample: InputSample, include_hypernyms=True, include_meronyms=True):
        res = [sample.context]
        self._add_sense_context(res, self.sense)
        if include_hypernyms:
            for hypernym in self.sense.hypernyms():
                self._add_sense_context(res, hypernym)
            for hypernym in self.sense.instance_hypernyms():
                self._add_sense_context(res, hypernym)
        if include_meronyms:
            for meronym in self.sense.member_meronyms():
                self._add_sense_context(res, meronym)
            for meronym in self.sense.substance_meronyms():
                self._add_sense_context(res, meronym)
        res = [val.lower() for val in res if val]
        return ", ".join(res)

    def _add_sense_context(self, res: List[str], sense: Synset):
        for lemma_name in sense.lemma_names(lang=self.wn_lang):
            res.append(re.sub(r"[\W_]+", " ", lemma_name, flags=re.UNICODE))
