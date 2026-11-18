"""Fictional world: entities, facts, and paragraph rendering.

Every queried value in the world is unique (years, river names, entity names), so a
string search tells us exactly which paragraphs could answer a question. Paragraphs
are a fact sentence plus neutral sentences about the same subject, padded to a
target token length. Neutral sentences never state a queryable attribute.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- seeding


def derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from any parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def rng_for(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))


# --------------------------------------------------------------------------- tokens


class TokenCounter:
    """Counts tokens with a Hugging Face tokenizer (Qwen2.5 by default)."""

    def __init__(self, name: str):
        from tokenizers import Tokenizer

        self.name = name
        self._tok = Tokenizer.from_pretrained(name)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids)


# --------------------------------------------------------------------------- names

_ONSETS = ["b", "br", "d", "f", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v",
           "z", "dr", "tr", "st", "th"]
_VOWELS = ["a", "e", "i", "o", "u", "a", "e", "o", "ai", "ei", "ou"]
_CODAS = ["", "", "", "", "n", "l", "r", "s", "m"]
_TOWN_ENDINGS = ["avel", "orin", "eth", "ussa", "iri", "ondel", "ask", "umbra", "esk", "ovar"]
_COMPANY_KINDS = ["Works", "Foundry", "Textiles", "Instruments", "Optics", "Mills",
                  "Ceramics", "Laboratories", "Ironworks", "Glassworks"]
_DEVICES = ["loom", "lamp", "press", "valve", "compass", "kiln", "pump", "lens",
            "clock", "sextant", "bellows", "lathe"]


def _syllable(rng: random.Random) -> str:
    return rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)


def _word(rng: random.Random, n_syll: int) -> str:
    return "".join(_syllable(rng) for _ in range(n_syll)).capitalize()


# --------------------------------------------------------------------------- relations


@dataclass(frozen=True)
class Relation:
    name: str
    subject_type: str
    value_kind: str  # "town" | "person" | "company" | "year" | "river"
    templates: tuple[str, ...]  # {s} subject, {v} value; at least 6 for paraphrases
    questions: tuple[str, ...]


RELATIONS: dict[str, Relation] = {r.name: r for r in [
    Relation("born_in", "person", "town", (
        "{s} was born in {v}.",
        "{v} is the town where {s} was born.",
        "Records kept in {v} list it as the birthplace of {s}.",
        "{s} came into the world in the town of {v}.",
        "The birthplace of {s} was {v}.",
        "Born in {v}, {s} spent the early years of childhood there.",
    ), ("In which town was {s} born?", "What is the birthplace of {s}?", "Where was {s} born?")),
    Relation("works_for", "person", "company", (
        "{s} works for {v}.",
        "{s} is employed by {v}.",
        "{v} counts {s} among its staff.",
        "{s} holds a permanent position at {v}.",
        "The employer of {s} is {v}.",
        "Most weekdays, {s} can be found at {v}, the company that employs {s}.",
    ), ("Which company does {s} work for?", "Who employs {s}?", "What company employs {s}?")),
    Relation("birth_year", "person", "year", (
        "{s} was born in the year {v}.",
        "The year of birth of {s} was {v}.",
        "Parish books record the birth of {s} in {v}.",
        "{v} was the year {s} was born.",
        "It was in {v} that {s} was born.",
        "{s}, born in {v}, grew up in a large and noisy household.",
    ), ("In what year was {s} born?", "What is the birth year of {s}?")),
    Relation("town_founded", "town", "year", (
        "{s} was founded in {v}.",
        "The town of {s} dates its founding to {v}.",
        "Settlers established {s} in {v}.",
        "{v} is the year {s} was founded.",
        "The founding of {s} took place in {v}.",
        "{s} has existed since its founding in {v}.",
    ), ("In what year was {s} founded?", "When was the town of {s} founded?")),
    Relation("river", "town", "river", (
        "The river {v} runs through {s}.",
        "{s} lies on the banks of the river {v}.",
        "The {v} is the river that flows through {s}.",
        "{s} was built where the river {v} widens.",
        "The river passing through {s} is the {v}.",
        "Boats on the river {v} pass through the centre of {s}.",
    ), ("Which river runs through {s}?", "What river flows through the town of {s}?")),
    Relation("founded_by", "company", "person", (
        "{s} was founded by {v}.",
        "{v} founded {s}.",
        "The founder of {s} is {v}.",
        "{s} was started by {v}.",
        "{v} is credited with establishing {s}.",
        "{s} owes its existence to its founder, {v}.",
    ), ("Who founded {s}?", "Who is the founder of {s}?")),
    Relation("hq_in", "company", "town", (
        "{s} is headquartered in {v}.",
        "The headquarters of {s} are in {v}.",
        "{s} has its main offices in {v}.",
        "{v} is home to the headquarters of {s}.",
        "{s} runs its operations from a headquarters in {v}.",
        "From its headquarters in {v}, {s} manages all of its business.",
    ), ("In which town is {s} headquartered?", "Where are the headquarters of {s}?")),
    Relation("company_founded", "company", "year", (
        "{s} was founded in the year {v}.",
        "{s} opened for business in {v}.",
        "The company {s} was established in {v}.",
        "{v} is the year {s} was established.",
        "{s} began trading in {v}.",
        "Since it was established in {v}, {s} has grown steadily.",
    ), ("In what year was {s} founded?", "When was the company {s} established?")),
    Relation("invented_by", "invention", "person", (
        "The {s} was invented by {v}.",
        "{v} invented the {s}.",
        "The inventor of the {s} was {v}.",
        "The first {s} was built by {v}.",
        "Credit for inventing the {s} goes to {v}.",
        "{v} designed and built the very first {s}.",
    ), ("Who invented the {s}?", "Who is the inventor of the {s}?")),
    Relation("made_by", "invention", "company", (
        "The {s} is manufactured by {v}.",
        "{v} makes the {s}.",
        "The {s} is produced by {v}.",
        "The maker of the {s} is {v}.",
        "Every {s} sold today comes from the workshops of {v}.",
        "{v} is the company that manufactures the {s}.",
    ), ("Which company manufactures the {s}?", "Who makes the {s}?")),
    Relation("invention_year", "invention", "year", (
        "The {s} was invented in {v}.",
        "The {s} first appeared in {v}.",
        "In {v}, the first {s} was built.",
        "The {s} dates from {v}.",
        "{v} is the year the {s} was invented.",
        "The year the {s} was invented was {v}.",
    ), ("In what year was the {s} invented?", "When was the {s} invented?")),
]}

# (first relation, second relation) -> question; the first relation's value is the
# subject of the second. Answer = value of the second fact.
MULTI_HOP: dict[tuple[str, str], str] = {
    ("born_in", "town_founded"): "In what year was the town where {s} was born founded?",
    ("born_in", "river"): "Which river runs through the town where {s} was born?",
    ("works_for", "hq_in"): "In which town is the company that employs {s} headquartered?",
    ("works_for", "company_founded"): "In what year was the company that employs {s} founded?",
    ("founded_by", "born_in"): "In which town was the founder of {s} born?",
    ("founded_by", "birth_year"): "In what year was the founder of {s} born?",
    ("invented_by", "born_in"): "In which town was the inventor of the {s} born?",
    ("invented_by", "birth_year"): "In what year was the inventor of the {s} born?",
    ("made_by", "hq_in"): "In which town is the maker of the {s} headquartered?",
    ("made_by", "company_founded"): "In what year was the maker of the {s} founded?",
    ("hq_in", "river"): "Which river runs through the town where {s} is headquartered?",
    ("hq_in", "town_founded"): "In what year was the town where {s} is headquartered founded?",
}

# Relations whose value is a unique string never reused elsewhere: safe to hide for
# unanswerable questions (the value then appears nowhere in the dataset).
HIDEABLE = ("birth_year", "town_founded", "company_founded", "invention_year", "river")

# Words a neutral sentence must never contain (they would state or hint at a fact).
FACT_WORDS = re.compile(
    r"\b(born|birth|found\w*|establish\w*|work\w*|employ\w*|staff|river\w*|headquarter\w*|"
    r"invent\w*|manufactur\w*|maker|made|produc\w*|year)\b|\d", re.I)

# --------------------------------------------------------------------------- neutral text

_SLOTS: dict[str, list[str]] = {
    "habit": ["arriving early to every appointment", "telling long stories with surprising "
              "endings", "remembering everyone's favourite tea", "fixing broken things "
              "without being asked", "humming while concentrating", "keeping meticulous lists"],
    "collection": ["old maps", "pressed flowers", "brass buttons", "pocket knives",
                   "sea glass", "postcards", "seashells", "antique keys"],
    "hobby": ["walking along quiet lanes", "sketching birds", "repairing clocks",
              "baking bread", "reading old almanacs", "tending a vegetable patch",
              "solving crossword puzzles", "playing chess in the park"],
    "trait": ["patient", "curious", "meticulous", "generous", "stubborn", "cheerful",
              "reserved", "resourceful", "thoughtful", "blunt"],
    "skill": ["bind books by hand", "carve wood", "sail a small boat", "weave baskets",
              "identify wild mushrooms", "paint with watercolours", "keep bees"],
    "food": ["lentil stew", "roasted root vegetables", "barley soup", "spiced rice",
             "fresh bread with honey", "baked apples"],
    "season": ["spring", "summer", "autumn", "winter"],
    "activity": ["feeding the birds", "sweeping the front step", "jogging slowly",
                 "reading on a bench", "watering the plants", "walking a dog"],
    "instrument": ["violin", "flute", "accordion", "cello", "guitar", "piano"],
    "animals": ["cats", "dogs", "horses", "owls", "goats", "rabbits"],
    "tree": ["linden", "maple", "chestnut", "birch", "poplar", "elm"],
    "color": ["gold", "deep red", "bright orange", "pale yellow", "copper"],
    "feature": ["the painted shutters on the houses", "the steep cobbled lanes",
                "the many small fountains", "the quiet courtyards", "the old stone bridges",
                "the flower boxes on every balcony"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "produce": ["cheese and eggs", "apples and pears", "cabbages and onions",
                "honey and wax", "wool and leather"],
    "material": ["grey stone", "red brick", "pale timber", "whitewashed clay", "dark slate"],
    "specialty": ["smoked cheese", "woven blankets", "plum brandy", "painted pottery",
                  "lace", "rye bread"],
    "exhibit": ["old farming tools", "embroidered costumes", "hand-drawn maps",
                "carved wooden masks", "antique toys"],
    "people_doing": ["chatting over coffee", "playing cards", "listening to street "
                     "musicians", "strolling arm in arm"],
    "festfood": ["roasted chestnuts", "sweet pastries", "grilled sausages", "spiced cider"],
    "wclimate": ["cold and snowy", "damp and grey", "mild but windy", "crisp and clear"],
    "sclimate": ["warm and dry", "hot and dusty", "pleasantly breezy", "humid and green"],
    "landscape": ["fields", "hills", "orchards", "woodland", "meadows"],
    "building": ["grain store", "chapel", "school house", "customs office", "stable"],
    "landmark": ["a tall bell tower", "a crooked clock tower", "a hilltop chapel",
                 "a ruined watchtower", "a copper-domed hall"],
    "quality": ["durable", "carefully finished", "affordable", "elegant", "reliable"],
    "decor": ["framed photographs of old machinery", "potted ferns", "hand-woven rugs",
              "maps of distant coastlines", "watercolour landscapes"],
    "logo": ["a stylised owl", "a pair of crossed keys", "a rising sun", "an anchor",
             "a coiled spring", "a simple oak leaf"],
    "rep": ["paying suppliers promptly", "answering every letter", "careful record keeping",
            "long apprenticeships", "never missing a delivery date"],
    "praise": ["clear instructions", "quick repairs", "honest pricing",
               "friendly advice", "sturdy packaging"],
    "event": ["science fair", "spelling contest", "summer football match", "drawing contest"],
    "smell": ["machine oil", "fresh sawdust", "hot metal", "beeswax", "wet clay"],
    "drink": ["black tea", "strong coffee", "mint tea", "hot chocolate"],
    "compview": ["a careful and conservative rival", "a tireless rival",
                 "a stubborn but fair rival", "an unpredictable rival"],
    "imaterial": ["brass", "oak", "steel", "copper", "walnut", "glass", "iron"],
    "adj": ["quiet", "light", "sturdy", "simple to use", "precise", "easy to clean"],
    "size": ["bread box", "shoe box", "small suitcase", "kitchen chair", "teapot"],
    "tool": ["a small screwdriver", "a pair of pliers", "a soft cloth and some oil",
             "a single spanner"],
    "condition": ["dry", "warm", "cool", "dusty", "calm"],
    "ivalue": ["craftsmanship", "rarity", "polished fittings", "elegant lines"],
    "duration": ["a week", "an afternoon", "a few days", "a month"],
    "sound": ["clicking", "humming", "whirring", "ticking", "rattling"],
}

_NEUTRAL: dict[str, list[str]] = {
    "person": [
        "{n} is known among friends for {habit}.",
        "{n} keeps a collection of {collection} on a shelf by the window.",
        "On free afternoons, {n} enjoys {hobby}.",
        "Friends describe {n} as {trait} and {trait}.",
        "{n} once spent a whole summer learning to {skill}.",
        "A favourite meal of {n} is {food}, cooked slowly over a low flame.",
        "{n} prefers {season} to any other season.",
        "Neighbours often see {n} {activity} early in the morning.",
        "{n} has a quiet way of speaking that makes people lean in to listen.",
        "Letters written by {n} are usually short, practical and signed with a small sketch.",
        "{n} rarely travels without a worn leather bag full of {collection}.",
        "Acquaintances say {n} can recall a conversation long after everyone else has "
        "forgotten it.",
        "{n} has always been fond of {animals}.",
        "When asked about the future, {n} tends to answer with another question.",
        "{n} learned to play the {instrument} as a child and still practises on quiet "
        "evenings.",
        "The handwriting of {n} is small, neat and slightly slanted to the left.",
    ],
    "town": [
        "The streets of {n} are lined with {tree} trees that turn {color} in autumn.",
        "Visitors to {n} often remark on {feature}.",
        "The market in {n} is busiest on {day} mornings, when farmers bring {produce}.",
        "Most houses in {n} are built from {material}, with steep roofs against the rain.",
        "{n} is known in the region for its {specialty}.",
        "A small museum in {n} displays {exhibit}.",
        "In the evenings, the main square of {n} fills with people {people_doing}.",
        "The people of {n} hold a festival each {season} with music and {festfood}.",
        "Winters in {n} are {wclimate}, and summers are {sclimate}.",
        "Narrow footpaths connect the older quarters of {n} to the surrounding {landscape}.",
        "Many families in {n} have lived there for generations.",
        "The library of {n} is housed in a former {building}.",
        "Travellers approaching {n} first see {landmark} rising above the rooftops.",
        "Cats sleep on the warm stone steps of {n} on sunny afternoons.",
    ],
    "company": [
        "{n} is known for its {quality} goods and {quality} customer service.",
        "The offices of {n} are decorated with {decor}.",
        "Everyone at {n} shares a long lunch every {day}.",
        "{n} sends out a short newsletter each {season} describing its latest projects.",
        "The logo of {n} shows {logo}.",
        "{n} has a reputation for {rep}.",
        "Customers often praise {n} for {praise}.",
        "Each {season}, {n} sponsors a {event} for local schools.",
        "The premises of {n} smell faintly of {smell}.",
        "{n} keeps detailed archives of every order it has ever filled.",
        "Meetings at {n} traditionally begin with a cup of {drink}.",
        "Competitors regard {n} as {compview}.",
        "Visitors to {n} are greeted at the door and offered a short tour.",
    ],
    "invention": [
        "The {n} is assembled mostly from {imaterial} and {imaterial}.",
        "Early users of the {n} praised how {adj} it was.",
        "The {n} is roughly the size of a {size}.",
        "Repairing the {n} requires only {tool}.",
        "The {n} performs best in {condition} conditions.",
        "Collectors value well-kept examples of the {n} for their {ivalue}.",
        "The design of the {n} has changed little over time.",
        "Operating the {n} takes some practice, but most people learn within {duration}.",
        "The {n} gives off a soft {sound} sound while in use.",
        "Some households keep the {n} on display as a decorative object.",
        "Instructions for the {n} are printed on a small card attached to its side.",
        "The {n} is often painted {color} so it is easy to find in a crowded shed.",
    ],
}

# Off-topic filler: everyday subjects with no named entities and no digits.
_OFFTOPIC: dict[str, list[str]] = {
    "clouds": [
        "Clouds form when warm, moist air rises and cools until water vapour condenses.",
        "Thin, wispy clouds high in the sky are made almost entirely of ice crystals.",
        "A sky full of {cloudshape} clouds often means the weather is about to change.",
        "Fog is simply a cloud that has settled close to the ground.",
        "Pilots learn to read cloud shapes because they reveal how the air is moving.",
        "On a {season} afternoon, towering clouds can build quickly and bring heavy showers.",
        "The colour of a cloud depends on how thick it is and where the sun sits.",
        "Some people keep a cloud diary, noting each morning what the sky looks like.",
        "Low grey clouds that cover the whole sky can linger for days without rain.",
        "Sunlight scattered through a thin cloud sometimes forms a bright ring around the sun.",
        "Mountains force air upwards, so their peaks are often wrapped in cloud.",
        "Children love to find the shapes of animals in slowly drifting clouds.",
    ],
    "baking": [
        "Bread dough rises faster in a warm kitchen because yeast becomes more active.",
        "Kneading develops gluten, which gives bread its chewy texture.",
        "A loaf is usually done when it sounds hollow when tapped on the bottom.",
        "Many bakers add a pinch of salt to {pastry} dough to bring out its flavour.",
        "Resting dough in the fridge overnight gives a deeper, slightly sour taste.",
        "Steam in the oven during the first minutes of baking helps form a crisp crust.",
        "Butter should be cold when making flaky {pastry}, so it melts in the oven.",
        "Old bread can be turned into crumbs, puddings or thick soups.",
        "Flour with more protein makes a stronger dough that holds its shape.",
        "A sourdough starter must be fed regularly with fresh flour and water.",
        "Cooling bread on a wire rack stops the bottom of the loaf from turning soggy.",
        "Seeds sprinkled on top of a loaf add crunch and a nutty flavour.",
    ],
    "stars": [
        "On a clear night far from city lights, thousands of stars are visible.",
        "Stars appear to twinkle because their light passes through moving air.",
        "Planets usually shine with a steady light, unlike the flickering stars.",
        "The {celestial} is easiest to see on a moonless night.",
        "Early sailors used the positions of stars to find their way across the sea.",
        "A small pair of binoculars reveals far more stars than the naked eye.",
        "Shooting stars are tiny grains of dust burning up high in the atmosphere.",
        "Stargazers let their eyes adjust to the dark for a while before looking up.",
        "Constellations are patterns that people have imagined among the stars for ages.",
        "The moon is bright enough to wash out faint stars when it is full.",
        "Some stars look reddish and others bluish because they burn at different heats.",
        "A star chart turned to match the season helps beginners find their bearings.",
    ],
    "gardening": [
        "Tomatoes grow best in full sun with regular, deep watering.",
        "A layer of mulch keeps soil moist and stops weeds from spreading.",
        "Gardeners often plant {flower} near vegetables to attract helpful insects.",
        "Compost turns kitchen scraps and fallen leaves into rich, dark soil.",
        "Pruning in late winter encourages fruit trees to produce more blossom.",
        "Seedlings grown indoors need a few weeks outside before they are planted out.",
        "Rainwater collected in barrels is gentler on plants than tap water.",
        "Slugs can be kept away from lettuce with a ring of crushed eggshells.",
        "Bean plants climb happily up a frame of canes tied together at the top.",
        "Watering in the early morning lets leaves dry before the heat of the day.",
        "Earthworms loosen the soil and help air and water reach the roots.",
        "Herbs such as thyme and rosemary prefer poor, well-drained soil.",
    ],
    "tides": [
        "Tides rise and fall as the pull of the moon tugs on the oceans.",
        "Rock pools left by the falling tide are home to crabs, snails and anemones.",
        "The strongest tides happen when the sun and moon line up.",
        "Walkers on a beach should check the tide times before exploring.",
        "Seaweed clings to rocks with a root-like grip so the waves cannot wash it away.",
        "In narrow bays, the incoming tide can rush in surprisingly fast.",
        "Wading birds follow the edge of the water as the tide goes out.",
        "Sand ripples on the shore show the direction the water last flowed.",
        "Harbour walls protect moored boats from the force of incoming waves.",
        "At low tide, long stretches of wet sand reveal shells and driftwood.",
        "Fishermen time their trips so they can return before the water drops too far.",
        "The sound of the sea changes as the tide moves closer to the shore.",
    ],
    "bees": [
        "Honeybees communicate the location of flowers through a waggle dance.",
        "A single hive can contain tens of thousands of worker bees in summer.",
        "Bees collect nectar to make honey and pollen to feed their young.",
        "Beekeepers use gentle smoke to calm a colony before opening the hive.",
        "Bumblebees can fly in cooler weather than most other bees.",
        "Wax is produced by young worker bees and shaped into six-sided cells.",
        "Many crops depend on bees to carry pollen from flower to flower.",
        "In {season}, a colony may split and send a swarm to find a new home.",
        "A queen bee can lay a great many eggs on a single summer day.",
        "Honey stored in sealed cells can keep for a remarkably long time.",
        "Bees fan their wings at the hive entrance to cool the colony on hot days.",
        "Wildflower meadows offer bees a varied diet across the warm months.",
    ],
    "tea": [
        "Green tea is usually brewed with water that is hot but not boiling.",
        "Black tea leaves are fully oxidised, which gives them their dark colour.",
        "Warming the pot before brewing keeps the tea hot for longer.",
        "Some people add {teaextra} to their tea, while others drink it plain.",
        "Loose leaves need room to unfold, so a roomy strainer gives better flavour.",
        "Steeping tea for too long can make it taste bitter.",
        "Herbal infusions are made from flowers, leaves or roots rather than tea leaves.",
        "Tea should be stored in a dry, dark container away from strong smells.",
        "Tea grown on high, misty slopes is often said to have a delicate flavour.",
        "A tight lid on the pot keeps the aroma from escaping while the tea steeps.",
        "Some cultures serve tea with sweets, others with savoury snacks.",
        "Iced tea is brewed strong and then poured over plenty of ice.",
    ],
    "chess": [
        "In chess, controlling the centre of the board early is usually an advantage.",
        "Each player begins with sixteen pieces arranged in two rows.",
        "Knights are the only pieces that can jump over others.",
        "Beginners are often advised to castle early to keep the king safe.",
        "A pawn that reaches the far side of the board can become a queen.",
        "Strong players study endgames because small advantages decide them.",
        "The {chessidea} is one of the first tactics most new players learn.",
        "A game can end in a draw when neither side can make progress.",
        "Opening moves that develop knights and bishops quickly are generally sound.",
        "A clock is used in many games so that each player has a limited time to think.",
        "Sacrificing a piece can be worthwhile if it leads to a strong attack.",
        "Rooks become most powerful once the board has cleared in the later stages.",
    ],
    "cycling": [
        "Keeping bicycle tyres properly inflated makes riding easier and safer.",
        "A drop of oil on the chain every few weeks keeps the gears running smoothly.",
        "Riders lean slightly into a turn to keep their balance.",
        "Lights and reflective clothing help cyclists be seen at dusk.",
        "Climbing a long hill is easier in a low gear with a steady rhythm.",
        "Many towns now paint separate lanes for bicycles on busy roads.",
        "A {bikepart} that squeaks usually just needs cleaning and a little grease.",
        "Long rides go better with water, a snack and a small repair kit.",
        "A helmet that fits snugly offers much better protection than a loose one.",
        "Gentle braking with both levers keeps a bicycle stable on a wet road.",
        "Mudguards keep spray off the rider when the roads are wet.",
        "Riding in a group lets cyclists take turns sheltering from the wind.",
    ],
    "birds": [
        "Many songbirds sing most loudly in the hour around dawn.",
        "Migrating birds use the sun, the stars and landmarks to find their way.",
        "The {bird} is often recognised by its call before it is seen.",
        "Feeders filled with seeds attract birds to gardens in the colder months.",
        "Young birds leave the nest before they can fly well and are fed on the ground.",
        "Woodpeckers drum on trees both to find food and to mark their territory.",
        "Birdwatchers keep notebooks of the species they spot on each walk.",
        "Feathers are replaced gradually, so most birds can still fly while moulting.",
        "Some birds build nests from moss, twigs and even scraps of string.",
        "Gulls often follow ploughs and fishing boats in search of an easy meal.",
        "Owls hunt mostly at night, using sharp hearing to find their prey.",
        "Bright plumage helps many male birds attract a mate in spring.",
    ],
    "knitting": [
        "Knitting needles come in many sizes, and thicker needles make looser stitches.",
        "Wool keeps its warmth even when it is slightly damp.",
        "A swatch knitted before starting shows how large the finished piece will be.",
        "Cables are made by crossing groups of stitches over one another.",
        "Many knitters find the steady rhythm of the needles relaxing.",
        "Dropped stitches can be picked up again with a small crochet hook.",
        "{yarn} yarn is popular for scarves because it is soft against the skin.",
        "Blocking a finished garment evens out the stitches and sets its shape.",
        "Circular needles make it easy to knit a seamless hat or sleeve.",
        "Patterns use short abbreviations so instructions fit neatly on a page.",
        "Mixing two colours of yarn can produce striking striped or checked designs.",
        "Hand-knitted socks are prized for their warmth on cold mornings.",
    ],
    "rocks": [
        "Granite forms when molten rock cools slowly deep underground.",
        "Sandstone is made of grains of sand pressed and cemented together.",
        "Fossils are most often found in layered rocks such as limestone and shale.",
        "Streams smooth stones over time by tumbling them against one another.",
        "The {mineral} is prized by collectors for its colour and shine.",
        "Volcanic rock can be so full of air bubbles that it floats on water.",
        "Marble is limestone that has been changed by heat and pressure.",
        "Geologists read rock layers like the pages of a very old book.",
        "Pebbles on a beach come in many colours depending on where they formed.",
        "Crystals grow slowly as minerals settle out of water seeping through rock.",
        "Slate splits into thin, flat sheets that have long been used for roofs.",
        "Caves form where slightly acidic water dissolves limestone over long ages.",
    ],
}
_OFFTOPIC_SLOTS = {
    "cloudshape": ["puffy", "streaky", "lumpy", "layered", "patchy"],
    "pastry": ["pie", "tart", "biscuit", "scone"],
    "celestial": ["band of the galaxy", "faint glow of distant nebulae", "evening star"],
    "flower": ["marigolds", "lavender", "borage", "nasturtiums"],
    "teaextra": ["milk", "lemon", "honey", "a sprig of mint"],
    "chessidea": ["fork", "pin", "skewer", "discovered attack"],
    "bikepart": ["brake", "pedal", "chain", "saddle"],
    "bird": ["wren", "cuckoo", "blackbird", "robin", "owl"],
    "yarn": ["Merino", "Alpaca", "Cashmere", "Cotton"],
    "mineral": ["amethyst", "quartz crystal", "garnet", "malachite"],
    "season": ["spring", "summer", "autumn", "winter"],
}

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _fill_slots(template: str, slots: dict[str, list[str]], rng: random.Random,
                fixed: dict[str, str]) -> str:
    def sub(m: re.Match) -> str:
        key = m.group(1)
        return fixed[key] if key in fixed else rng.choice(slots[key])

    return _SLOT_RE.sub(sub, template)


def _overlaps(a: str, b: str) -> bool:
    """True if either name occurs as whole words inside the other."""
    def inside(x: str, y: str) -> bool:
        return re.search(r"\b" + re.escape(x) + r"\b", y) is not None
    return inside(a, b) or inside(b, a)


def contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word, case-insensitive containment."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text, re.I) is not None


def _sentence_case(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- world


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    name: str


@dataclass(frozen=True)
class Fact:
    id: str
    subject: str  # entity id
    relation: str
    value: str  # rendered value string
    object_id: str | None = None  # entity id when the value is an entity


@dataclass
class Paragraph:
    id: str
    text: str
    role: str  # "gold" | "filler" | "conflict"
    tokens: int
    kind: str  # "fact" | "offtopic"
    subject: str | None = None
    relation: str | None = None
    value: str | None = None

    def to_dict(self) -> dict:
        d = {"id": self.id, "text": self.text, "tokens": self.tokens, "kind": self.kind}
        if self.kind == "fact":
            d.update(subject=self.subject, relation=self.relation, value=self.value)
        return d


class World:
    """A seeded fictional world plus paragraph rendering."""

    YEAR_RANGE = (1600, 1999)

    def __init__(self, seed: int, people: int, towns: int, companies: int, inventions: int):
        self.seed = seed
        rng = rng_for(seed, "world")
        self.entities: dict[str, Entity] = {}
        self._used_names: set[str] = set()
        self._used_years: set[int] = set()
        self._rng = rng

        towns_ = [self._add("town", self._town_name, i) for i in range(towns)]
        companies_ = [self._add("company", self._company_name, i) for i in range(companies)]
        people_ = [self._add("person", self._person_name, i) for i in range(people)]
        inventions_ = [self._add("invention", self._invention_name, i)
                       for i in range(inventions)]

        facts: list[Fact] = []

        def add(subj: Entity, rel: str, value: str, obj: Entity | None = None) -> None:
            facts.append(Fact(f"{subj.id}.{rel}", subj.id, rel, value,
                              obj.id if obj else None))

        for p in people_:
            t = rng.choice(towns_)
            c = rng.choice(companies_)
            add(p, "born_in", t.name, t)
            add(p, "works_for", c.name, c)
            add(p, "birth_year", self.fresh_year(rng))
        for t in towns_:
            add(t, "town_founded", self.fresh_year(rng))
            add(t, "river", self.fresh_name(rng, "river"))
        for c in companies_:
            p = rng.choice(people_)
            t = rng.choice(towns_)
            add(c, "founded_by", p.name, p)
            add(c, "hq_in", t.name, t)
            add(c, "company_founded", self.fresh_year(rng))
        for inv in inventions_:
            p = rng.choice(people_)
            c = rng.choice(companies_)
            add(inv, "invented_by", p.name, p)
            add(inv, "made_by", c.name, c)
            add(inv, "invention_year", self.fresh_year(rng))
        self.facts: dict[str, Fact] = {f.id: f for f in facts}

    # ---- names and values
    def _person_name(self, rng: random.Random) -> str:
        return f"{_word(rng, 2)} {_word(rng, rng.choice([2, 3]))}"

    def _town_name(self, rng: random.Random) -> str:
        return _word(rng, 1) + rng.choice(_TOWN_ENDINGS)

    def _company_name(self, rng: random.Random) -> str:
        return f"{_word(rng, 2)} {rng.choice(_COMPANY_KINDS)}"

    def _invention_name(self, rng: random.Random) -> str:
        return f"{_word(rng, 2)} {rng.choice(_DEVICES)}"

    def _river_name(self, rng: random.Random) -> str:
        return _word(rng, 2)

    def _unique(self, make, rng: random.Random) -> str:
        for _ in range(10_000):
            name = make(rng)
            if len(name) >= 4 and not any(_overlaps(name, u) for u in self._used_names):
                self._used_names.add(name)
                return name
        raise RuntimeError("could not generate a unique name")

    def _add(self, etype: str, make, i: int) -> Entity:
        e = Entity(f"{etype}_{i:03d}", etype, self._unique(make, self._rng))
        self.entities[e.id] = e
        return e

    def fresh_year(self, rng: random.Random) -> str:
        lo, hi = self.YEAR_RANGE
        free = [y for y in range(lo, hi + 1) if y not in self._used_years]
        if not free:
            raise RuntimeError("ran out of unique years")
        y = rng.choice(free)
        self._used_years.add(y)
        return str(y)

    def fresh_name(self, rng: random.Random, kind: str) -> str:
        make = {"river": self._river_name, "town": self._town_name,
                "person": self._person_name, "company": self._company_name}[kind]
        return self._unique(make, rng)

    def fresh_value(self, kind: str, rng: random.Random) -> str:
        """A value of the given kind that appears nowhere else in the world."""
        return self.fresh_year(rng) if kind == "year" else self.fresh_name(rng, kind)

    # ---- helpers
    def name(self, entity_id: str) -> str:
        return self.entities[entity_id].name

    def all_names(self) -> set[str]:
        return set(self._used_names)

    def question_text(self, fact: Fact, variant: int) -> str:
        rel = RELATIONS[fact.relation]
        return rel.questions[variant % len(rel.questions)].format(s=self.name(fact.subject))

    def fact_sentence(self, subject_id: str, relation: str, value: str, variant: int) -> str:
        rel = RELATIONS[relation]
        return _sentence_case(rel.templates[variant % len(rel.templates)]
                              .format(s=self.name(subject_id), v=value))

    # ---- paragraphs
    def render_fact_paragraph(self, pid: str, subject_id: str, relation: str, value: str,
                              variant: int, counter: TokenCounter, lo: int, hi: int,
                              role: str = "filler") -> Paragraph:
        """Fact sentence + neutral sentences about the subject, length in [lo, hi] tokens.

        Fully determined by (world seed, pid), so the same paragraph id always yields the
        same text.
        """
        rng = rng_for(self.seed, "para", pid)
        subject = self.entities[subject_id]
        lead = self.fact_sentence(subject_id, relation, value, variant)
        pool_t = _NEUTRAL[subject.type]

        def neutral(r: random.Random) -> list[str]:
            out = [_sentence_case(_fill_slots(t, _SLOTS, r, {"n": subject.name}))
                   for t in pool_t]
            r.shuffle(out)
            return out

        text, n = _compose(lead, neutral, counter, lo, hi, rng)
        return Paragraph(pid, text, role, n, "fact", subject_id, relation, value)

    def offtopic_pool(self, size: int, counter: TokenCounter, lo: int, hi: int) -> list[Paragraph]:
        """`size` distinct off-topic paragraphs, cycling through the topics."""
        topics = sorted(_OFFTOPIC)
        out: list[Paragraph] = []
        seen: set[str] = set()
        i = 0
        while len(out) < size:
            if i > size * 50:
                raise RuntimeError("could not build enough distinct off-topic paragraphs")
            topic = topics[i % len(topics)]
            rng = rng_for(self.seed, "offtopic", i)
            i += 1

            templates = list(_OFFTOPIC[topic])
            lead = _fill_slots(templates.pop(rng.randrange(len(templates))),
                               _OFFTOPIC_SLOTS, rng, {})

            def rest(r: random.Random, templates=templates) -> list[str]:
                out = [_fill_slots(t, _OFFTOPIC_SLOTS, r, {}) for t in templates]
                r.shuffle(out)
                return out

            text, n = _compose(lead, rest, counter, lo, hi, rng, lead_first=True)
            if text in seen:
                continue
            seen.add(text)
            out.append(Paragraph(f"o:{len(out):04d}", text, "filler", n, "offtopic"))
        return out


def _compose(lead: str, pool_fn, counter: TokenCounter, lo: int, hi: int,
             rng: random.Random, lead_first: bool = False, attempts: int = 200) -> tuple[str, int]:
    """Greedily add pool sentences around `lead` until the token count lands in [lo, hi]."""
    for _ in range(attempts):
        lead_pos = 0 if lead_first else rng.choice([0, 0, 1])
        chosen: list[str] = []
        for s in pool_fn(rng):
            trial = chosen + [s]
            parts = trial[:lead_pos] + [lead] + trial[lead_pos:]
            text = " ".join(parts)
            n = counter.count(text)
            if n <= hi:
                chosen = trial
                if n >= lo:
                    return text, n
        # Not long enough or overshot: try another ordering.
    raise RuntimeError(f"could not compose a paragraph of {lo}-{hi} tokens around: {lead!r}")
