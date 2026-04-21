"""A 100-document toy corpus + 10 QA pairs for smoke testing.

Layout: 10 topical categories × 10 passages each. The first passage in each
category is the *gold* passage for that category's QA pair; the other nine
are same-topic distractors that BGE will rank similarly without training,
so improvements in retrieval recall are meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToyPassage:
    id: str
    text: str


@dataclass(frozen=True)
class ToyQA:
    question: str
    answer: str
    gold_doc_id: str


_CORPUS_DATA: list[tuple[str, list[str]]] = [
    ("capital", [
        "Paris is the capital and largest city of France, situated on the river Seine.",
        "Berlin is the capital of Germany and its largest city by population.",
        "Madrid is the capital of Spain, located in the center of the Iberian peninsula.",
        "Rome is the capital of Italy and a major Mediterranean cultural hub.",
        "Lisbon is the capital of Portugal, on the Atlantic coast at the mouth of the Tagus.",
        "Warsaw is the capital of Poland, situated on the Vistula river.",
        "Vienna is the capital of Austria and historically the seat of the Habsburg empire.",
        "Athens is the capital of Greece, with a history spanning over three thousand years.",
        "Helsinki is the capital of Finland on the southern coast of the Gulf of Finland.",
        "Oslo is the capital of Norway at the head of the Oslofjord in the country's southeast.",
    ]),
    ("inventor", [
        "Alexander Graham Bell is credited with inventing the first practical telephone in 1876.",
        "Thomas Edison is famous for inventing the phonograph and the long-lasting electric lightbulb.",
        "Nikola Tesla pioneered the alternating current electrical supply system.",
        "Karl Benz built the first practical automobile powered by an internal combustion engine in 1885.",
        "The Wright brothers, Orville and Wilbur, invented and flew the first successful airplane in 1903.",
        "Tim Berners-Lee invented the World Wide Web at CERN in 1989.",
        "Guglielmo Marconi developed the first practical long-distance radio transmission system.",
        "James Watt made fundamental improvements to the steam engine that drove the Industrial Revolution.",
        "Johannes Gutenberg invented the movable-type printing press in the mid-15th century.",
        "George Eastman invented roll film and made photography accessible to the masses with the Kodak camera.",
    ]),
    ("author", [
        "William Shakespeare wrote the tragedy Hamlet around 1600 in early modern England.",
        "Jane Austen wrote Pride and Prejudice, published in 1813, a cornerstone of English literature.",
        "Leo Tolstoy authored the epic novel War and Peace, set during the Napoleonic era.",
        "Fyodor Dostoevsky wrote Crime and Punishment, exploring guilt and redemption in 19th-century Russia.",
        "Victor Hugo penned Les Miserables, a sweeping novel of post-revolutionary France.",
        "Gabriel Garcia Marquez wrote One Hundred Years of Solitude, a landmark of magical realism.",
        "Mark Twain authored The Adventures of Huckleberry Finn, a classic of American fiction.",
        "Virginia Woolf wrote Mrs Dalloway, a modernist novel set over a single day in London.",
        "James Joyce wrote Ulysses, an experimental novel published in 1922.",
        "Toni Morrison wrote Beloved, exploring the legacy of slavery in the United States.",
    ]),
    ("physics", [
        "Albert Einstein developed the theory of relativity, fundamentally changing our understanding of space and time.",
        "Isaac Newton formulated the three laws of motion and the law of universal gravitation.",
        "Max Planck originated quantum theory by quantizing the energy of electromagnetic radiation.",
        "Niels Bohr proposed the model of the atom in which electrons occupy discrete energy levels.",
        "Werner Heisenberg formulated the uncertainty principle in quantum mechanics.",
        "Erwin Schrödinger formulated the wave equation that bears his name.",
        "Marie Curie pioneered research on radioactivity and discovered the elements polonium and radium.",
        "Richard Feynman developed the path-integral formulation of quantum mechanics.",
        "Stephen Hawking proposed that black holes emit thermal radiation, now known as Hawking radiation.",
        "Paul Dirac unified quantum mechanics with special relativity in his Dirac equation.",
    ]),
    ("mountain", [
        "Mount Everest is the tallest mountain on Earth, reaching 8,849 meters above sea level.",
        "K2 is the second-highest mountain at 8,611 meters, located on the China-Pakistan border.",
        "Kangchenjunga, on the India-Nepal border, is the third-highest mountain at 8,586 meters.",
        "Denali in Alaska is the tallest mountain in North America, at 6,190 meters.",
        "Aconcagua in the Andes is the tallest mountain in South America at 6,961 meters.",
        "Mount Kilimanjaro is the highest free-standing mountain in the world, located in Tanzania.",
        "Mount Elbrus in Russia is the highest peak in Europe at 5,642 meters.",
        "Mount Vinson is the tallest mountain in Antarctica, in the Sentinel Range.",
        "Mont Blanc is the highest peak in the Alps, on the border of France and Italy.",
        "Mount Fuji is the tallest mountain in Japan, an iconic stratovolcano southwest of Tokyo.",
    ]),
    ("history", [
        "World War 2 ended in 1945 with the surrender of Germany in May and Japan in September.",
        "The French Revolution began in 1789 with the storming of the Bastille in Paris.",
        "The American Civil War took place from 1861 to 1865 between the Union and Confederate states.",
        "The Berlin Wall fell in 1989, marking a symbolic end to the Cold War divide of Europe.",
        "Christopher Columbus reached the Americas in 1492 while seeking a westward route to Asia.",
        "The Magna Carta was signed by King John of England in 1215, limiting royal power.",
        "The Roman Empire fell in 476 CE when the last western emperor was deposed.",
        "The Industrial Revolution began in Britain in the late 18th century with mechanized textile production.",
        "Apollo 11 landed on the Moon in July 1969, with Neil Armstrong as the first human to walk on its surface.",
        "The Treaty of Versailles, signed in 1919, formally ended the First World War.",
    ]),
    ("language", [
        "Python was created by Guido van Rossum and first released in 1991, emphasizing code readability.",
        "JavaScript was created by Brendan Eich at Netscape in 1995 for client-side web scripting.",
        "C was developed by Dennis Ritchie at Bell Labs in the early 1970s, originally for Unix.",
        "Java was developed at Sun Microsystems by James Gosling and released in 1995.",
        "Ruby was created by Yukihiro Matsumoto and first released publicly in 1995.",
        "Rust was developed at Mozilla by Graydon Hoare and emphasizes memory safety without garbage collection.",
        "Go was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson, released in 2009.",
        "Haskell is a purely functional programming language standardized in the late 1990s.",
        "Lisp is one of the oldest programming language families, originally specified by John McCarthy in 1958.",
        "Swift was developed by Apple and unveiled in 2014 as a modern replacement for Objective-C.",
    ]),
    ("biology", [
        "The blue whale is the largest mammal on Earth, reaching over 30 meters in length and 200 tonnes.",
        "African elephants are the largest land mammals, with bulls weighing up to 6,000 kilograms.",
        "Giraffes are the tallest land animals, using their long necks to browse the leaves of acacia trees.",
        "Cheetahs are the fastest land animals, capable of running over 100 kilometers per hour in short bursts.",
        "Sperm whales have the largest brains of any animal, weighing about 8 kilograms.",
        "Honey bees pollinate a large fraction of the crops humans rely on for food production.",
        "Octopuses have three hearts and blue copper-based blood instead of iron-based hemoglobin.",
        "Tardigrades, also called water bears, can survive extreme conditions including the vacuum of space.",
        "Mitochondria generate most of a cell's energy and are often called the powerhouse of the cell.",
        "DNA was identified as the molecule of heredity through experiments by Avery, MacLeod, and McCarty in 1944.",
    ]),
    ("cuisine", [
        "Sushi originated in Japan and combines vinegared rice with fish, vegetables, or other ingredients.",
        "Pizza originated in Naples, Italy, and has become one of the most popular foods worldwide.",
        "Pasta is a staple of Italian cuisine, with hundreds of regional shapes and sauces.",
        "Tacos are a traditional Mexican dish consisting of a corn or wheat tortilla folded around a filling.",
        "Curry refers to a wide variety of spice-based dishes originating on the Indian subcontinent.",
        "Croissants are a buttery, flaky pastry of Austrian origin that became iconic in French baking.",
        "Pad thai is a stir-fried rice noodle dish that is one of the national dishes of Thailand.",
        "Borscht is a beet-based soup that is a staple of Ukrainian and Russian cuisine.",
        "Paella is a Spanish rice dish from Valencia traditionally cooked in a wide shallow pan.",
        "Kimchi is a traditional Korean side dish of fermented vegetables, most commonly napa cabbage.",
    ]),
    ("astronomy", [
        "The Sun is the closest star to Earth, about 150 million kilometers away on average.",
        "Proxima Centauri is the closest star beyond the Sun, about 4.24 light-years away.",
        "Jupiter is the largest planet in our solar system, more massive than all other planets combined.",
        "Saturn is famous for its prominent ring system composed of ice and rock particles.",
        "The Moon is the only natural satellite of Earth and the fifth-largest in the solar system.",
        "Mars is often called the red planet because of iron oxide on its surface.",
        "The Milky Way is the spiral galaxy that contains the Sun and our solar system.",
        "Andromeda is the nearest large galaxy to the Milky Way at about 2.5 million light-years.",
        "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
        "The cosmic microwave background is residual radiation from the hot early universe, about 13.8 billion years old.",
    ]),
]


def _build_corpus() -> tuple[list[ToyPassage], list[ToyQA]]:
    passages: list[ToyPassage] = []
    qas: list[ToyQA] = []
    qa_seeds = [
        ("What is the capital of France?", "Paris"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("Who wrote Hamlet?", "William Shakespeare"),
        ("Who developed the theory of relativity?", "Albert Einstein"),
        ("What is the tallest mountain on Earth?", "Mount Everest"),
        ("When did World War 2 end?", "1945"),
        ("Who created the Python programming language?", "Guido van Rossum"),
        ("What is the largest mammal on Earth?", "blue whale"),
        ("Where did sushi originate?", "Japan"),
        ("What is the closest star to Earth?", "the Sun"),
    ]
    for cat_idx, (cat_name, texts) in enumerate(_CORPUS_DATA):
        for j, text in enumerate(texts):
            passages.append(ToyPassage(id=f"{cat_name}-{j:02d}", text=text))
        question, answer = qa_seeds[cat_idx]
        qas.append(ToyQA(
            question=question,
            answer=answer,
            gold_doc_id=f"{cat_name}-00",
        ))
    return passages, qas


PASSAGES, QA_PAIRS = _build_corpus()
assert len(PASSAGES) == 100
assert len(QA_PAIRS) == 10
