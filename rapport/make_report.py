# -*- coding: utf-8 -*-
"""Rapport PDF (version longue, mise en page noir et blanc, figures en couleur) du projet NLP SSB.

Usage, depuis la racine du dépôt (Windows, polices Times New Roman) :
    python rapport/make_report.py rapport/Rapport_Projet_NLP_SSB.pdf
"""
import sys
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image,
                                KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "fr"
LOGO = ROOT / "rapport" / "logo_ensiasd.png"
OUT = Path(sys.argv[1])

# ------------------------------------------------------------------ données
M = pd.read_parquet(ROOT / "results/main/metrics.parquet")
LB = pd.read_csv(ROOT / "results/main/leaderboard.csv")


def g(model, curve, n, metric="score", col="value"):
    r = M[(M.model == model) & (M.curve == curve) & (M.N == n) & (M.metric == metric)]
    return float(r[col].iloc[0]) if len(r) else float("nan")


def pct(x, d=1):
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def num(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


CLS = ["tfidf-lr", "bert-truncate", "bert-chunk-mean"]
RET = ["tfidf", "bm25", "minilm"]
QA = ["distilbert-squad", "roberta-squad2"]
ALL = CLS + RET + QA
TASK_OF = {**{m: "Classification" for m in CLS}, **{m: "Recherche" for m in RET},
           **{m: "QA" for m in QA}}

# ------------------------------------------------------------------ polices et styles
F = r"C:\Windows\Fonts"
for name, file in [("Times", "times.ttf"), ("Times-B", "timesbd.ttf"), ("Times-I", "timesi.ttf"),
                   ("Times-BI", "timesbi.ttf"), ("Sym", "seguisym.ttf"),
                   ("Mono", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(name, f"{F}\\{file}"))
registerFontFamily("Times", normal="Times", bold="Times-B", italic="Times-I",
                   boldItalic="Times-BI")
IN = "<font name='Sym'>∈</font>"

BLACK = colors.black
GREY = colors.HexColor("#444444")
PAGE_W, PAGE_H = A4
MX = 2.5 * cm
TEXT_W = PAGE_W - 2 * MX

body = ParagraphStyle("body", fontName="Times", fontSize=11.5, leading=16, alignment=TA_JUSTIFY,
                      textColor=BLACK, spaceAfter=8, firstLineIndent=0.8 * cm)
body0 = ParagraphStyle("body0", parent=body, firstLineIndent=0)
chap_kicker = ParagraphStyle("kick", fontName="Times", fontSize=13, leading=16,
                             textColor=GREY, alignment=TA_LEFT, spaceAfter=4)
h0 = ParagraphStyle("h0", fontName="Times-B", fontSize=22, leading=27, textColor=BLACK,
                    spaceAfter=6, keepWithNext=1)
h1 = ParagraphStyle("h1", fontName="Times-B", fontSize=14.5, leading=19, textColor=BLACK,
                    spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Times-BI", fontSize=12.5, leading=17, textColor=BLACK,
                    spaceBefore=8, spaceAfter=4, keepWithNext=1)
cap = ParagraphStyle("cap", fontName="Times", fontSize=10.5, leading=13.5, textColor=BLACK,
                     alignment=TA_CENTER, spaceBefore=4, spaceAfter=14)
tcap = ParagraphStyle("tcap", parent=cap, spaceBefore=8, spaceAfter=5)
cell = ParagraphStyle("cell", fontName="Times", fontSize=10.5, leading=13, textColor=BLACK)
cellc = ParagraphStyle("cellc", parent=cell, alignment=TA_CENTER)
cellh = ParagraphStyle("cellh", fontName="Times-B", fontSize=10.5, leading=13,
                       textColor=BLACK)
cellhc = ParagraphStyle("cellhc", parent=cellh, alignment=TA_CENTER)
eqs = ParagraphStyle("eq", fontName="Times-I", fontSize=12, leading=17, alignment=TA_CENTER)
eqn = ParagraphStyle("eqn", fontName="Times", fontSize=12, leading=17, alignment=TA_LEFT)
mono = ParagraphStyle("mono", fontName="Mono", fontSize=8.6, leading=11, textColor=BLACK)
quote = ParagraphStyle("quote", parent=body0, fontSize=11, leading=15.5, leftIndent=1 * cm,
                       rightIndent=1 * cm, fontName="Times-I")


def roman(n):
    out = ""
    for v, s in [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]:
        while n >= v:
            out += s
            n -= v
    return out


class StartArabic(Flowable):
    def wrap(self, *a):
        return 0, 0

    def draw(self):
        self.canv._doctemplate.arabic_start = self.canv.getPageNumber()


class Chapter(Flowable):
    """Marque le titre courant pour l'en-tête."""

    def __init__(self, title):
        super().__init__()
        self.title = title

    def wrap(self, *a):
        return 0, 0

    def draw(self):
        self.canv._doctemplate.running = self.title


class ListOf(TableOfContents):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.dotsMinLevel = 0
        self.levelStyles = [ParagraphStyle("lof", fontName="Times", fontSize=11.5, leading=16,
                                           leftIndent=0, firstLineIndent=0)]

    def notify(self, kind, stuff):
        if kind == self.kind:
            self.addEntry(*stuff)


class Doc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(str(path), pagesize=A4, leftMargin=MX, rightMargin=MX,
                         topMargin=2.6 * cm, bottomMargin=2.4 * cm,
                         title="Semantic Saturation Benchmark - Rapport de projet NLP",
                         author="Y. Bouachrine, A. Nahimy, M. Khacha, A. Hirri",
                         subject="Projet NLP - ENSIASD Taroudant")
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [fr]),
                               PageTemplate("body", [fr], onPageEnd=self._decor)])
        self.arabic_start = None
        self.running = ""
        self._seq = 0

    def beforeDocument(self):
        self._seq = 0
        self.running = ""

    def label(self, page):
        if self.arabic_start is None or page < self.arabic_start:
            return roman(page)
        return str(page - self.arabic_start + 1)

    def afterFlowable(self, fl):
        if not isinstance(fl, Paragraph):
            return
        name = fl.style.name
        kinds = {"h0": ("TOCEntry", 0), "h1": ("TOCEntry", 1), "h2": ("TOCEntry", 2),
                 "cap": ("LOF", 0), "tcap": ("LOT", 0)}
        if name not in kinds:
            return
        kind, level = kinds[name]
        text = fl.getPlainText()
        if name == "h0" and getattr(fl, "_toc", None):
            text = fl._toc
        if name in ("cap", "tcap"):
            text = text.replace(" — ", " : ", 1)
            text = text if len(text) < 95 else text[:92].rsplit(" ", 1)[0] + "…"
        self._seq += 1
        key = f"k{self._seq}"
        self.canv.bookmarkPage(key)
        if kind == "TOCEntry":
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            if level == 2:
                return
        self.notify(kind, (level, text, self.page, key))

    def _decor(self, c, d):
        c.saveState()
        c.setStrokeColor(BLACK)
        c.setLineWidth(0.5)
        y = PAGE_H - 1.75 * cm
        c.setFont("Times-I", 9.5)
        c.setFillColor(BLACK)
        c.drawString(MX, y + 4, "Semantic Saturation Benchmark")
        c.drawRightString(PAGE_W - MX, y + 4, self.running)
        c.line(MX, y, PAGE_W - MX, y)
        c.setFont("Times", 10.5)
        c.drawCentredString(PAGE_W / 2, 1.3 * cm, self.label(c.getPageNumber()))
        c.restoreState()


# ------------------------------------------------------------------ helpers de contenu
story = []


def P(t, s=body):
    story.append(Paragraph(t, s))


def P0(t):
    story.append(Paragraph(t, body0))


def front(title):
    story.append(PageBreak())
    story.append(Chapter(title))
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph(title, h0))
    rule()
    story.append(Spacer(1, 0.4 * cm))


def chapter(n, title):
    story.append(PageBreak())
    story.append(Chapter(f"Chapitre {n}. {title}"))
    story.append(Spacer(1, 1.6 * cm))
    story.append(Paragraph(f"Chapitre {n}", chap_kicker))
    p = Paragraph(title, h0)
    p._toc = f"Chapitre {n}. {title}"
    story.append(p)
    rule()
    story.append(Spacer(1, 0.5 * cm))


def rule(w=TEXT_W, t=0.8):
    story.append(Table([[""]], colWidths=[w], rowHeights=[t],
                       style=[("LINEABOVE", (0, 0), (-1, -1), t, BLACK)], hAlign="LEFT"))


def H1(t):
    story.append(CondPageBreak(3 * cm))
    story.append(Paragraph(t, h1))


def H2(t):
    story.append(CondPageBreak(2.5 * cm))
    story.append(Paragraph(t, h2))


figno, tabno, eqno = [0], [0], [0]


def figure(name, text, width=TEXT_W):
    figno[0] += 1
    w, h = PILImage.open(FIG / name).size
    img = Image(str(FIG / name), width=width, height=width * h / w)
    story.append(KeepTogether([Spacer(1, 4), img,
                               Paragraph(f"<b>Figure {figno[0]}</b> — {text}", cap)]))
    return figno[0]


def table(rows, widths, title, center_from=1, font=None):
    tabno[0] += 1
    hs, cs = (cellh, cellhc), (cell, cellc)
    data = [[Paragraph(c, hs[j >= center_from]) for j, c in enumerate(rows[0])]]
    data += [[Paragraph(c, cs[j >= center_from]) for j, c in enumerate(r)] for r in rows[1:]]
    t = Table(data, colWidths=[w * TEXT_W for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, BLACK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, BLACK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, BLACK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    story.append(KeepTogether([Paragraph(f"<b>Tableau {tabno[0]}</b> — {title}", tcap), t,
                               Spacer(1, 12)]))
    return tabno[0]


def eq(formula):
    eqno[0] += 1
    t = Table([["", Paragraph(formula, eqs), Paragraph(f"({eqno[0]})", eqn)]],
              colWidths=[0.08 * TEXT_W, 0.84 * TEXT_W, 0.08 * TEXT_W],
              style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                     ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)])
    story.append(t)
    return eqno[0]


# ================================================================== PAGE DE GARDE
lw, lh = PILImage.open(LOGO).size
story.append(Image(str(LOGO), width=TEXT_W, height=TEXT_W * lh / lw))
story.append(Spacer(1, 2.2 * cm))
cv = lambda **k: ParagraphStyle("cv", alignment=TA_CENTER, textColor=BLACK, **k)
story.append(Paragraph("Rapport de projet", cv(fontName="Times", fontSize=15, leading=19)))
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Module : Traitement Automatique du Langage Naturel",
                       cv(fontName="Times-I", fontSize=13, leading=17)))
story.append(Spacer(1, 1.0 * cm))
rule(t=1.4)
story.append(Spacer(1, 0.45 * cm))
story.append(Paragraph("SEMANTIC SATURATION BENCHMARK",
                       cv(fontName="Times-B", fontSize=25, leading=31)))
story.append(Spacer(1, 0.25 * cm))
story.append(Paragraph("Quand le contexte cesse d'aider : mesure contrôlée de la saturation "
                       "sémantique des modèles NLP classiques",
                       cv(fontName="Times-I", fontSize=14.5, leading=20)))
story.append(Spacer(1, 0.45 * cm))
rule(t=1.4)
story.append(Spacer(1, 2.6 * cm))
lab = cv(fontName="Times-B", fontSize=12.5, leading=17)
nam = cv(fontName="Times", fontSize=13, leading=19)
story.append(Table(
    [[Paragraph("Réalisé par :", lab), Paragraph("Enseignant du module :", lab)],
     [Paragraph("M. Yassine BOUACHRINE<br/>M. Abdeljalil NAHIMY<br/>M. Mohamed KHACHA<br/>"
                "M. Abdelali HIRRI", nam), Paragraph("Dr. Adil BAQACH", nam)]],
    colWidths=[TEXT_W / 2] * 2, style=[("VALIGN", (0, 0), (-1, -1), "TOP")]))
story.append(Spacer(1, 3.0 * cm))
rule(t=0.5)
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Année universitaire 2026 – 2027",
                       cv(fontName="Times", fontSize=12.5, leading=16)))
story.append(NextPageTemplate("body"))

# ================================================================== PAGES LIMINAIRES
front("Remerciements")
P("Avant de présenter ce travail, nous souhaitons remercier toutes les personnes qui ont "
  "contribué, de près ou de loin, à sa réalisation.")
P("Nous remercions tout particulièrement Dr. Adil Baqach, enseignant du module de Traitement "
  "Automatique du Langage Naturel, pour la qualité de son enseignement et la clarté de ses "
  "explications. Il nous a donné les bases théoriques et pratiques sans lesquelles ce projet "
  "n'aurait pas été possible. Ses conseils, sa disponibilité et ses remarques nous ont aidés à "
  "poser les bonnes questions, à adopter une démarche expérimentale rigoureuse et à prendre du "
  "recul sur nos résultats.")
P("Nous remercions aussi l'ensemble du corps enseignant et administratif de l'École Nationale "
  "Supérieure de l'Intelligence Artificielle et Sciences des Données de Taroudant, ainsi que "
  "l'Université Ibn Zohr. Leur engagement nous offre chaque jour un cadre de formation exigeant "
  "et stimulant.")
P("Enfin, nous pensons à la communauté scientifique et open source. Les jeux de données "
  "(AG News, SQuAD 2.0), les bibliothèques et les modèles pré-entraînés qu'elle met librement à "
  "disposition ont rendu ce banc d'essai possible.")

front("Résumé")
P("Fournir davantage de contexte à un modèle de traitement automatique du langage est souvent "
  "présenté comme un moyen sûr d'améliorer ses prédictions. Ce projet soumet cette intuition à "
  "une vérification expérimentale contrôlée. Nous proposons le <i>Semantic Saturation "
  "Benchmark</i> (SSB), un banc d'essai dans lequel un paragraphe porteur de l'information "
  "utile, appelé signal, reste strictement identique, tandis que le contexte qui l'entoure croît "
  "jusqu'à cent paragraphes, soit environ douze mille tokens. Cinq facteurs sont manipulés "
  "indépendamment : la dilution, le bruit sémantique, la redondance, la contradiction et la "
  "position du signal. Huit modèles classiques, du TF-IDF aux encodeurs de la famille BERT, sont "
  "évalués sur trois tâches : la classification thématique AG News, la recherche de passage et "
  "la question-réponse extractive sur SQuAD 2.0.")
P("Sur l'exécution complète, qui porte sur 25 950 échantillons, aucun modèle ne bénéficie d'un "
  "contexte plus long : les huit se dégradent significativement entre un et cent paragraphes. "
  "La classification s'effondre au niveau du hasard, de 96–98 % à 24–28 % de précision, parce "
  "que les classifieurs basculent vers la classe <i>Sci/Tech</i> tout en conservant une "
  "confiance quasi totale. La recherche et la question-réponse résistent bien mieux à la "
  "dilution, avec des pertes de quatre à douze points, mais un paragraphe contradictoire presque "
  "identique au signal les trompe dans 27 à 39 % des cas, et le premier paragraphe lu tend à "
  "l'emporter. Sur les huit hypothèses formulées avant l'expérience, deux sont confirmées, cinq "
  "le sont partiellement et une est rejetée.")
P0("<b>Mots-clés :</b> contexte long, robustesse, saturation sémantique, calibration, recherche "
   "d'information, question-réponse extractive, BM25, BERT, évaluation pré-enregistrée.")
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph("Abstract", h1))
P("Adding context to a natural language processing model is commonly assumed to improve its "
  "predictions. This project tests that assumption under controlled conditions. We introduce the "
  "Semantic Saturation Benchmark (SSB), in which a fixed evidence paragraph is surrounded by up "
  "to one hundred paragraphs (about twelve thousand tokens), while five factors are varied "
  "independently: dilution, semantic noise, redundancy, contradiction and position. Eight "
  "classical models are evaluated on AG News topic classification, SQuAD 2.0 passage retrieval "
  "and extractive question answering. More context helped none of them: all eight degrade "
  "significantly from one to one hundred paragraphs. Classifiers collapse to chance level by "
  "drifting towards the <i>Sci/Tech</i> class while remaining confident, whereas retrieval and "
  "question answering models resist dilution but are fooled by a near-duplicate conflicting "
  "paragraph in 27 to 39 % of cases.", ParagraphStyle("abs", parent=body, fontName="Times-I"))

front("Table des matières")
toc = TableOfContents()
toc.dotsMinLevel = 0
toc.levelStyles = [
    ParagraphStyle("t0", fontName="Times-B", fontSize=12, leading=19, leftIndent=0,
                   firstLineIndent=0, spaceBefore=4),
    ParagraphStyle("t1", fontName="Times", fontSize=11.5, leading=16, leftIndent=18,
                   firstLineIndent=0),
    ParagraphStyle("t2", fontName="Times-I", fontSize=11, leading=15, leftIndent=36,
                   firstLineIndent=0),
]
story.append(toc)

front("Liste des figures")
lof = ListOf("LOF")
story.append(lof)
story.append(Spacer(1, 1.2 * cm))
story.append(Paragraph("Liste des tableaux", h0))
rule()
story.append(Spacer(1, 0.4 * cm))
lot = ListOf("LOT")
story.append(lot)

front("Liste des abréviations")
abbr = [("AG News", "corpus d'articles de presse en quatre classes thématiques"),
        ("BERT", "Bidirectional Encoder Representations from Transformers"),
        ("BM25", "Best Matching 25, fonction de score probabiliste de recherche"),
        ("CLS", "token spécial de début de séquence des encodeurs BERT"),
        ("ECE", "Expected Calibration Error, erreur de calibration attendue"),
        ("EM", "Exact Match, correspondance exacte de la réponse"),
        ("IC", "intervalle de confiance"),
        ("MRR", "Mean Reciprocal Rank, rang réciproque moyen"),
        ("NER", "Named Entity Recognition, reconnaissance d'entités nommées"),
        ("NLP / TALN", "Natural Language Processing / traitement automatique du langage"),
        ("QA", "Question Answering, question-réponse"),
        ("R@k", "Recall@k, part des requêtes dont le passage pertinent est dans les k premiers"),
        ("SQuAD", "Stanford Question Answering Dataset"),
        ("SSB", "Semantic Saturation Benchmark"),
        ("SSS", "Semantic Saturation Score, aire normalisée sous la courbe de score"),
        ("TF-IDF", "Term Frequency – Inverse Document Frequency")]
t = Table([[Paragraph(f"<b>{a}</b>", cell), Paragraph(b, cell)] for a, b in abbr],
          colWidths=[0.22 * TEXT_W, 0.78 * TEXT_W],
          style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                 ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.8, BLACK),
                 ("LINEABOVE", (0, 0), (-1, 0), 0.8, BLACK)])
story.append(t)

# ================================================================== INTRODUCTION
story.append(PageBreak())
story.append(StartArabic())
story.append(Chapter("Introduction générale"))
story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph("Introduction générale", h0))
rule()
story.append(Spacer(1, 0.5 * cm))
P("Les systèmes de traitement automatique du langage naturel reçoivent des entrées de plus en "
  "plus longues. Un moteur de recherche concatène plusieurs documents avant de répondre, un "
  "assistant conserve l'historique complet d'une conversation, un classifieur de documents lit "
  "des rapports entiers plutôt que des titres. Dans tous ces cas, une hypothèse implicite guide "
  "la conception : fournir plus de texte ne peut qu'aider, puisque le modèle dispose alors de "
  "davantage d'informations pour décider.")
P("Cette hypothèse est pourtant fragile. Un texte supplémentaire n'est jamais neutre. Il peut "
  "diluer l'information utile au point de la rendre minoritaire dans la représentation du "
  "document ; il peut introduire des passages proches du sujet mais trompeurs ; il peut contenir "
  "une affirmation contradictoire ; il peut enfin repousser l'élément pertinent au-delà de ce que "
  "le modèle est physiquement capable de lire, la plupart des encodeurs Transformer étant limités "
  "à 512 tokens. Ces différents mécanismes sont généralement confondus dans les évaluations "
  "existantes, ce qui rend difficile l'identification de la cause d'une dégradation.")
P("Nous appelons <i>saturation sémantique</i> le point à partir duquel l'ajout de contexte "
  "cesse d'améliorer un modèle, puis commence à le rendre moins fiable. La problématique de ce "
  "projet peut se formuler ainsi : <i>à quel moment, et pour quelle raison, un contexte plus "
  "long cesse-t-il d'aider un modèle NLP classique, et comment cette dégradation se manifeste-t-"
  "elle sur la précision comme sur la fiabilité de ses prédictions ?</i> Pour y répondre, nous "
  "avons conçu un banc d'essai dans lequel un seul facteur varie à la fois, et dans lequel "
  "l'information nécessaire à la bonne réponse est garantie présente et intacte à chaque niveau "
  "de contexte.")
P("Nous avons volontairement restreint l'étude aux modèles NLP classiques : représentations "
  "TF-IDF, fonction de score BM25, encodeur de phrases dense et encodeurs Transformer de taille "
  "modeste, sans grand modèle de langage ni génération de texte. Ce choix présente deux "
  "avantages. Il permet d'abord d'isoler le comportement de représentations dont le "
  "fonctionnement est bien compris, et donc d'expliquer les résultats plutôt que de seulement les "
  "constater. Il rend ensuite l'expérience complète reproductible en moins d'une heure sur un "
  "seul GPU T4 gratuit, ce qui est compatible avec les moyens d'un projet de module.")
P("La contribution du projet est triple. Sur le plan des données, nous construisons un jeu de "
  "25 950 échantillons dans lequel le signal est vérifié octet par octet et où la réponse "
  "n'apparaît jamais dans le texte de remplissage. Sur le plan méthodologique, nous définissons "
  "un protocole statistique fondé sur des intervalles de confiance bootstrap appariés, une "
  "correction pour comparaisons multiples et huit hypothèses pré-enregistrées, qui permet de "
  "distinguer un effet réel d'une fluctuation d'échantillonnage. Sur le plan empirique, nous "
  "rapportons des résultats parfois contre-intuitifs, dont le plus marquant est l'effondrement "
  "systématique des classifieurs vers une seule classe, accompagné d'une confiance qui ne "
  "faiblit pas.")
P("Le rapport est organisé en six chapitres. Le premier présente le contexte théorique et les "
  "travaux connexes. Le deuxième décrit la conception du benchmark et la construction du jeu de "
  "données. Le troisième détaille les modèles évalués et leur implémentation. Le quatrième "
  "expose le protocole expérimental, les métriques et les hypothèses. Le cinquième présente les "
  "résultats axe par axe, et le sixième les discute, en précisant les limites de l'étude. Une "
  "conclusion générale résume les apports et ouvre des perspectives.")

# ================================================================== CHAPITRE 1
chapter(1, "Contexte et état de l'art")
P("Ce chapitre situe le projet. Nous rappelons d'abord pourquoi la longueur du contexte pose un "
  "problème spécifique aux modèles NLP, puis nous présentons les représentations de texte "
  "étudiées, avant de passer en revue les travaux qui ont déjà mis en évidence la sensibilité des "
  "modèles au contexte ajouté.")
H1("1.1 Le problème du contexte long")
P("Un modèle de traitement du langage ne lit pas un texte comme un lecteur humain, qui sait "
  "repérer le passage pertinent et ignorer le reste. Il transforme l'entrée en une "
  "représentation numérique, puis prend sa décision à partir de cette représentation. Dès lors, "
  "la question centrale est celle de l'agrégation : comment l'information répartie sur des "
  "milliers de tokens est-elle résumée ? Si la représentation est une moyenne ou une somme sur "
  "tout le texte, un passage court et pertinent pèse d'autant moins que le reste est long. Si la "
  "représentation est limitée à une fenêtre fixe, tout ce qui se trouve hors de cette fenêtre "
  "est tout simplement ignoré.")
P("Trois phénomènes distincts peuvent donc dégrader un modèle lorsque le contexte s'allonge. "
  "La <i>dilution</i> réduit la part du signal dans la représentation globale. La "
  "<i>distraction</i> introduit des passages qui ressemblent au signal et peuvent être pris pour "
  "lui. La <i>troncature</i> supprime physiquement une partie de l'entrée. À ces trois "
  "phénomènes s'ajoute la question de la <i>cohérence</i> : lorsqu'un contexte contient deux "
  "affirmations incompatibles, rien ne garantit que le modèle choisisse la bonne. Un banc "
  "d'essai utile doit permettre de séparer ces effets.")
H1("1.2 Les représentations de texte étudiées")
H2("1.2.1 Le modèle TF-IDF")
P("La pondération TF-IDF représente un document par un vecteur creux dont chaque dimension "
  "correspond à un terme du vocabulaire. Le poids d'un terme <i>t</i> dans un document <i>d</i> "
  "combine sa fréquence dans le document et sa rareté dans la collection :")
eq("w(t, d) = tf(t, d) · log( |D| / df(t) )")
P0("où <i>tf(t, d)</i> est le nombre d'occurrences de <i>t</i> dans <i>d</i>, |<i>D</i>| le "
   "nombre de documents et <i>df(t)</i> le nombre de documents contenant <i>t</i>. La "
   "similarité entre une requête et un document est le cosinus entre leurs vecteurs. Cette "
   "représentation ignore l'ordre des mots : elle est donc par construction insensible à la "
   "position du signal, mais très sensible à la proportion de termes qu'il apporte. Associée à "
   "une régression logistique, elle forme un classifieur simple et robuste.")
H2("1.2.2 La fonction de score BM25")
P("BM25 est la référence historique de la recherche d'information. Elle corrige deux défauts "
  "du TF-IDF : la croissance linéaire de l'importance d'un terme avec sa fréquence, et "
  "l'avantage donné aux documents longs. Pour une requête <i>q</i> et un document <i>d</i> :")
eq(f"score(q, d) = Σ<sub>t {IN} q</sub> IDF(t) · f(t, d) (k<sub>1</sub> + 1) / "
   "[ f(t, d) + k<sub>1</sub> (1 − b + b · |d| / avgdl) ]")
P0("où <i>f(t, d)</i> est la fréquence du terme, |<i>d</i>| la longueur du document, "
   "<i>avgdl</i> la longueur moyenne, et <i>k</i><sub>1</sub> et <i>b</i> deux paramètres de "
   "saturation et de normalisation, fixés ici à leurs valeurs usuelles (1,5 et 0,75). Comme le "
   "TF-IDF, BM25 ignore l'ordre des mots.")
H2("1.2.3 Les encodeurs Transformer")
P("Les encodeurs de la famille BERT construisent une représentation contextuelle de chaque "
  "token par un empilement de couches d'auto-attention. Chaque token peut ainsi tenir compte de "
  "tous les autres, ce qui permet de saisir le sens au-delà des mots exacts. Ce mécanisme a "
  "cependant un coût quadratique en la longueur de la séquence, et les modèles utilisés ici sont "
  "pré-entraînés avec des positions limitées à 512 tokens. Pour un texte plus long, il faut soit "
  "le tronquer, soit le découper en blocs dont on agrège ensuite les sorties. Nous étudions les "
  "deux stratégies. Nous évaluons également un encodeur de phrases dense, all-MiniLM-L6-v2, qui "
  "projette chaque paragraphe dans un espace vectoriel où la proximité reflète la similarité de "
  "sens, et deux modèles de question-réponse extractive qui prédisent pour chaque token la "
  "probabilité d'être le début ou la fin de la réponse.")
H1("1.3 Travaux connexes")
P("Plusieurs études ont déjà montré que le contexte ajouté peut nuire. Jia et Liang ont établi "
  "dès 2017 qu'une seule phrase distractrice, ajoutée à la fin d'un paragraphe SQuAD, suffit à "
  "faire chuter fortement les modèles de compréhension de lecture. Shi et ses co-auteurs ont "
  "observé un effet comparable sur le raisonnement arithmétique en insérant des phrases hors "
  "sujet dans des problèmes mathématiques. Liu et ses co-auteurs ont mis en évidence une courbe "
  "en U de la précision selon la position du passage pertinent dans un long contexte, les "
  "passages placés au milieu étant les moins bien exploités. Du côté de la recherche "
  "d'information, le banc d'essai BEIR a rappelé que BM25 reste une base zéro-shot remarquablement "
  "difficile à battre pour les modèles denses. Le tableau ci-dessous résume ces constats.")
table([["Travail", "Constat principal"],
       ["Jia &amp; Liang, <i>Adversarial Examples for Evaluating Reading Comprehension "
        "Systems</i>, EMNLP 2017",
        "Une phrase distractrice fait passer le F1 SQuAD de 75 % à 36 % en moyenne sur 16 "
        "modèles."],
       ["Thakur et al., <i>BEIR</i>, NeurIPS 2021",
        "Sur 18 jeux de recherche, BM25 reste une base robuste que les modèles denses battent "
        "rarement hors de leur domaine."],
       ["Shi et al., <i>Large Language Models Can Be Easily Distracted by Irrelevant "
        "Context</i>, ICML 2023",
        "Des phrases non pertinentes (GSM-IC) font chuter fortement la précision en "
        "arithmétique."],
       ["Liu et al., <i>Lost in the Middle</i>, TACL 2024",
        "La précision en QA multi-documents suit une courbe en U selon la position du passage "
        "pertinent."]],
      [0.48, 0.52], "Principaux travaux sur la sensibilité des modèles au contexte", 2)
H1("1.4 Positionnement du projet")
P("Ces travaux partagent une limite commune : chacun fait varier un seul facteur, sur une seule "
  "tâche, et le plus souvent avec un seul type de modèle. Il est alors difficile de savoir si une "
  "dégradation observée provient de la quantité de texte, de sa proximité thématique, de sa "
  "cohérence ou de la position du passage utile. Par ailleurs, la plupart de ces études "
  "rapportent uniquement la précision, sans se demander si le modèle reste conscient de ses "
  "erreurs.")
P("Le Semantic Saturation Benchmark comble ce manque de trois façons. Il croise cinq axes "
  "contrôlés et indépendants sur trois familles de tâches, ce qui permet de comparer directement "
  "l'effet de chaque facteur. Il mesure la fiabilité en plus de la précision, au travers de la "
  "calibration de la confiance, de la capacité d'abstention et du comportement face à une "
  "information contradictoire. Enfin, il fixe ses hypothèses avant l'expérience, ce qui protège "
  "les conclusions contre la tentation d'interpréter les résultats après coup.")

# ================================================================== CHAPITRE 2
chapter(2, "Conception du benchmark")
P("Ce chapitre décrit la construction du jeu de données. Nous présentons d'abord le principe "
  "directeur et les sources utilisées, puis les cinq axes de perturbation, les procédures de "
  "génération des contradictions et des paraphrases, et enfin les contrôles qui garantissent la "
  "validité de chaque échantillon.")
H1("2.1 Principe directeur")
P("Le principe du benchmark tient en une phrase : le paragraphe signal ne change jamais, seul "
  "son environnement évolue. Chaque échantillon est formé d'une requête, éventuellement vide en "
  "classification, et d'un contexte de <i>N</i> paragraphes, avec <i>N</i> pris dans l'ensemble "
  "{1, 5, 20, 100}. Le signal est l'unique paragraphe qui permet de répondre ; les autres "
  "paragraphes constituent le remplissage. À <i>N</i> = 1, le contexte se réduit au signal seul, "
  "ce qui fournit la référence la plus propre possible.")
P("Tous les paragraphes de remplissage sont normalisés entre 108 et 132 tokens du tokenizer "
  "bert-base-uncased, soit environ 120 tokens à ±10 %. Cette normalisation se fait en conservant "
  "des phrases entières consécutives, jamais en coupant un mot ou une phrase, de sorte que le "
  "texte reste naturel. Un contexte de cent paragraphes représente ainsi environ douze mille "
  "tokens, le plus long contexte du jeu atteignant 13 119 tokens. Les signaux, eux, ne sont "
  "jamais modifiés : les articles AG News conservent leur longueur naturelle, d'une médiane de "
  "47 tokens, et les paragraphes SQuAD sont choisis parmi ceux qui se trouvent déjà dans la "
  "fenêtre de longueur.")
P("Un point de conception essentiel est que chaque exemple de base apparaît à tous les niveaux "
  "de <i>N</i> et sur tous les axes. Les courbes comparent donc toujours exactement les mêmes "
  "exemples, et les différences entre conditions peuvent être testées de manière appariée, ce "
  "qui réduit fortement la variance des estimations.")
H1("2.2 Sources de données")
P("Trois sources alimentent le benchmark. Pour la tâche de classification, les signaux sont des "
  "articles de l'ensemble de test d'AG News, un corpus de dépêches réparties en quatre classes : "
  "<i>World</i>, <i>Sports</i>, <i>Business</i> et <i>Sci/Tech</i>. Pour la recherche de passage "
  "et la question-réponse, les signaux et les questions proviennent de l'ensemble de "
  "développement de SQuAD 2.0, dont la particularité est de contenir des questions sans réponse "
  "dans le paragraphe. Le remplissage aléatoire est tiré de paragraphes Wikipedia de l'ensemble "
  "d'entraînement de SQuAD 2.0, issus de 442 articles qui ne partagent aucun titre avec les 35 "
  "articles de développement. Cette séparation garantit qu'aucun paragraphe de remplissage ne "
  "parle du même sujet que la question, sauf lorsque l'axe de bruit le prévoit explicitement.")
H1("2.3 Les cinq axes de perturbation")
P("Un seul facteur varie à la fois, à partir d'un contexte de base composé du signal et d'un "
  "remplissage aléatoire. Le tableau suivant résume les cinq axes ; chacun est ensuite décrit en "
  "détail.")
table([["Axe", "Ce qui change", "Niveaux"],
       ["Dilution", "Seulement N (part du signal égale à 1/N)", "N = 1, 5, 20, 100"],
       ["Bruit sémantique", "Remplissage aléatoire ou négatifs difficiles choisis par mpnet",
        "aléatoire / difficile"],
       ["Redondance", "Le signal apparaît k fois", "k = 1 ou 5 ; copies ou paraphrases"],
       ["Contradiction", "Un paragraphe contradictoire accolé au signal", "avant / après"],
       ["Position", "Profondeur relative du signal dans le contexte", "d = 0 ; 0,5 ; 1"]],
      [0.2, 0.52, 0.28], "Les cinq axes de perturbation du benchmark", 2)
H2("2.3.1 Dilution")
P("L'axe de dilution est le plus simple : seul le nombre de paragraphes de remplissage "
  "aléatoire augmente. La part du signal dans le contexte passe ainsi de 100 % à 20 %, 5 % puis "
  "1 %. Cet axe répond directement à la question « que se passe-t-il quand on ajoute du texte "
  "sans rapport ? » et sert de référence à tous les autres.")
H2("2.3.2 Bruit sémantique")
P("L'axe de bruit remplace le remplissage aléatoire par des négatifs difficiles. Ce sont les "
  "paragraphes les plus proches de la requête selon l'encodeur all-mpnet-base-v2, à la condition "
  "qu'ils ne contiennent pas la réponse. Pour la classification, ce sont les paragraphes Wikipedia "
  "les plus proches de l'article lui-même. L'encodeur de sélection n'est pas l'un des modèles "
  "évalués, afin de ne pas favoriser artificiellement l'un d'eux. Cet axe mesure si un contexte "
  "proche du sujet est plus trompeur qu'un contexte quelconque.")
H2("2.3.3 Redondance")
P("L'axe de redondance fait apparaître le signal cinq fois au lieu d'une. Deux modes sont "
  "comparés : des copies exactes, et des paraphrases obtenues par rétro-traduction de l'anglais "
  "vers le français puis du français vers l'anglais. À <i>N</i> = 1, le contexte contient alors "
  "cinq paragraphes, tous porteurs du signal. Cet axe est le seul qui teste la possibilité d'un "
  "gain : répéter l'information utile permet-il de résister à la dilution ?")
H2("2.3.4 Contradiction")
P("L'axe de contradiction place à côté du signal, en tête de contexte, un paragraphe qui "
  "affirme une autre réponse, soit avant le signal, soit juste après. Sa construction dépend de "
  "la tâche et fait l'objet de la section 2.4.")
H2("2.3.5 Position")
P("L'axe de position déplace le signal au début (<i>d</i> = 0), au milieu (<i>d</i> = 0,5) ou à "
  "la fin (<i>d</i> = 1) du contexte. Il n'a de sens qu'à partir de <i>N</i> = 5. Cet axe permet "
  "de tester l'effet de la troncature à 512 tokens et un éventuel biais de position.")
H1("2.4 Génération des contradictions")
P("En classification, le paragraphe contradictoire est simplement un article d'une autre "
  "classe, choisie uniformément parmi les trois restantes. En recherche et en question-réponse, "
  "la construction est plus exigeante. Nous fabriquons une copie du paragraphe gold dans "
  "laquelle chaque occurrence de l'entité réponse est remplacée par une autre entité du même type "
  "selon le reconnaisseur d'entités de spaCy, absente du paragraphe d'origine. Le texte obtenu se "
  "lit presque exactement comme l'original mais affirme une réponse différente, ce qui en fait un "
  "piège particulièrement difficile.")
P("Deux détails techniques ont une importance pratique. D'une part, la réponse SQuAD ne "
  "coïncide pas toujours exactement avec une entité : « the Denver Broncos » contient l'entité "
  "« Denver Broncos » sans lui être identique. Nous acceptons donc l'appariement dès que les deux "
  "segments se recouvrent sur au moins 60 % du plus long. D'autre part, le remplacement doit être "
  "insensible à la casse, faute de quoi la copie peut encore contenir la réponse sous une autre "
  "graphie, par exemple « oxygen » lorsque la réponse est « Oxygen ». Cette erreur a été "
  "détectée par les contrôles automatiques lors d'une première exécution complète et corrigée. "
  "Sur les 1 088 questions examinées, 374 ont une réponse qui est une entité, et 372 admettent un "
  "remplacement valide. Le tableau suivant donne la répartition des types d'entités retenus.")
man_types = [("DATE", 70), ("PERSON", 61), ("ORG", 46), ("CARDINAL", 31), ("GPE", 21),
             ("LOC", 21), ("NORP", 14), ("FAC", 9), ("MONEY", 5), ("LAW", 4), ("PERCENT", 4),
             ("QUANTITY", 4), ("PRODUCT", 3), ("EVENT", 2), ("LANGUAGE", 2),
             ("WORK_OF_ART", 2), ("ORDINAL", 1)]
desc = {"DATE": "dates et périodes", "PERSON": "personnes", "ORG": "organisations",
        "CARDINAL": "nombres", "GPE": "pays, villes, États", "LOC": "lieux géographiques",
        "NORP": "nationalités, groupes", "FAC": "bâtiments, infrastructures",
        "MONEY": "montants", "LAW": "textes de loi", "PERCENT": "pourcentages",
        "QUANTITY": "quantités", "PRODUCT": "produits", "EVENT": "événements",
        "LANGUAGE": "langues", "WORK_OF_ART": "œuvres", "ORDINAL": "ordinaux"}
half = (len(man_types) + 1) // 2
rows = [["Type", "Signification", "Effectif", "Type", "Signification", "Effectif"]]
for i in range(half):
    a = man_types[i]
    b = man_types[i + half] if i + half < len(man_types) else ("", 0)
    rows.append([a[0], desc[a[0]], str(a[1]), b[0], desc.get(b[0], ""),
                 str(b[1]) if b[0] else ""])
table(rows, [0.15, 0.22, 0.11, 0.15, 0.24, 0.13],
      "Types spaCy des entités réponses retenues pour la contradiction", 6)
H1("2.5 Génération des paraphrases")
P("Les paraphrases sont produites par rétro-traduction avec les modèles MarianMT "
  "opus-mt-en-fr et opus-mt-fr-en. Une première version traduisait chaque paragraphe d'un seul "
  "bloc. Nous avons constaté que MarianMT omet alors fréquemment la dernière phrase du paragraphe, "
  "ce qui rendait la plupart des paraphrases trop courtes d'une vingtaine de tokens pour respecter "
  "la fenêtre de longueur. La construction du jeu complet a d'ailleurs échoué pour cette raison, "
  "faute d'un nombre suffisant de questions utilisables.")
P("La version finale traduit chaque phrase séparément. La recherche en faisceau renvoie quatre "
  "traductions françaises par phrase, puis quatre traductions anglaises de chacune, soit seize "
  "paires de faisceaux. Un candidat de paraphrase est formé en choisissant une paire pour chaque "
  "phrase ; en décalant le choix de la paire d'une phrase à l'autre, on obtient jusqu'à trois fois "
  "plus de candidats, de longueurs variées, sans aucun coût de traduction supplémentaire. Cette "
  "modification a fait passer la part de questions utilisables de 46 % à 85 %. Chaque paraphrase "
  "n'est conservée que si elle garde la réponse, pour la recherche et la question-réponse, ou "
  "l'étiquette selon un classifieur AG News indépendant, dont la précision sur l'entraînement est "
  "de 96,75 %, pour la classification. Les traductions sont mises en cache sur disque, de sorte "
  "qu'une reconstruction ne retraduit jamais un texte.")
H1("2.6 Garanties de validité")
P("La construction vérifie automatiquement un ensemble d'invariants, qui sont également "
  "couverts par les tests unitaires. Le signal doit être présent et identique octet par octet à "
  "la source dans chaque échantillon. Le nombre de paragraphes, le nombre de tokens et la "
  "profondeur du signal doivent correspondre exactement à la condition demandée. En recherche et "
  "en question-réponse, la réponse ne doit apparaître que dans les copies du signal, jamais dans "
  "le remplissage, ce qui exclut qu'un modèle trouve la bonne réponse par hasard ailleurs. Une "
  "copie modifiée ne doit plus contenir la réponse, mais doit contenir l'entité de remplacement. "
  "Un article contradictoire doit avoir une étiquette différente de celle du signal. Toute "
  "violation interrompt la construction : il vaut mieux échouer que produire un échantillon "
  "silencieusement invalide.")
H1("2.7 Statistiques du jeu de données")
P("La version v0.1 compte 300 exemples de base par tâche. Le croisement des axes, des niveaux "
  "de <i>N</i> et des variantes donne 25 950 échantillons et 10 900 paragraphes distincts, pour "
  "un total de plus de 105 millions de tokens de contexte. La construction complète prend 24 "
  "minutes sur un GPU T4. Le tableau suivant détaille la composition par tâche et par axe.")
bta = {"classification": {"dilution": 1200, "noise": 1200, "redundancy": 2400,
                          "contradiction": 2400, "position": 2700},
       "retrieval": {"dilution": 1200, "noise": 1200, "redundancy": 2400,
                     "contradiction": 2400, "position": 2700},
       "qa": {"dilution": 1200, "noise": 1200, "redundancy": 1200, "contradiction": 1200,
              "position": 1350}}
axes_fr = [("dilution", "Dilution"), ("noise", "Bruit"), ("redundancy", "Redondance"),
           ("contradiction", "Contradiction"), ("position", "Position")]
rows = [["Tâche"] + [b for _, b in axes_fr] + ["Total"]]
for tk, name in [("classification", "Classification"), ("retrieval", "Recherche"),
                 ("qa", "QA")]:
    vals = [bta[tk][a] for a, _ in axes_fr]
    rows.append([name] + [f"{v:,}".replace(",", " ") for v in vals] +
                [f"{sum(vals):,}".replace(",", " ")])
tot = [sum(bta[t][a] for t in bta) for a, _ in axes_fr]
rows.append(["<b>Total</b>"] + [f"<b>{v:,}</b>".replace(",", " ") for v in tot] +
            [f"<b>{sum(tot):,}</b>".replace(",", " ")])
table(rows, [0.2, 0.12, 0.11, 0.14, 0.15, 0.13, 0.15],
      "Nombre d'échantillons par tâche et par axe (version v0.1)")
P("Les effectifs de la question-réponse sont plus faibles sur la redondance, la contradiction "
  "et la position, parce que les questions sans réponse n'ont pas de signal à répéter, à "
  "contredire ou à déplacer. L'entonnoir de construction, présenté ci-dessous, montre combien "
  "d'exemples candidats ont dû être examinés pour obtenir 300 exemples complets par tâche. Un "
  "exemple réel, avec le signal, sa copie contradictoire et une paraphrase, figure en annexe A.")
table([["Étape", "Classification", "Recherche et QA"],
       ["Candidats examinés", "384", "1 088"],
       ["Réponse identifiée comme entité", "—", "374"],
       ["Remplacement d'entité valide", "—", "372"],
       ["Paraphrases valides en nombre suffisant", "319", "315"],
       ["Exemples retenus", "300", "300"]],
      [0.5, 0.25, 0.25], "Entonnoir de sélection des exemples de base")

# ================================================================== CHAPITRE 3
chapter(3, "Modèles évalués et implémentation")
P("Ce chapitre présente les huit modèles évalués, la manière dont chacun traite un contexte "
  "long, puis l'architecture logicielle du projet et l'environnement d'exécution.")
H1("3.1 Vue d'ensemble")
P("Les huit modèles couvrent trois formes de compréhension du texte, des représentations "
  "lexicales aux encodeurs Transformer. Leur différence essentielle, au regard de notre "
  "problématique, tient à la manière dont ils traitent un contexte plus long que leur fenêtre "
  "d'entrée. Le tableau suivant les récapitule.")
table([["Tâche", "Modèle", "Type", "Contexte long"],
       ["Classification", "tfidf-lr", "TF-IDF + régression logistique", "aucune limite"],
       ["Classification", "bert-truncate", "BERT-base fine-tuné AG News",
        "512 premiers tokens"],
       ["Classification", "bert-chunk-mean", "même BERT", "moyenne des logits par blocs"],
       ["Recherche", "tfidf", "lexical", "score par paragraphe"],
       ["Recherche", "bm25", "lexical (BM25Okapi)", "score par paragraphe"],
       ["Recherche", "minilm", "dense, all-MiniLM-L6-v2", "score par paragraphe"],
       ["QA extractive", "distilbert-squad", "DistilBERT SQuAD 1.1",
        "fenêtres glissantes 384/128"],
       ["QA extractive", "roberta-squad2", "RoBERTa-base SQuAD 2.0",
        "fenêtres glissantes 384/128"]],
      [0.19, 0.2, 0.33, 0.28], "Modèles évalués et traitement des contextes longs", 4)
H1("3.2 Tâche A : classification thématique")
P("Le premier classifieur, tfidf-lr, combine une représentation TF-IDF et une régression "
  "logistique entraînée sur l'ensemble d'entraînement d'AG News. Il n'a aucune limite de "
  "longueur : le contexte entier est transformé en un seul vecteur. Les deux autres "
  "classifieurs partagent le même modèle, un BERT-base fine-tuné sur AG News et publié par "
  "TextAttack, et ne diffèrent que par leur stratégie de lecture. bert-truncate ne conserve que "
  "les 512 premiers tokens, ce qui est le comportement par défaut de la plupart des "
  "bibliothèques. bert-chunk-mean découpe le contexte en blocs de 512 tokens, classe chaque bloc "
  "et moyenne les logits obtenus. Cette paire de modèles permet d'isoler l'effet de la stratégie "
  "de lecture, à poids identiques.")
H1("3.3 Tâche B : recherche de passage")
P("En recherche, chaque modèle attribue un score à chaque paragraphe du contexte pour la "
  "question posée, et les paragraphes sont classés par score décroissant. La tâche consiste à "
  "placer le paragraphe gold, ou l'une de ses copies, en première position. tfidf utilise la "
  "similarité cosinus des vecteurs TF-IDF, bm25 la fonction BM25Okapi présentée au chapitre 1, "
  "et minilm la similarité cosinus des plongements produits par l'encodeur "
  "all-MiniLM-L6-v2. Ces trois modèles évaluent chaque paragraphe indépendamment ; la longueur "
  "du contexte n'agit donc qu'à travers le nombre de concurrents.")
H1("3.4 Tâche C : question-réponse extractive")
P("La question-réponse extractive consiste à extraire du contexte le segment qui répond à la "
  "question, ou à s'abstenir lorsque la réponse n'y figure pas. Nous évaluons distilbert-squad, "
  "entraîné sur SQuAD 1.1 et donc incapable de s'abstenir, et roberta-squad2, entraîné sur "
  "SQuAD 2.0 avec des questions sans réponse. Le contexte étant bien plus long que la fenêtre de "
  "384 tokens de ces modèles, il est découpé en fenêtres glissantes qui se chevauchent avec un "
  "pas de 128 tokens.")
P("La version 5 de la bibliothèque transformers ayant supprimé le pipeline de question-réponse "
  "extractive, nous avons réimplémenté le décodage. Pour chaque fenêtre, les logits de début et "
  "de fin sont normalisés par une fonction softmax sur les tokens du contexte et le token CLS. "
  "Le score d'une plage allant du token <i>i</i> au token <i>j</i> est :")
eq("s(i, j) = p<sub>début</sub>(i) · p<sub>fin</sub>(j),  avec  i ≤ j &lt; i + 30")
P0("et le score d'abstention est s<sub>nul</sub> = p<sub>début</sub>(CLS) · p<sub>fin</sub>(CLS). "
   "La meilleure plage est retenue sur l'ensemble des fenêtres ; pour roberta-squad2, le modèle "
   "s'abstient lorsque le plus petit score d'abstention sur les fenêtres est supérieur ou égal "
   "au meilleur score de plage. La réponse est enfin élargie aux frontières de mots à l'aide "
   "des décalages de caractères fournis par le tokenizer. Trois tests unitaires vérifient ce "
   "décodage sur des cas construits à la main.")
H1("3.5 Architecture logicielle")
P("Le projet est organisé comme un paquet Python, nommé <font name='Mono'>ssb</font>, piloté "
  "par une interface en ligne de commande et par des fichiers de configuration YAML validés au "
  "chargement. Chaque étape de la chaîne, construction, exécution et rapport, lit la "
  "configuration et écrit ses sorties sur disque, de sorte qu'elle peut être relancée isolément. "
  "Le tableau suivant décrit le rôle de chaque module.")
table([["Module", "Rôle"],
       ["config.py", "lecture et validation stricte de la configuration YAML"],
       ["data.py", "téléchargement et chargement d'AG News et de SQuAD 2.0"],
       ["paragraphs.py", "découpage en phrases, normalisation de longueur, recherche de réponse"],
       ["paraphrase.py", "rétro-traduction phrase par phrase avec cache disque"],
       ["contradict.py", "remplacement d'entités nommées et contradiction inter-classes"],
       ["perturb.py", "placement du signal et assemblage des contextes"],
       ["build.py", "construction du jeu de données et contrôle des invariants"],
       ["tasks/", "les trois familles de modèles : classification, recherche, QA"],
       ["run.py", "exécution des modèles, prédictions et temps de calcul"],
       ["metrics.py, analysis.py", "métriques, bootstrap, tests et verdicts des hypothèses"],
       ["plots.py, report.py", "figures (anglais et français) et rapport de résultats"],
       ["cli.py", "interface en ligne de commande : build, run, report"]],
      [0.3, 0.7], "Modules du paquet ssb", 9)
P("La qualité du code est contrôlée par 36 fonctions de test, dont plusieurs paramétrées, soit "
  "45 cas au total. Ils couvrent la validation de la configuration, la normalisation des "
  "paragraphes, le placement du signal, le remplacement d'entités, la construction d'un jeu "
  "miniature complet, le calcul des métriques et le décodage de la question-réponse.")
H1("3.6 Environnement d'exécution")
P("L'expérience complète a été exécutée sur la plateforme Kaggle, avec un GPU NVIDIA Tesla T4, "
  "PyTorch 2.11 et transformers 5.23. Un script unique clone le dépôt, construit le jeu de "
  "données, exécute les huit modèles et produit le rapport. La construction a pris 1 416 "
  "secondes et l'évaluation des modèles 0,55 heure. Le temps de calcul de chaque modèle est "
  "donné en annexe B. Un jeu réduit de 20 exemples par tâche, soit 1 730 échantillons, permet de "
  "vérifier toute la chaîne en quelques minutes sur un simple processeur avant de lancer "
  "l'exécution complète.")

# ================================================================== CHAPITRE 4
chapter(4, "Protocole expérimental")
P("Ce chapitre définit précisément ce qui est mesuré et comment les conclusions sont tirées. "
  "Nous présentons les métriques, la méthode d'estimation de l'incertitude, les indicateurs de "
  "saturation et enfin les huit hypothèses pré-enregistrées.")
H1("4.1 Métriques de performance")
P("Chaque tâche possède une métrique principale. En classification, il s'agit de la précision, "
  "c'est-à-dire la part d'articles correctement classés. En recherche, il s'agit du Recall@1, "
  "la part de questions pour lesquelles le paragraphe classé premier est le gold ou l'une de ses "
  "copies. En question-réponse, la métrique principale est l'exact match sur les questions avec "
  "réponse, après normalisation usuelle de SQuAD, et la précision d'abstention sur les questions "
  "sans réponse.")
P("Des métriques secondaires complètent ce tableau : le macro-F1 en classification, qui donne "
  "le même poids à chaque classe et révèle donc un effondrement vers une classe unique ; le "
  "Recall@5 et le rang réciproque moyen en recherche ; et le F1 au niveau des tokens en "
  "question-réponse, qui accorde un crédit partiel aux réponses incomplètes. Le rang réciproque "
  "moyen est défini par :")
eq("MRR = (1 / n) Σ<sub>i</sub> 1 / rang<sub>i</sub>")
P0("où rang<sub>i</sub> est la position du premier paragraphe pertinent pour la question "
   "<i>i</i>.")
H1("4.2 Mesures de fiabilité")
P("Un modèle fiable ne se contente pas d'avoir raison souvent : sa confiance doit refléter ses "
  "chances réelles d'avoir raison. La confiance est la probabilité softmax maximale pour les "
  "classifieurs et le score de la plage retenue pour la question-réponse. Les modèles de "
  "recherche ne produisent pas de probabilité et sont exclus de cette analyse. Nous mesurons "
  "d'abord la surconfiance, différence entre la confiance moyenne et le score, puis l'erreur de "
  "calibration attendue, calculée en regroupant les prédictions en <i>B</i> = 10 intervalles de "
  "confiance de largeur égale :")
eq("ECE = Σ<sub>b</sub> ( |B<sub>b</sub>| / n ) · | acc(B<sub>b</sub>) − conf(B<sub>b</sub>) |")
P0("Pour la classification, le score de Brier mesure en outre l'écart quadratique entre la "
   "distribution prédite et l'étiquette réelle :")
eq("Brier = (1 / n) Σ<sub>i</sub> Σ<sub>c</sub> ( p<sub>ic</sub> − y<sub>ic</sub> )<super>2"
   "</super>")
P("Pour l'axe de contradiction, nous classons enfin chaque prédiction selon qu'elle suit le "
  "signal, le paragraphe contradictoire, aucun des deux, ou qu'elle les place à égalité. La "
  "<i>part de conflit</i> est la proportion de réponses contradictoires parmi celles qui "
  "désignent l'une des deux entités.")
H1("4.3 Estimation de l'incertitude")
P("Avec 300 exemples par tâche, une différence de quelques points peut résulter du seul hasard "
  "de l'échantillonnage. Toutes les différences sont donc accompagnées d'un intervalle de "
  "confiance à 95 %, obtenu par 1 000 rééchantillonnages bootstrap. Le rééchantillonnage porte "
  "sur les exemples de base et non sur les échantillons : lorsqu'un exemple est tiré, toutes ses "
  "variantes le sont avec lui. Les comparaisons entre conditions sont ainsi appariées, ce qui "
  "élimine la variabilité propre à chaque exemple. Les tests unilatéraux utilisent la "
  "distribution bootstrap de la différence appariée.")
P("Lorsqu'une hypothèse implique plusieurs comparaisons, par exemple une par modèle, les "
  "valeurs <i>p</i> sont corrigées par la méthode de Holm. Les <i>m</i> valeurs sont triées par "
  "ordre croissant, et la <i>i</i>-ième est déclarée significative si elle et toutes les "
  "précédentes vérifient :")
eq("p<sub>(i)</sub> ≤ α / (m − i + 1),  avec  α = 0,05")
P0("Cette procédure contrôle le risque de déclarer à tort au moins un effet, tout en étant "
   "uniformément plus puissante que la correction de Bonferroni.")
H1("4.4 Indicateurs de saturation")
P("Trois indicateurs résument chaque courbe de score en fonction de <i>N</i>. Le point de "
  "saturation <i>N</i>* est le plus petit <i>N</i> dont le gain par rapport au niveau précédent "
  "est inférieur à un point : au-delà, ajouter du contexte n'apporte plus rien. Le point de "
  "dégradation <i>N</i>† est le plus petit <i>N</i>, après le pic de la courbe, dont le score est "
  "significativement inférieur à celui du pic selon le test bootstrap corrigé par Holm. Enfin, le "
  "<i>Semantic Saturation Score</i> résume toute la courbe en un nombre : c'est l'aire sous la "
  "courbe du score en fonction de log<sub>10</sub> <i>N</i>, calculée par la méthode des "
  "trapèzes et divisée par l'étendue logarithmique :")
eq("SSS = [ 1 / (log<sub>10</sub> N<sub>max</sub> − log<sub>10</sub> N<sub>min</sub>) ] · "
   "∫ s(N) d(log<sub>10</sub> N)")
P0("Le SSS vaut 1 pour un modèle parfait à tous les niveaux de contexte, et 0,25 pour un "
   "classifieur qui répondrait au hasard partout. L'échelle logarithmique donne le même poids "
   "à chaque multiplication de la taille du contexte.")
H1("4.5 Hypothèses pré-enregistrées")
P("Afin de ne pas ajuster nos conclusions aux résultats, huit hypothèses ont été rédigées et "
  "versionnées dans le dépôt avant toute évaluation sur la version v0.1. Chacune est accompagnée "
  "d'un test précis et d'un critère de décision : elle est dite supportée si le critère est "
  "rempli pour tous les modèles concernés, partiellement supportée s'il l'est pour une partie "
  "d'entre eux, et non supportée sinon. Les exécutions sur le jeu réduit ne servaient qu'à "
  "vérifier la chaîne et n'ont été utilisées pour aucune décision. Tout ce qui ne figure pas dans "
  "cette liste, comme les analyses par classe ou par type d'entité, est considéré comme "
  "exploratoire.")
table([["Id", "Hypothèse"],
       ["H1", "Tous les modèles se dégradent quand N croît, et le bruit difficile fait plus de "
              "mal que le bruit aléatoire."],
       ["H2", "Les sacs de mots n'ont pas de biais de position ; bert-truncate s'effondre "
              "au-delà des 512 premiers tokens."],
       ["H3", "chunk-mean corrige le biais de position mais se dégrade toujours avec la "
              "dilution."],
       ["H4", "Les modèles lexicaux battent l'encodeur dense sur négatifs difficiles ; le dense "
              "gagne sur bruit aléatoire."],
       ["H5", "La redondance ralentit la dégradation ; les copies aident plus TF-IDF, les "
              "paraphrases plus les Transformers."],
       ["H6", "Face à une contradiction, les modèles de QA choisissent la réponse "
              "contradictoire environ une fois sur deux, et la position joue."],
       ["H7", "La confiance reste élevée pendant que la précision baisse."],
       ["H8", "Plus N croît, plus les questions sans réponse reçoivent de fausses réponses."]],
      [0.08, 0.92], "Hypothèses pré-enregistrées", 9)

# ================================================================== CHAPITRE 5
chapter(5, "Résultats expérimentaux")
P("L'exécution complète porte sur 25 950 échantillons. Le constat général est sans ambiguïté : "
  "plus de contexte n'a aidé aucun modèle. Les huit se dégradent significativement entre "
  "<i>N</i> = 1 et <i>N</i> = 100, mais l'ampleur de cette dégradation varie considérablement "
  "selon la tâche. Ce chapitre présente les résultats axe par axe ; tous les écarts sont donnés "
  "avec leur intervalle de confiance bootstrap à 95 %.")
H1("5.1 Dilution")
P("Le tableau suivant donne le score principal de chaque modèle aux quatre niveaux de contexte, "
  "avec remplissage aléatoire.")
rows = [["Tâche", "Modèle", "N = 1", "N = 5", "N = 20", "N = 100", "Baisse [IC 95 %]"]]
drops = {}
for m in ALL:
    v = [g(m, "dilution", n) for n in (1, 5, 20, 100)]
    drops[m] = v[0] - v[3]
for m in sorted(ALL, key=lambda m: -drops[m]):
    v = [g(m, "dilution", n) for n in (1, 5, 20, 100)]
    rows.append([TASK_OF[m], m] + [pct(x) for x in v] + [""])
ci = {"tfidf-lr": "−68,0 [−73,3 ; −62,3]", "bert-truncate": "−73,7 [−78,7 ; −68,7]",
      "bert-chunk-mean": "−72,7 [−78,0 ; −67,3]", "tfidf": "−5,3 [−8,0 ; −3,0]",
      "bm25": "−6,3 [−9,3 ; −3,7]", "minilm": "−5,7 [−8,7 ; −3,3]",
      "distilbert-squad": "−12,0 [−18,0 ; −6,0]", "roberta-squad2": "−4,0 [−8,5 ; −0,6]"}
for r in rows[1:]:
    r[-1] = ci[r[1]]
table(rows, [0.15, 0.19, 0.1, 0.1, 0.1, 0.11, 0.25],
      "Score principal sous dilution (remplissage aléatoire) et baisse entre N = 1 et N = 100, "
      "en points", 2)
P("En classification, la précision tombe au niveau du hasard, soit 25 % pour quatre classes, "
  "dès vingt paragraphes, et la chute est déjà massive à cinq paragraphes : bert-truncate passe "
  "de 97,7 % à 34,7 %. L'examen des prédictions montre qu'il ne s'agit pas d'un bruit aléatoire "
  "mais d'un biais systématique : les classifieurs basculent vers la classe <i>Sci/Tech</i>. À "
  "<i>N</i> = 100, bert-chunk-mean prédit cette classe pour la totalité des 300 articles, "
  "bert-truncate pour 276 d'entre eux et tfidf-lr pour 182, les 118 autres étant attribués à "
  "<i>World</i>. Le macro-F1, qui serait de 1 pour un classifieur parfait, tombe ainsi à 0,10 "
  "pour bert-chunk-mean, valeur caractéristique d'un modèle qui ne prédit qu'une seule classe.")
P("La recherche et la question-réponse résistent bien mieux. Les trois modèles de recherche "
  "conservent un Recall@1 compris entre 93,7 % et 94,7 % à cent paragraphes, et roberta-squad2 "
  "ne perd que quatre points. La différence de comportement s'explique par la manière "
  "d'agréger le texte : ces modèles évaluent chaque paragraphe ou chaque fenêtre séparément, si "
  "bien qu'un paragraphe hors sujet n'altère pas la représentation du signal, il ajoute seulement "
  "un concurrent. Le point de saturation <i>N</i>* vaut 5 pour les huit modèles : au-delà de "
  "cinq paragraphes, ajouter du contexte n'apporte plus jamais de gain.")
figure("fig1_main.png", "Score en fonction de la taille du contexte avec remplissage "
       "aléatoire ; les bandes colorées représentent l'intervalle de confiance bootstrap à 95 %")
H1("5.2 Bruit sémantique")
P("Le tableau suivant compare, à <i>N</i> = 100, le score obtenu avec un remplissage aléatoire "
  "et avec des négatifs difficiles, ainsi que l'évolution avec <i>N</i> sous bruit difficile.")
rows = [["Tâche", "Modèle", "N = 5", "N = 20", "N = 100", "Aléatoire N = 100",
         "Écart"]]
for m in ALL:
    h = [g(m, "noise", n) for n in (5, 20, 100)]
    r_ = g(m, "dilution", 100)
    d = (r_ - h[2]) * 100
    rows.append([TASK_OF[m], m] + [pct(x) for x in h] + [pct(r_),
                 (f"+{d:.1f}" if d >= 0 else f"−{-d:.1f}").replace(".", ",")])
table(rows, [0.15, 0.2, 0.11, 0.11, 0.11, 0.18, 0.14],
      "Score sous bruit difficile et comparaison au remplissage aléatoire ; un écart positif "
      "signifie que le bruit difficile fait plus de mal", 2)
P("En recherche et en question-réponse, les négatifs difficiles font nettement plus de mal que "
  "le hasard. À <i>N</i> = 100, le Recall@1 tombe à 77,7 % pour tfidf, 79,3 % pour bm25 et "
  "seulement 65,7 % pour minilm, tandis que l'exact match de distilbert-squad descend à 49,3 %. "
  "L'encodeur dense minilm, pourtant réputé saisir le sens au-delà des mots, est le plus "
  "affecté, avec une perte de 28,7 points par rapport au bruit aléatoire. Il perd d'ailleurs "
  "l'essentiel dès cinq paragraphes, ce qui indique que quelques voisins sémantiques suffisent à "
  "le tromper. roberta-squad2 est le plus robuste des modèles de cette famille, avec un écart de "
  "huit points seulement.")
P("En classification, l'effet est inversé. tfidf-lr conserve 84,0 % de précision avec des "
  "paragraphes proches de l'article, contre 27,7 % avec un remplissage aléatoire. Les deux "
  "variantes de BERT conservent environ 45 %, contre 24 à 25 %. L'explication est simple : un "
  "paragraphe Wikipedia proche d'un article sportif parle lui-même de sport, et renforce donc "
  "l'étiquette du signal au lieu de la diluer. Ce résultat montre que la proximité thématique "
  "n'a pas la même signification pour une tâche de classification, où elle est une aide, et pour "
  "une tâche de localisation de l'information, où elle est un piège.")
figure("fig2_noise_types.png", "Remplissage aléatoire (trait plein) et négatifs difficiles "
       "(tirets) selon la taille du contexte")
H1("5.3 Position du signal")
rows = [["Tâche", "Modèle", "d = 0", "d = 0,5", "d = 1", "Écart max."]]
for m in ALL:
    v = [g(m, "position_d0", 100), g(m, "position", 100), g(m, "position_d1", 100)]
    rows.append([TASK_OF[m], m] + [pct(x) for x in v] +
                [f"{100 * (max(v) - min(v)):.1f}".replace(".", ",")])
table(rows, [0.17, 0.23, 0.14, 0.14, 0.14, 0.18],
      "Score à N = 100 selon la profondeur du signal (0 = début, 1 = fin), écart en points", 2)
P("Les modèles à sac de mots, tfidf-lr, tfidf et bm25, obtiennent exactement le même score "
  "quelle que soit la profondeur du signal, ce qui est attendu puisque leur représentation "
  "ignore l'ordre des mots. À l'inverse, bert-truncate atteint 53,7 % lorsque le signal est en "
  "tête du contexte, mais seulement 23,3 % lorsqu'il se trouve au milieu ou à la fin, positions "
  "qui sont simplement coupées par la troncature à 512 tokens. Il est d'ailleurs remarquable que "
  "même avec le signal en tête, la précision ne dépasse pas 54 % : les quelques paragraphes de "
  "remplissage qui entrent dans la fenêtre suffisent à faire basculer une prédiction sur deux.")
P("La stratégie chunk-mean supprime totalement ce biais de position, mais au prix d'un score "
  "uniformément faible : en moyennant les logits sur une centaine de blocs, elle noie le bloc "
  "qui contient le signal. En question-réponse, la fenêtre glissante rend la position presque "
  "neutre, avec des écarts inférieurs à quatre points. Nous n'observons donc pas, pour ces "
  "modèles, la courbe en U décrite par Liu et ses co-auteurs pour les grands modèles de langage.")
figure("fig3_position_heatmap.png", "Score selon la profondeur du signal (0 = début, 1 = fin "
       "du contexte) et selon la taille du contexte")
H1("5.4 Contradiction")
rows = [["Tâche", "Modèle", "Vraie réponse", "Contradictoire", "Égalité", "Autre",
         "Part de conflit"]]
for m in RET + QA:
    def avg(metric):
        return (g(m, "before", 100, metric) + g(m, "after", 100, metric)) / 2
    rows.append([TASK_OF[m], m, pct(avg("outcome_true")), pct(avg("outcome_conflicting")),
                 pct(avg("outcome_tie")), pct(avg("outcome_other")),
                 pct(g(m, "all", 0, "conflict_share"))])
table(rows, [0.12, 0.19, 0.13, 0.17, 0.1, 0.1, 0.19],
      "Issue des prédictions à N = 100 face à un paragraphe contradictoire (moyenne des "
      "positions avant et après)", 2)
P("Un paragraphe presque identique au signal mais affirmant une autre réponse trompe souvent "
  "les modèles. En recherche, la copie modifiée est classée première dans 33,0 % des cas pour "
  "tfidf, 39,3 % pour minilm et 27,0 % pour bm25, ce dernier produisant en outre 27,7 % "
  "d'égalités parfaites. Ces chiffres ne varient pas avec <i>N</i> : la difficulté ne vient pas "
  "du remplissage mais de l'impossibilité de distinguer deux paragraphes qui ne diffèrent que "
  "par une entité. Pour un modèle lexical, la question ne contient généralement pas la réponse, "
  "et les deux paragraphes partagent tous les termes de la question.")
P("En question-réponse, parmi les réponses qui reprennent l'une des deux entités, "
  "distilbert-squad choisit l'entité contradictoire dans 33,5 % des cas [28,1 ; 38,2] et "
  "roberta-squad2 dans 30,9 % [25,1 ; 36,8]. La position joue un rôle net. Lorsque le "
  "paragraphe contradictoire précède le signal, distilbert-squad le choisit 19,6 points plus "
  "souvent (<i>p</i> &lt; 0,001) et roberta-squad2 10,7 points plus souvent (<i>p</i> = 0,048). "
  "Autrement dit, le premier paragraphe lu l'emporte. En classification, enfin, la contradiction "
  "se superpose à l'effondrement vers <i>Sci/Tech</i> : à cent paragraphes, l'article d'une autre "
  "classe n'est plus suivi que dans environ un quart des cas, mais le signal ne l'est pas "
  "davantage.")
figure("fig5_contradiction.png", "Part des prédictions qui suivent le signal (bleu) ou le "
       "paragraphe contradictoire (orange), conflit placé avant (Av) ou après (Ap) le signal")
H1("5.5 Calibration et abstention")
rows = [["Modèle", "Confiance N = 1", "Confiance N = 100", "Score N = 100", "ECE N = 1",
         "ECE N = 100"]]
for m in CLS + QA:
    rows.append([m, pct(g(m, "dilution", 1, "confidence")),
                 pct(g(m, "dilution", 100, "confidence")), pct(g(m, "dilution", 100)),
                 num(g(m, "dilution", 1, "ece")), num(g(m, "dilution", 100, "ece"))])
table(rows, [0.22, 0.16, 0.17, 0.15, 0.15, 0.15],
      "Confiance moyenne, score et erreur de calibration sous dilution")
P("Les classifieurs se trompent en restant sûrs d'eux. Entre <i>N</i> = 1 et <i>N</i> = 100, "
  "l'écart entre la confiance moyenne et la précision augmente de 53,0 points pour tfidf-lr, de "
  "72,4 points pour bert-truncate et de 73,3 points pour bert-chunk-mean. Ce dernier maintient "
  "une confiance moyenne de 99,3 % alors que sa précision est de 25 %, et son erreur de "
  "calibration passe de 0,010 à 0,743. Rien dans la sortie du modèle ne permet donc à un "
  "utilisateur de détecter l'effondrement.")
P("La situation est plus nuancée en question-réponse. distilbert-squad devient modérément "
  "surconfiant, avec une hausse de 13,5 points, mais roberta-squad2 reste bien calibré, son "
  "écart variant de 1,9 point seulement, ce qui n'est pas significatif. Sur les questions sans "
  "réponse, roberta-squad2 s'abstient correctement dans environ 82 % des cas quel que soit "
  "<i>N</i> sous dilution, mais cette précision d'abstention perd 6,7 points en présence de "
  "négatifs difficiles : des paragraphes proches de la question l'incitent à proposer une "
  "réponse qui n'existe pas.")
figure("fig6_calibration.png", "Confiance moyenne en fonction du score, de N = 1 à N = 100 ; "
       "la diagonale pointillée correspond à une calibration parfaite", width=TEXT_W * 0.62)
H1("5.6 Redondance")
rows = [["Tâche", "Modèle", "k = 1", "k = 5 copies", "k = 5 paraphrases", "SSS k = 1",
         "SSS copies"]]
lbv = lambda m, a: float(LB[(LB.model == m) & (LB.axis == a)].SSS.iloc[0])
for m in ALL:
    rows.append([TASK_OF[m], m, pct(g(m, "dilution", 100)),
                 pct(g(m, "redundancy_copy", 100)), pct(g(m, "redundancy_paraphrase", 100)),
                 num(lbv(m, "dilution")), num(lbv(m, "redundancy_copy"))])
table(rows, [0.16, 0.19, 0.1, 0.13, 0.16, 0.13, 0.13],
      "Score à N = 100 et SSS avec un signal unique, cinq copies ou cinq paraphrases", 2)
P("Répéter le signal cinq fois atténue significativement la baisse pour quatre modèles sur "
  "huit. L'effet est le plus visible en classification sur l'ensemble de la courbe : pour "
  "tfidf-lr, le SSS passe de 0,451 sous dilution seule à 0,681 avec des copies, car la "
  "précision reste à 96 % tant que les copies sont majoritaires. Dès que le remplissage dépasse "
  "largement les copies, à <i>N</i> = 20 puis <i>N</i> = 100, la classification retombe vers le "
  "hasard. La redondance retarde donc la saturation sans l'empêcher.")
P("Copies exactes et paraphrases donnent des résultats très proches, avec des écarts de zéro à "
  "cinq points qui ne sont presque jamais significatifs. Ce résultat contredit notre hypothèse "
  "selon laquelle les Transformers, sensibles au sens, profiteraient davantage de formulations "
  "variées, et les modèles lexicaux de copies exactes. Une explication plausible est que les "
  "paraphrases par rétro-traduction restent lexicalement très proches de l'original, comme "
  "l'illustre l'exemple de l'annexe A : elles ne diversifient pas assez le signal pour qu'un "
  "effet apparaisse.")
figure("fig4_redundancy.png", "Signal unique (k = 1) comparé à cinq copies exactes et à cinq "
       "paraphrases obtenues par rétro-traduction, pour chaque modèle")
H1("5.7 Métriques secondaires")
rows = [["Modèle", "Métrique", "N = 1", "N = 5", "N = 20", "N = 100"]]
for m, met, lab in [("tfidf-lr", "macro_f1", "macro-F1"), ("bert-truncate", "macro_f1", "macro-F1"),
                    ("bert-chunk-mean", "macro_f1", "macro-F1"), ("tfidf", "mrr", "MRR"),
                    ("bm25", "mrr", "MRR"), ("minilm", "mrr", "MRR"), ("tfidf", "r5", "Recall@5"),
                    ("bm25", "r5", "Recall@5"), ("minilm", "r5", "Recall@5"),
                    ("distilbert-squad", "f1", "F1 token"), ("roberta-squad2", "f1", "F1 token")]:
    rows.append([m, lab] + [num(g(m, "dilution", n, met)) for n in (1, 5, 20, 100)])
table(rows, [0.26, 0.18, 0.14, 0.14, 0.14, 0.14],
      "Métriques secondaires sous dilution", 2)
P("Les métriques secondaires confirment les conclusions précédentes. Le macro-F1 des "
  "classifieurs chute encore plus fortement que leur précision, jusqu'à 0,10 pour "
  "bert-chunk-mean et 0,13 pour bert-truncate, signe que les prédictions se concentrent sur une "
  "seule classe. En recherche, le Recall@5 reste supérieur à 97,7 % pour les trois modèles à "
  "cent paragraphes : même lorsque le gold n'est pas classé premier, il figure presque toujours "
  "parmi les cinq premiers, ce qui est rassurant pour un usage en chaîne de recherche suivie "
  "d'une lecture. Le F1 au niveau des tokens suit l'exact match, avec quelques points de plus.")
H1("5.8 Classement global")
axes_lb = [("dilution", "Dilution"), ("noise", "Bruit"), ("redundancy_copy", "Copies"),
           ("redundancy_paraphrase", "Paraphr."), ("contradiction", "Contrad."),
           ("position", "Position")]
rows = [["Modèle"] + [b for _, b in axes_lb]]
for m in sorted(ALL, key=lambda m: -lbv(m, "dilution")):
    rows.append([m] + [num(lbv(m, a)) for a, _ in axes_lb])
table(rows, [0.22] + [0.13] * 6, "Semantic Saturation Score de chaque modèle sur chaque axe")
P("Le tableau ci-dessus résume la robustesse de chaque modèle sur chaque axe par son SSS. Les "
  "trois modèles de recherche dominent sur la dilution, la redondance et la position, avec des "
  "scores proches de 0,98, suivis des modèles de question-réponse autour de 0,80 à 0,85, tandis "
  "que les classifieurs ferment la marche autour de 0,42 à 0,45. La contradiction est l'axe le "
  "plus difficile pour toutes les tâches : aucun modèle de recherche ni de question-réponse n'y "
  "dépasse 0,66. Le bruit difficile sépare nettement minilm, à 0,722, des deux modèles lexicaux, "
  "à 0,853 et 0,870. L'axe de position, enfin, est le plus pénalisant pour les classifieurs, "
  "parce qu'il ne commence qu'à <i>N</i> = 5, niveau où ils sont déjà effondrés.")
H1("5.9 Bilan des hypothèses")
table([["Id", "Verdict", "Élément déterminant"],
       ["H1", "partiellement supportée", "les 8 modèles se dégradent ; le bruit difficile est "
        "pire pour 5 sur 8 mais aide les classifieurs"],
       ["H2", "supportée", "écart de position nul pour les sacs de mots ; bert-truncate 23,3 % "
        "en milieu et fin contre 53,7 % en tête"],
       ["H3", "supportée", "chunk-mean : aucun écart de position, mais −72,7 points de "
        "dilution"],
       ["H4", "partiellement supportée", "bm25 et tfidf battent minilm sur négatifs difficiles "
        "(+13,7 et +12,0 points, significatifs) ; aucune différence sur bruit aléatoire"],
       ["H5", "non supportée", "la redondance atténue la baisse pour 4 modèles sur 8 ; copies "
        "et paraphrases ne se distinguent pas"],
       ["H6", "partiellement supportée", "part de conflit de 31 à 34 %, sous la fourchette "
        "35–65 % ; effet de position significatif pour les deux modèles"],
       ["H7", "partiellement supportée", "surconfiance de +53 à +73 points pour les "
        "classifieurs et +13,5 pour distilbert, mais pas pour roberta-squad2"],
       ["H8", "partiellement supportée", "abstention stable sous dilution (+0,7 point), "
        "−6,7 points sous bruit difficile"]],
      [0.07, 0.25, 0.68], "Verdicts des hypothèses pré-enregistrées", 9)
P("Les deux hypothèses portant sur la position, H2 et H3, sont pleinement confirmées : elles "
  "découlent directement de la structure des modèles et les résultats suivent exactement la "
  "prédiction. La plupart des autres sont partiellement confirmées, la tendance annoncée étant "
  "observée mais pas pour tous les modèles ou pas avec l'ampleur prévue. H1 échoue sur un point "
  "instructif, l'effet inverse du bruit difficile en classification ; H6 sous-estimait la "
  "résistance des modèles à la contradiction ; H7 et H8 ne s'appliquent pas à roberta-squad2, "
  "qui se révèle le modèle le mieux calibré. Seule H5 est rejetée, faute de différence entre "
  "copies et paraphrases.")

# ================================================================== CHAPITRE 6
chapter(6, "Discussion")
P("Ce chapitre interprète les résultats, en dégage les enseignements pratiques et précise les "
  "limites qui encadrent leur portée.")
H1("6.1 L'agrégation explique l'essentiel")
P("Le fil conducteur de nos résultats est la manière dont chaque modèle agrège le texte. Les "
  "modèles qui résument tout le contexte en une seule représentation, tfidf-lr et "
  "bert-chunk-mean, ou qui en lisent un préfixe fixe, bert-truncate, s'effondrent sous la "
  "dilution. Les modèles qui examinent chaque passage séparément, les trois modèles de recherche "
  "et les deux modèles de question-réponse à fenêtres glissantes, résistent. Cette distinction "
  "pèse bien plus que la sophistication du modèle : bert-chunk-mean, un Transformer de 110 "
  "millions de paramètres, obtient un SSS inférieur à celui d'un simple TF-IDF en recherche.")
P("L'effondrement vers <i>Sci/Tech</i> en est l'illustration la plus frappante. Un contexte hors "
  "sujet ne dilue pas seulement le signal : il le remplace par une autre étiquette, celle qui "
  "correspond le mieux au remplissage. Les paragraphes encyclopédiques de Wikipedia, avec leur "
  "vocabulaire technique et descriptif, ressemblent davantage à des dépêches scientifiques qu'à "
  "des nouvelles sportives ou économiques. Le modèle répond donc correctement à la question "
  "« de quoi parle ce texte dans son ensemble ? », qui n'est simplement pas la question posée.")
H1("6.2 La confiance n'est pas un signal d'alarme")
P("Le second enseignement concerne la fiabilité. Les classifieurs effondrés conservent une "
  "confiance supérieure à 97 % dans le cas des modèles BERT. Une stratégie courante en "
  "production consiste à rejeter les prédictions peu confiantes ; elle serait ici totalement "
  "inefficace. Seul roberta-squad2, entraîné explicitement à reconnaître l'absence de réponse, "
  "conserve une calibration stable. Ce résultat suggère que l'entraînement à l'abstention a un "
  "effet bénéfique qui dépasse les questions sans réponse.")
H1("6.3 Deux résultats contre-intuitifs")
P("Deux résultats vont à l'encontre de l'intuition. D'une part, des paragraphes proches du "
  "thème aident la classification au lieu de la gêner, puisqu'ils partagent l'étiquette du "
  "signal. Ce qui est du bruit pour une tâche de localisation est un renfort pour une tâche de "
  "catégorisation. D'autre part, l'encodeur dense minilm, réputé capter le sens plutôt que les "
  "mots, est le plus trompé par les négatifs difficiles, avec 65,7 % de Recall@1 contre 78 à "
  "79 % pour les modèles lexicaux. Les négatifs difficiles ont été choisis précisément pour leur "
  "proximité sémantique, par un encodeur de la même famille que minilm ; il est donc logique que "
  "ce soit lui qui les confonde le plus avec le signal, alors que les modèles lexicaux "
  "s'appuient sur les mots exacts de la question.")
H1("6.4 La primauté du premier paragraphe")
P("Face à une contradiction, les modèles de question-réponse ne choisissent pas au hasard : ils "
  "favorisent le paragraphe qui apparaît en premier. Pour distilbert-squad, l'écart atteint "
  "près de vingt points. Ce biais de primauté a une conséquence pratique directe dans les "
  "systèmes qui concatènent plusieurs documents avant de les lire : l'ordre de présentation des "
  "documents, souvent déterminé par un moteur de recherche, influence la réponse finale "
  "lorsqu'ils se contredisent. Il ne s'agit plus seulement de trouver le bon document, mais de le "
  "placer avant les autres.")
H1("6.5 Limites de l'étude")
P("Ces conclusions doivent être lues à la lumière de plusieurs limites. La taille de "
  "l'échantillon d'abord : avec 300 exemples par tâche, les intervalles de confiance atteignent "
  "environ ±5,5 points autour de 50 %, de sorte que des effets plus faibles peuvent passer "
  "inaperçus. Nous avons toutefois rapporté les tailles d'effet avec leurs intervalles dans tous "
  "les cas, significatifs ou non. Le benchmark est ensuite limité à l'anglais, et la sélection "
  "des négatifs difficiles dépend de l'encodeur mpnet utilisé pour les choisir ; un autre "
  "sélecteur aurait produit d'autres paragraphes.")
P("La conception elle-même impose une asymétrie : comme le signal est fixe et que <i>N</i> = 1 "
  "constitue le contexte le plus propre, la plupart des courbes ne peuvent que décroître, et seule "
  "la redondance teste réellement la possibilité d'un gain. Le remplissage provient en outre de "
  "Wikipedia, et il est probable qu'un autre corpus attirerait les classifieurs vers une autre "
  "classe que <i>Sci/Tech</i>. Enfin, le remplacement d'entités produit parfois des phrases peu "
  "naturelles : dans l'exemple de l'annexe A, « the 17th century » devient « between 2004 and "
  "2014 », ce qui donne une phrase grammaticalement maladroite. Un lecteur humain repérerait "
  "l'anomalie ; le fait que les modèles s'y laissent prendre dans un tiers des cas renforce "
  "plutôt notre conclusion, mais une contradiction plus fluide serait probablement encore plus "
  "trompeuse.")

# ================================================================== CONCLUSION
story.append(PageBreak())
story.append(Chapter("Conclusion générale"))
story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph("Conclusion générale et perspectives", h0))
rule()
story.append(Spacer(1, 0.5 * cm))
P("Ce projet a conçu, construit et exploité un banc d'essai contrôlé pour mesurer la saturation "
  "sémantique des modèles NLP classiques. En faisant varier séparément la quantité, la "
  "pertinence, la redondance, la cohérence et la position du contexte, sur trois tâches et huit "
  "modèles, il apporte une réponse nette à la question de départ : pour les modèles étudiés, plus "
  "de contexte n'aide jamais au-delà de cinq paragraphes, et il nuit souvent.")
P("La gravité de la dégradation dépend moins de la sophistication du modèle que de la manière "
  "dont il agrège le texte. Les modèles qui résument tout le contexte en une seule représentation "
  "s'effondrent au niveau du hasard, en basculant vers la classe qui ressemble le plus au "
  "remplissage, tout en restant sûrs d'eux. Ceux qui examinent chaque passage séparément "
  "résistent à la dilution, mais restent vulnérables aux contradictions, qu'ils tranchent souvent "
  "en faveur du premier paragraphe lu. Sur le plan méthodologique, le projet montre l'intérêt "
  "d'un protocole rigoureux : les hypothèses fixées à l'avance ont permis d'identifier sans "
  "ambiguïté ce qui était attendu, ce qui l'était partiellement et ce qui ne l'était pas.")
P("Ce travail nous a également permis de mettre en pratique de nombreuses compétences du "
  "module : manipulation de grands corpus, reconnaissance d'entités nommées, traduction "
  "automatique, modèles de recherche d'information, fine-tuning et inférence de Transformers, "
  "ainsi que l'évaluation statistique rigoureuse des modèles. Les difficultés rencontrées, comme "
  "la suppression du pipeline de question-réponse dans transformers 5 ou les phrases omises par "
  "le traducteur, ont été autant d'occasions de comprendre en profondeur les outils utilisés.")
P("Plusieurs prolongements se dégagent naturellement. Il serait utile d'étendre le benchmark à "
  "d'autres langues, notamment le français et l'arabe, et à d'autres corpus de remplissage afin "
  "de vérifier si le biais vers une classe dépend de la source. L'augmentation du nombre "
  "d'exemples resserrerait les intervalles de confiance et permettrait de détecter des effets "
  "plus fins. Des paraphrases plus diverses, produites par exemple par un modèle de reformulation "
  "dédié, permettraient de tester à nouveau l'hypothèse H5. Enfin, appliquer le même protocole à "
  "des modèles de langage génératifs dotés de longues fenêtres de contexte permettrait de situer "
  "ces modèles récents par rapport aux références classiques étudiées ici, et de vérifier si les "
  "mêmes mécanismes de dilution et de primauté s'y retrouvent.")

# ================================================================== BIBLIOGRAPHIE
story.append(PageBreak())
story.append(Chapter("Bibliographie"))
story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph("Bibliographie", h0))
rule()
story.append(Spacer(1, 0.5 * cm))
refs = [
    "Devlin, J., Chang, M.-W., Lee, K. et Toutanova, K. (2019). BERT: Pre-training of Deep "
    "Bidirectional Transformers for Language Understanding. <i>Proceedings of NAACL-HLT</i>.",
    "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. <i>Scandinavian "
    "Journal of Statistics</i>, 6(2), 65–70.",
    "Efron, B. et Tibshirani, R. (1993). <i>An Introduction to the Bootstrap</i>. Chapman &amp; "
    "Hall.",
    "Guo, C., Pleiss, G., Sun, Y. et Weinberger, K. Q. (2017). On Calibration of Modern Neural "
    "Networks. <i>Proceedings of ICML</i>.",
    "Jia, R. et Liang, P. (2017). Adversarial Examples for Evaluating Reading Comprehension "
    "Systems. <i>Proceedings of EMNLP</i>.",
    "Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. et Liang, P. "
    "(2024). Lost in the Middle: How Language Models Use Long Contexts. <i>Transactions of the "
    "ACL</i>.",
    "Liu, Y., Ott, M., Goyal, N. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining "
    "Approach. <i>arXiv:1907.11692</i>.",
    "Rajpurkar, P., Jia, R. et Liang, P. (2018). Know What You Don't Know: Unanswerable "
    "Questions for SQuAD. <i>Proceedings of ACL</i>.",
    "Reimers, N. et Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese "
    "BERT-Networks. <i>Proceedings of EMNLP</i>.",
    "Robertson, S. et Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and "
    "Beyond. <i>Foundations and Trends in Information Retrieval</i>, 3(4).",
    "Sanh, V., Debut, L., Chaumond, J. et Wolf, T. (2019). DistilBERT, a Distilled Version of "
    "BERT. <i>arXiv:1910.01108</i>.",
    "Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., Schärli, N. et Zhou, D. "
    "(2023). Large Language Models Can Be Easily Distracted by Irrelevant Context. "
    "<i>Proceedings of ICML</i>.",
    "Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. et Gurevych, I. (2021). BEIR: A "
    "Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. "
    "<i>NeurIPS Datasets and Benchmarks</i>.",
    "Tiedemann, J. et Thottingal, S. (2020). OPUS-MT: Building Open Translation Services for "
    "the World. <i>Proceedings of EAMT</i>.",
    "Vaswani, A., Shazeer, N., Parmar, N. et al. (2017). Attention Is All You Need. "
    "<i>Advances in NeurIPS</i>.",
    "Zhang, X., Zhao, J. et LeCun, Y. (2015). Character-level Convolutional Networks for Text "
    "Classification. <i>Advances in NeurIPS</i>.",
]
refstyle = ParagraphStyle("ref", parent=body0, fontSize=11, leading=14.5, leftIndent=24,
                          firstLineIndent=-24, alignment=TA_LEFT, spaceAfter=6)
for i, r in enumerate(sorted(refs), 1):
    story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", refstyle))

# ================================================================== ANNEXES
story.append(PageBreak())
story.append(Chapter("Annexes"))
story.append(Spacer(1, 1.6 * cm))
story.append(Paragraph("Annexes", h0))
rule()
story.append(Spacer(1, 0.4 * cm))
story.append(Paragraph("Annexe A. Exemple d'échantillon de contradiction", h1))
P("L'échantillon ci-dessous est extrait tel quel du jeu de données v0.1. Il s'agit d'une "
  "question de la tâche de question-réponse, sur l'axe de contradiction, à <i>N</i> = 1 et avec "
  "le paragraphe contradictoire placé avant le signal. La différence entre les deux paragraphes "
  "porte uniquement sur l'entité réponse, en gras.")
P0("<b>Question :</b> <i>When were the shortcomings of Aristotle's physics overcome?</i><br/>"
   "<b>Réponses acceptées :</b> « 17th century », « the 17th century ».<br/>"
   "<b>Entité de remplacement :</b> « between 2004 and 2014 » (type DATE).")
P0("<b>Paragraphe contradictoire (placé en premier) :</b>")
story.append(Paragraph(
    "The shortcomings of Aristotelian physics would not be fully corrected until <b>between "
    "2004 and 2014</b> work of Galileo Galilei, who was influenced by the late Medieval idea "
    "that objects in forced motion carried an innate force of impetus. Galileo constructed an "
    "experiment in which stones and cannonballs were both rolled down an incline to disprove the "
    "Aristotelian theory of motion early in <b>between 2004 and 2014</b>. He showed that the "
    "bodies were accelerated by gravity to an extent that was independent of their mass and "
    "argued that objects retain their velocity unless acted on by a force, for example "
    "friction.", quote))
P0("<b>Paragraphe signal (gold) :</b>")
story.append(Paragraph(
    "The shortcomings of Aristotelian physics would not be fully corrected until <b>the 17th "
    "century</b> work of Galileo Galilei, who was influenced by the late Medieval idea that "
    "objects in forced motion carried an innate force of impetus. Galileo constructed an "
    "experiment in which stones and cannonballs were both rolled down an incline to disprove the "
    "Aristotelian theory of motion early in <b>the 17th century</b>. He showed that the bodies "
    "were accelerated by gravity to an extent that was independent of their mass and argued that "
    "objects retain their velocity unless acted on by a force, for example friction.", quote))
P0("<b>Paraphrase du signal par rétro-traduction (axe de redondance) :</b>")
story.append(Paragraph(
    "The shortcomings of Aristotelian physics would not be completely corrected before the work "
    "of the 17th century of Galilee, which was influenced by the late medieval idea that objects "
    "in forced movement carried an innate force of impulse. Galileo built an experiment in which "
    "stones and cannon balls were both rolled down a slope to refute the Aristotelian theory of "
    "movement at the beginning of the 17th century. He showed that the bodies were accelerated "
    "by gravity to an extent independent of their mass and argued that the objects kept their "
    "speed unless they acted by force, for example friction.", quote))
P("La paraphrase conserve la réponse et l'essentiel du vocabulaire, tout en modifiant plusieurs "
  "formulations (« fully corrected until » devient « completely corrected before », « incline » "
  "devient « slope »). Elle illustre pourquoi copies et paraphrases donnent des résultats si "
  "proches : la rétro-traduction diversifie peu le lexique.")

story.append(Paragraph("Annexe B. Temps de calcul", h1))
P("Le tableau suivant donne le temps d'évaluation de chaque modèle sur les 25 950 échantillons, "
  "mesuré sur le GPU T4 de Kaggle. Les modèles lexicaux sont quasi instantanés ; les modèles qui "
  "lisent l'intégralité du contexte par blocs ou par fenêtres, bert-chunk-mean et roberta-squad2, "
  "sont les plus coûteux.")
secs = {"tfidf-lr": 44.5, "bert-truncate": 157.5, "bert-chunk-mean": 698.9, "tfidf": 10.8,
        "bm25": 8.3, "minilm": 3.6, "distilbert-squad": 339.3, "roberta-squad2": 706.8}
rows = [["Modèle", "Tâche", "Temps (s)", "Part du total"]]
tot = sum(secs.values())
for m in ALL:
    rows.append([m, TASK_OF[m], f"{secs[m]:.1f}".replace(".", ","), pct(secs[m] / tot)])
rows.append(["<b>Total</b>", "", f"<b>{tot:.1f}</b>".replace(".", ","), "<b>100 %</b>"])
table(rows, [0.3, 0.25, 0.2, 0.25], "Temps d'évaluation par modèle sur GPU T4", 2)

story.append(Paragraph("Annexe C. Reproduction des résultats", h1))
P("L'ensemble du travail est reproductible. Le jeu de données v0.1, accompagné de ses empreintes "
  "SHA-256, les prédictions brutes, les métriques avec leurs intervalles de confiance et les "
  "figures en anglais et en français sont versionnés dans le dépôt du projet. Après "
  "installation du paquet, trois commandes suffisent à reconstruire la chaîne complète : la "
  "construction du jeu de données, l'exécution des modèles et la production du rapport.")
story.append(Preformatted(
    "pip install -e .\n"
    "ssb build  --config configs/main.yaml\n"
    "ssb run    --config configs/main.yaml\n"
    "ssb report --config configs/main.yaml --fr", ParagraphStyle("m2", parent=mono,
                                                                fontSize=10, leading=14,
                                                                leftIndent=1 * cm)))
story.append(Spacer(1, 8))
P("La dernière commande régénère à l'identique les résultats à partir des prédictions "
  "versionnées ; <font name='Mono'>kaggle/run.py</font> relance toute la chaîne sur un T4 en "
  "une heure environ.")

doc = Doc(OUT)
for _t in (toc, lof, lot):
    _t.formatter = doc.label
doc.multiBuild(story)
print("ok", OUT)
