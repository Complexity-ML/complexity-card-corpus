from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal


Tense = Literal["present", "past", "future"]
Aspect = Literal["simple", "progressive", "perfect", "perfect_progressive"]
Number = Literal["singular", "plural"]


_IRREGULAR_FORMS: dict[str, tuple[str, str]] = {
    "be": ("was", "been"),
    "begin": ("began", "begun"),
    "break": ("broke", "broken"),
    "bring": ("brought", "brought"),
    "build": ("built", "built"),
    "buy": ("bought", "bought"),
    "catch": ("caught", "caught"),
    "choose": ("chose", "chosen"),
    "come": ("came", "come"),
    "cost": ("cost", "cost"),
    "do": ("did", "done"),
    "draw": ("drew", "drawn"),
    "drink": ("drank", "drunk"),
    "drive": ("drove", "driven"),
    "eat": ("ate", "eaten"),
    "fall": ("fell", "fallen"),
    "feel": ("felt", "felt"),
    "find": ("found", "found"),
    "get": ("got", "gotten"),
    "give": ("gave", "given"),
    "go": ("went", "gone"),
    "grow": ("grew", "grown"),
    "have": ("had", "had"),
    "hear": ("heard", "heard"),
    "hold": ("held", "held"),
    "keep": ("kept", "kept"),
    "know": ("knew", "known"),
    "lead": ("led", "led"),
    "leave": ("left", "left"),
    "lose": ("lost", "lost"),
    "make": ("made", "made"),
    "meet": ("met", "met"),
    "pay": ("paid", "paid"),
    "read": ("read", "read"),
    "run": ("ran", "run"),
    "say": ("said", "said"),
    "see": ("saw", "seen"),
    "send": ("sent", "sent"),
    "set": ("set", "set"),
    "show": ("showed", "shown"),
    "sit": ("sat", "sat"),
    "speak": ("spoke", "spoken"),
    "stand": ("stood", "stood"),
    "take": ("took", "taken"),
    "teach": ("taught", "taught"),
    "tell": ("told", "told"),
    "think": ("thought", "thought"),
    "understand": ("understood", "understood"),
    "wake": ("woke", "woken"),
    "wear": ("wore", "worn"),
    "win": ("won", "won"),
    "write": ("wrote", "written"),
}

_DOUBLE_FINAL = frozenset(
    {
        "admit",
        "begin",
        "commit",
        "control",
        "drop",
        "fit",
        "forget",
        "get",
        "occur",
        "permit",
        "plan",
        "prefer",
        "refer",
        "regret",
        "run",
        "set",
        "sit",
        "stop",
        "submit",
    }
)

_AN_INITIALISMS = frozenset("AEFHILMNORSX")
_SILENT_H_PREFIXES = ("heir", "honest", "honor", "hour")
_CONSONANT_SOUND_PREFIXES = (
    "euro",
    "one",
    "once",
    "uni",
    "use",
    "user",
    "usual",
    "usable",
    "utility",
    "ufo",
)
_ARTICLE_PATTERN = re.compile(
    r"\b(?P<article>a|an)\s+(?P<word>[A-Za-z][A-Za-z0-9'-]*)",
    re.IGNORECASE,
)


def indefinite_article(value: str) -> str:
    """Choose ``a`` or ``an`` from common English pronunciation rules."""
    word = value.strip().split(maxsplit=1)[0] if value.strip() else ""
    if not word:
        raise ValueError("an indefinite article requires a following word")
    normalized = word.lower().strip("\"'([{<")
    if not normalized:
        raise ValueError("an indefinite article requires an alphabetic word")
    if word.isupper() and word[0] in _AN_INITIALISMS:
        return "an"
    if normalized.startswith(_SILENT_H_PREFIXES):
        return "an"
    if normalized.startswith(_CONSONANT_SOUND_PREFIXES):
        return "a"
    return "an" if normalized[0] in "aeiou" else "a"


def correct_indefinite_articles(value: str) -> str:
    """Correct explicit a/an pairs without otherwise paraphrasing text."""

    def replace(match: re.Match[str]) -> str:
        expected = indefinite_article(match.group("word"))
        if match.group("article")[0].isupper():
            expected = expected.capitalize()
        return f"{expected} {match.group('word')}"

    return _ARTICLE_PATTERN.sub(replace, value)


@dataclass(frozen=True)
class VerbPhrase:
    """A lexical verb lemma followed by its unchanged complement."""

    lemma: str
    complement: str = ""

    @classmethod
    def parse(cls, value: str) -> "VerbPhrase":
        parts = value.strip().split(maxsplit=1)
        if not parts or not parts[0].isalpha():
            raise ValueError("verb phrases must begin with an alphabetic lemma")
        return cls(parts[0].lower(), parts[1] if len(parts) == 2 else "")

    def join(self, verb_form: str) -> str:
        return f"{verb_form} {self.complement}" if self.complement else verb_form


@dataclass(frozen=True)
class VerbFeatures:
    tense: Tense = "present"
    aspect: Aspect = "simple"
    person: Literal[1, 2, 3] = 3
    number: Number = "singular"
    modal: str | None = None
    negated: bool = False
    interrogative: bool = False

    def __post_init__(self) -> None:
        if self.modal is not None:
            modal = self.modal.strip().lower()
            if not modal.isalpha():
                raise ValueError("modal must be one alphabetic word")
            object.__setattr__(self, "modal", modal)


def third_person_singular(lemma: str) -> str:
    lemma = lemma.lower()
    if lemma == "be":
        return "is"
    if lemma == "have":
        return "has"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return f"{lemma[:-1]}ies"
    if lemma.endswith(("s", "sh", "ch", "x", "z", "o")):
        return f"{lemma}es"
    return f"{lemma}s"


def regular_past(lemma: str) -> str:
    lemma = lemma.lower()
    if lemma.endswith("e"):
        return f"{lemma}d"
    if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
        return f"{lemma[:-1]}ied"
    if lemma in _DOUBLE_FINAL:
        return f"{lemma}{lemma[-1]}ed"
    return f"{lemma}ed"


def past_tense(lemma: str) -> str:
    lemma = lemma.lower()
    return _IRREGULAR_FORMS.get(lemma, (regular_past(lemma), ""))[0]


def past_participle(lemma: str) -> str:
    lemma = lemma.lower()
    irregular = _IRREGULAR_FORMS.get(lemma)
    return irregular[1] if irregular else regular_past(lemma)


def present_participle(lemma: str) -> str:
    lemma = lemma.lower()
    if lemma == "be":
        return "being"
    if lemma.endswith("ie"):
        return f"{lemma[:-2]}ying"
    if lemma in _DOUBLE_FINAL:
        return f"{lemma}{lemma[-1]}ing"
    if lemma.endswith("e") and not lemma.endswith(("ee", "oe", "ye")):
        return f"{lemma[:-1]}ing"
    return f"{lemma}ing"


def _be_form(features: VerbFeatures) -> str:
    if features.tense == "past":
        return "was" if features.number == "singular" and features.person != 2 else "were"
    if features.person == 1 and features.number == "singular":
        return "am"
    if features.person == 3 and features.number == "singular":
        return "is"
    return "are"


def _have_form(features: VerbFeatures) -> str:
    if features.tense == "past":
        return "had"
    if features.person == 3 and features.number == "singular":
        return "has"
    return "have"


def _positive_parts(phrase: VerbPhrase, features: VerbFeatures) -> list[str]:
    lemma = phrase.lemma
    if features.modal:
        if features.aspect == "simple":
            return [features.modal, phrase.join(lemma)]
        if features.aspect == "progressive":
            return [features.modal, "be", phrase.join(present_participle(lemma))]
        if features.aspect == "perfect":
            return [features.modal, "have", phrase.join(past_participle(lemma))]
        return [features.modal, "have", "been", phrase.join(present_participle(lemma))]

    if features.tense == "future":
        if features.aspect == "simple":
            return ["will", phrase.join(lemma)]
        if features.aspect == "progressive":
            return ["will", "be", phrase.join(present_participle(lemma))]
        if features.aspect == "perfect":
            return ["will", "have", phrase.join(past_participle(lemma))]
        return ["will", "have", "been", phrase.join(present_participle(lemma))]

    if features.aspect == "progressive":
        return [_be_form(features), phrase.join(present_participle(lemma))]
    if features.aspect == "perfect":
        return [_have_form(features), phrase.join(past_participle(lemma))]
    if features.aspect == "perfect_progressive":
        return [_have_form(features), "been", phrase.join(present_participle(lemma))]

    if lemma == "be":
        return [_be_form(features), *([phrase.complement] if phrase.complement else [])]
    if features.tense == "past":
        return [phrase.join(past_tense(lemma))]
    if features.person == 3 and features.number == "singular":
        return [phrase.join(third_person_singular(lemma))]
    return [phrase.join(lemma)]


def realize_verb_phrase(value: str, features: VerbFeatures = VerbFeatures()) -> str:
    """Inflect an English verb phrase without adding a grammatical subject."""
    phrase = VerbPhrase.parse(value)
    parts = _positive_parts(phrase, features)
    has_auxiliary = len(parts) > 1 or phrase.lemma == "be" or features.modal is not None
    if not features.negated:
        return " ".join(parts)
    if has_auxiliary:
        return " ".join((parts[0], "not", *parts[1:]))
    auxiliary = "did" if features.tense == "past" else (
        "does" if features.person == 3 and features.number == "singular" else "do"
    )
    return f"{auxiliary} not {phrase.join(phrase.lemma)}"


def realize_clause(
    subject: str,
    value: str,
    features: VerbFeatures = VerbFeatures(),
) -> str:
    """Realize a declarative or interrogative English clause."""
    subject = subject.strip()
    if not subject:
        raise ValueError("a clause requires a subject")
    phrase = VerbPhrase.parse(value)
    if not features.interrogative:
        return f"{subject} {realize_verb_phrase(value, features)}"

    parts = _positive_parts(phrase, features)
    has_auxiliary = len(parts) > 1 or phrase.lemma == "be" or features.modal is not None
    if has_auxiliary:
        tail = parts[1:]
        if features.negated:
            tail = ["not", *tail]
        return " ".join((parts[0], subject, *tail)).strip()

    auxiliary = "did" if features.tense == "past" else (
        "does" if features.person == 3 and features.number == "singular" else "do"
    )
    negation = " not" if features.negated else ""
    return f"{auxiliary} {subject}{negation} {phrase.join(phrase.lemma)}"


def verb_forms(value: str) -> dict[str, str]:
    """Return the core forms used by audits and lexical-card tooling."""
    phrase = VerbPhrase.parse(value)
    return {
        "base": phrase.join(phrase.lemma),
        "third_person_singular": phrase.join(third_person_singular(phrase.lemma)),
        "past": phrase.join(past_tense(phrase.lemma)),
        "past_participle": phrase.join(past_participle(phrase.lemma)),
        "present_participle": phrase.join(present_participle(phrase.lemma)),
    }


def audit_verb_phrases(values: list[str]) -> dict[str, int]:
    """Verify and summarize the morphological coverage of lexical intents."""
    phrases = sorted(set(values))
    realized = [verb_forms(value) for value in phrases]
    if any(not form for forms in realized for form in forms.values()):
        raise ValueError("morphological realization produced an empty form")
    return {
        "intent_phrases": len(phrases),
        "unique_lemmas": len({VerbPhrase.parse(value).lemma for value in phrases}),
        "forms_per_intent": 5,
        "forms_generated": sum(len(forms) for forms in realized),
        "unique_realized_forms": len(
            {form for forms in realized for form in forms.values()}
        ),
    }
