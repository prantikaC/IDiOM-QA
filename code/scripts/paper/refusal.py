"""Refusal detector for IDiOM-QA answers: an answer counts as a refusal when
its first sentence says the corpus lacks the information. Partial answers
("... The documents do not specify: X") are answers, not refusals."""
import re

_PAT = re.compile(
    r"could not find (sufficient |any )?information|(do|does) not (contain|include|provide|mention|specify|have)"
    r"|no (information|mention|details|data|record)|not (mentioned|specified|provided|available|stated|found)"
    r"|unable to (find|answer|determine|provide)|cannot (be determined|answer|find|provide)|^none yet"
    r"|insufficient information|there is no (specific )?(information|mention|record)|is not possible to (answer|determine)"
    r"|no answer|out[- ]of[- ](domain|scope)|outside the (scope|domain)|not (covered|documented|addressed)"
    r"|no specific (mention|information|details)",
    re.I)


def first_sentence(text: str) -> str:
    text = " ".join(str(text).split())
    m = re.search(r"(?<=[.!?])\s", text)
    return text[:m.start()] if m else text


def is_refusal(answer: str) -> bool:
    return bool(_PAT.search(first_sentence(answer)))
