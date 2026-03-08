import langchain_core.prompts
import pydantic


class RAGEvaluation(pydantic.BaseModel):

    factual_existence: int = pydantic.Field(
        ...,
        ge=0,
        le=1,
        description=(
            "Whether the model correctly decided if an answer exists in the Retrieved Facts.\n"
            "1 = The model correctly determined whether the facts contain a direct or partial answer.\n"
            "0 = The model incorrectly said answer_found=false when relevant text exists, "
            "or incorrectly said answer_found=true when no clear answer exists."
        )
    )

    factual_faithfulness: int = pydantic.Field(
        ...,
        ge=0,
        le=5,
        description=(
            "How strictly the model answer is supported by the provided facts.\n"
            "5 = Every claim is directly supported by the facts.\n"
            "4 = Mostly supported; very minor implied statements.\n"
            "3 = Mix of supported and weakly supported claims.\n"
            "2 = Significant unsupported or inferred claims.\n"
            "1 = Largely unsupported by the facts.\n"
            "0 = Mostly or completely hallucinated.\n"
            "If the model ignored clearly relevant facts and answered 'not found', reduce score."
        )
    )

    query_relevance: int = pydantic.Field(
        ...,
        ge=0,
        le=5,
        description=(
            "How well the answer addresses the question using the given facts.\n"
            "5 = Directly answers the question clearly.\n"
            "4 = Answers the question but with minor irrelevance.\n"
            "3 = Partially answers the question.\n"
            "2 = Mostly indirect or avoids clear answer.\n"
            "1 = Barely related.\n"
            "0 = Does not address the question.\n"
            "If relevant statutory text exists but model says answer_found=false, query_relevance must be <= 2."
        )
    )

    legal_precision: int = pydantic.Field(
        ...,
        ge=0,
        le=4,
        description=(
            "Accuracy of legal terms, sections, and statutory references in the response.\n"
            "4 = Exact act names, sections, and correct terminology.\n"
            "3 = Mostly correct with minor imprecision.\n"
            "2 = Vague or partially incorrect references.\n"
            "1 = Incorrect legal terminology.\n"
            "0 = No legal precision or clearly wrong law."
        )
    )

    clarity: int = pydantic.Field(
        ...,
        ge=0,
        le=3,
        description=(
            "Clarity, structure, and readability of the final answer.\n"
            "3 = Very clear and structured.\n"
            "2 = Mostly clear.\n"
            "1 = Hard to follow.\n"
            "0 = Confusing."
        )
    )

    citation_quality: int = pydantic.Field(
        ...,
        ge=0,
        le=5,
        description=(
            "How well the model selected useful citations from the retrieved facts in the final response.\n"
            "5 = Cited all key relevant portions.\n"
            "4 = Cited most relevant parts.\n"
            "3 = Cited some relevant parts but missed others.\n"
            "2 = Cited few relevant parts or some irrelevant parts.\n"
            "1 = Mostly irrelevant citations.\n"
            "0 = Relevant text exists but model cited nothing or wrong text.\n"
            "If relevant text exists and citations are empty, score <= 2."
        )
    )

    explanation_from_citations: int = pydantic.Field(
        ...,
        ge=0,
        le=5,
        description=(
            "How well the model explained the content of its own citations in the final response.\n"
            "5 = Clear explanation grounded in cited text.\n"
            "4 = Mostly grounded with minor gaps.\n"
            "3 = Partial explanation.\n"
            "2 = Weak connection to citations.\n"
            "1 = Vague or inconsistent with citations.\n"
            "0 = Citations exist but no meaningful explanation.\n"
            "If relevant text exists but model says 'not found', score <= 2."
        )
    )

    @pydantic.computed_field
    @property
    def total(self)-> int:
        return (
            self.factual_existence
            + self.factual_faithfulness
            + self.query_relevance
            + self.legal_precision
            + self.clarity
            + self.citation_quality
            + self.explanation_from_citations
        )

def setup_evaluator_prompt(parser):
    return langchain_core.prompts.PromptTemplate(
        input_variables=[
            "question",
            "facts",
            "model_answer",
            "citations",
        ],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
        template="""
You are a strict and impartial evaluator for a legal RAG system.

Your task is to evaluate the MODEL ANSWER using ONLY the Retrieved Facts.

You must reason in the following order:

STEP 1:
Determine whether the Retrieved Facts contain statutory language that directly
or partially answers the Question.

STEP 2:
Determine whether the model correctly decided if an answer exists
(i.e., whether answer_found should have been true or false).

STEP 3:
Score all fields strictly according to the rubric.

Rules:
- Do NOT use outside legal knowledge.
- Do NOT assume missing facts.
- If unsure, prefer LOWER scores.
- Penalize unsupported claims.
- Penalize omission if relevant statutory text exists but the model ignored it.
- If relevant text exists and model outputs "answer not found", this is a major failure.

Mandatory Scoring Constraints:
- If relevant statutory text exists and model said answer_found=false:
    - factual_existence = 0
    - query_relevance <= 2
    - citation_quality <= 2
    - explanation_from_citations <= 2
- If no relevant text exists and model correctly abstained:
    - factual_existence = 1
- If model hallucinated beyond facts:
    - factual_faithfulness <= 2

Scoring:
- Use integer values only.
- total MUST equal the sum of all individual fields.

Return ONLY valid JSON matching the schema below.
Do NOT add explanations or extra text.

Schema:
{format_instructions}

---
Definitions:

Citations are verbatim text snippets copied from the Retrieved Facts.
They are not the model’s own words.

The explanation is the model’s summary of its own citations.

---
Inputs:

Question:
{question}

Retrieved Facts:
{facts}

Model Answer (explanation only):
{model_answer}

Model Citations:
{citations}
"""
    )