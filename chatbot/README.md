## Legal Data Setup

### Directory structure (after setup)

```text
chatbot/
├── README.md
├── centralActsDownloader.py
├── main.py
├── vectorDbSetup.py
└── legalos_rag/
    ├── README.md
    ├── __init__.py  
    ├── runRag.py
    ├── queryRewriter.py
    ├── sourceLoader.py
    ├── sourceReview.py
    ├── sectionLookup.py
    ├── structuredCall.py
    ├── questionSplitter.py
    ├── factExtractor.py
    ├── referenceAgent.py
    ├── keywordSearch.py
    ├── retriever.py
    ├── evidenceFilter.py
    ├── evidenceAgent.py
    ├── conditionCheck.py
    ├── applicabilityAgent.py
    ├── contextCompactor.py
    ├── answerAgent.py
    ├── evaluator.py
    ├── orchestrator.py
    ├── workflow.py
    └── prompt
        ├── prompts.py
        └── promptSchema.py
```

---

### Step 1: Download Central Acts PDFs locally

This script **requires a mandatory named argument** specifying where PDFs should be stored.

**Recommended path (used throughout this project):**

```text
./factsDB/centralActPdfs/data
```

Run:

```bash
python -m chatbot.centralActsDownloader --outputDir ./factsDB/centralActPdfs/data
```

This will create (under legalos project root):

```
factsDB/
└── centralActPdfs/
    ├── data/               # Downloaded PDFs
    └── failed_pdfs.txt     # Download failures (if any)
```

---

### Step 2: Create Vector Database and Embed PDFs

This script **requires two mandatory named arguments**:

1. **PDF directory** (input)
2. **Vector DB directory** (output)

**Recommended paths:**

```text
PDFs : ./factsDB/centralActPdfs/data
DB   : ./vectorDB
```

Run:

```bash
python -m chatbot.vectorDbSetup \
  --pdfActsDirectory ./factsDB/centralActPdfs/data \
  --vectordbDirectory ./vectorDB
```

This will:

- Create a **local, on-disk Qdrant vector database**
- Embed all PDFs using **BAAI/bge-small-en**
- Store vectors persistently inside `./vectorDB`

---

## Load the POC law sources

This optional setup imports the six pinned POC snapshots for the next retrieval
path. It does not change the existing Qdrant database or chatbot answers. Every
imported Act is marked `unreviewed`; checking PDF section boundaries, material
conditions, and exceptions is the next step in [issue #23](https://github.com/prasenjitb-cloud/legalos/issues/23).

Run these commands from the repository root. Export the JSON snapshots from the
fixed [reference POC commit](https://github.com/prasenjitb-cloud/legalos/commit/b9f02f5a5e5df7ce1b17b3dd1c0cb1a1df2eeab7):

```bash
# Needed once if the reference commit is not available locally.
git fetch origin poc/evidence-recovery-clean
mkdir -p ./factsDB/pocSources
git archive b9f02f5a5e5df7ce1b17b3dd1c0cb1a1df2eeab7:chatbot/legalos_rag/poc_v2/data | tar -x -C ./factsDB/pocSources
```

The export copies only source data; the loader does not import or run POC code.
Once the inputs are local, loading works offline without Ollama or embeddings.
The PDFs from the download step must be the exact snapshots listed in
`config/poc_sources.json`:

| Act | Snapshot JSON | PDF | Numbered sections |
| --- | --- | --- | ---: |
| POSH, 2013 | `posh.json` | `A2013-14.pdf` | 30 |
| Hindu Marriage, 1955 | `hma_1955.json` | `A1955-25.pdf` | 37 |
| Motor Vehicles, 1988 | `mva_1988.json` | `aA1988-59.pdf` | 256 |
| Juvenile Justice, 2015 | `jja_2015.json` | `a2016-2.pdf` | 112 |
| Central Goods and Services Tax, 2017 | `cgst_2017.json` | `a2017-12.pdf` | 188 |
| Consumer Protection, 2019 | `cpa_2019.json` | `eng201935.pdf` | 107 |

```bash
python -m chatbot.legalos_rag.sourceLoader --config ./config/poc_sources.json
```

The config supplies `source_directory`, `pdf_directory`, `output_file`, and the
six source entries with their expected hashes, identities, page counts, and
section counts. Relative paths resolve from the repository root. Change these
paths in a local config when the inputs are stored elsewhere.

The command validates all six snapshots and PDFs before writing
`factsDB/sectionCorpus/poc_sources.json`, containing 730 section records with
their source details and smaller excerpts. It prints each Act's section count
and the output path. Missing or changed files, duplicate Acts or sections, and
malformed section records fail the command. An existing output file is preserved;
choose a new `output_file` to repeat an import. Both source exports and generated
corpora are ignored by Git.

A current download may differ from the pinned PDF. A hash mismatch requires the
matching archived PDF or a separate source-update review; changing a hash is not
source-quality approval. The snapshots cover numbered sections, not associated
Rules or all schedules. See the [source schema](legalos_rag/README.md#sourceloaderpy)
for source versions, IDs, excerpt offsets, and review limitations.

## Check the POC law sources

The imported sections come from snapshots, so each Act is checked against its PDF
before the new retrieval path may use it. Run this after the loader, from the
repository root:

```bash
python -m chatbot.legalos_rag.sourceReview --config ./config/poc_sources.json
```

The command reads the section corpus written by the loader and the PDFs in
`pdf_directory`. It needs no model or network. It writes
`factsDB/sectionCorpus/poc_source_review.json` (the `review_file` in the config),
with a `passed` or `failed` status and a list of findings for each Act. An existing
review file is preserved; choose a new `review_file` to repeat the check. The file
is ignored by Git.

For every section the check confirms that the heading and number are found in
order on the recorded first page, that the section text contains the words the PDF
shows for that section, and that every `Provided that`, `Explanation`, `Exception`
and `Illustration` in the PDF is also in the section. An Act also fails if its PDF
page count or name does not match. Any finding fails the whole Act.

Result on the six pinned Acts: POSH, Hindu Marriage, Juvenile Justice, Central
Goods and Services Tax, and Consumer Protection pass. Motor Vehicles fails on one
section, `217A`: 12 PDF words are not in the section text. They appear to come from
a footnote about the First Schedule rather than from the section, but that has not
been confirmed, so the Act stays refused until it is reviewed.

A pass means the section text matches the PDF text at this level. It is not legal
approval, and it does not check Rules or Schedules. See the
[source review notes](legalos_rag/README.md#sourcereviewpy) for the exact rules and
the gate that refuses unchecked Acts.

## Fetch a complete section

A search hit is only a short excerpt, and a condition or proviso may sit just
outside it. This command returns the whole section for a section ID or any of its
excerpt IDs. Run it from the repository root after the loader and the check:

```bash
python -m chatbot.legalos_rag.sectionLookup --config ./config/poc_sources.json --id "<source_id>"
```

IDs are the `source_id` values in `factsDB/sectionCorpus/poc_sources.json`, for
example `posh_2013:<pdf_sha256>:section:4` or one of its excerpts, which end in
`:chars:<start>-<end>`. The command prints one JSON object with the Act ID and name,
`source_version`, the section's `source_id`, number, title, full `text`, PDF `pages`
and `footnotes`. It needs no model or network.

It fails with a clear message when the ID is malformed, names an unknown Act,
section or excerpt, was made for a different version of the PDF, or belongs to an
Act that has not passed the source check above, such as Motor Vehicles at present.

## Split a question

A user may ask about different legal matters in one message. This command splits a
question into independent parts only when needed, and keeps each part linked to the
facts it relies on. It needs Ollama running with the model named in the config. Run it
from the repository root:

```bash
python -m chatbot.legalos_rag.questionSplitter --config ./config/rag_v1.json --question "<question>"
```

The config supplies only `model.model_name`. The command prints the original
`question` and its `parts`; each part has an `id`, a complete `question` and the
`facts` it depends on, copied exactly from the original question. It does not search
the law and does not change the chatbot.

Several details of one legal problem stay in one part. When the model fails, returns
more than four parts, or rewrites a part with words the user did not use, the whole
question is kept as one part. See the
[splitter notes](legalos_rag/README.md#questionsplitterpy) for what was measured.

## Record the facts a user states

Legal answers can depend on facts such as a worker count, a date or a relationship.
This command records only the facts the user wrote, each as an exact quote. A fact the
user did not give is never filled in; it is simply absent, so later steps treat it as
unknown. It needs Ollama running with the model named in the config. Run it from the
repository root:

```bash
python -m chatbot.legalos_rag.factExtractor --config ./config/rag_v1.json --question "<question>"
```

The command prints the original `question` and its `facts`, each with an `id` and the
`quote`. It does not search the law and does not change the chatbot. See the
[extractor notes](legalos_rag/README.md#factextractorpy) for the checks and results.

## Resolve a named Act and section

When a question names a section, such as "section 13 of the Hindu Marriage Act", the
section should be confirmed from the checked source, not guessed from search results.
This command finds each named section and Act and returns the verified, complete
provision, or says why it cannot. It needs the loader and the source check to have been
run, but no model or network. Run it from the repository root:

```bash
python -m chatbot.legalos_rag.referenceAgent --config ./config/poc_sources.json --question "<question>"
```

The command prints one result for each section number named. A `verified` result has the
full provision (the same output as the section lookup above). Other results have a
`status` and a `reason`:

| Status | Meaning |
| --- | --- |
| `ambiguous` | No single Act is named for the section; the result lists candidate sections from the checked Acts. |
| `unsupported_act` | The question names an Act that is not one of the six. |
| `act_not_verified` | The Act did not pass its source check, such as Motor Vehicles at present. |
| `not_found` | The Act has no such numbered section, or the year given is not the supported version. |

A question that names no section prints `[]`. Each source in `config/poc_sources.json`
can list short `aliases` (such as `POSH`) that are recognised as the Act. See the
[agent notes](legalos_rag/README.md#referenceagentpy) for the matching rules.

## Search a verified Act by keyword

The current vector search often misses an exact legal term or a section number. This
command ranks the excerpts of one checked Act by the words in the query (BM25). It needs
the loader and the source check to have been run, but no model, vector database or
network. Run it from the repository root:

```bash
python -m chatbot.legalos_rag.keywordSearch --config ./config/poc_sources.json --act posh_2013 --query "Local Committee" --k 5
```

`--act` is an Act ID from the config and `--k` is how many excerpts to return (default 5).
The command prints the hits, best first, each with its `rank`, `score`, excerpt `source_id`,
`section_source_id`, section number and title, and the excerpt `text`. Scores only order the
results of one query; they are not probabilities. An Act that has not passed its source check
is refused. A query with no matching word prints `[]`. See the
[search notes](legalos_rag/README.md#keywordsearchpy) for how it works and what was measured.

## Retrieve sections of a verified Act

One command finds a rule, a filing process or an outcome. It searches one checked Act by
keyword and by meaning, merges the two result lists, and returns the best matches as
complete sections. It needs the loader, the source check, and the vector database from
Step 2 above. It needs no model unless `--rewrite` is used, in which case Ollama must be
running. Run it from the repository root:

```bash
python -m chatbot.legalos_rag.retriever --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "How do two spouses get divorced together by agreement?" \
  --target "procedure for mutual consent divorce petition"
```

`--target` says what evidence is wanted ("definition of sexual harassment", "complaint filing
process", "penalty for non-compliance"); the same command serves every kind of request.
`--rewrite` also searches with the query rewriter's legal phrasings, which can help when the
question is informal. `--max-sections` sets how many sections come back (default 3). The
vector database path and model come from `--rag-config`, the same file the chatbot uses.

The output lists `results`, best first. Each has the `score`, whether it was `found_by`
`keyword`, `vector` or both, the `hit_source_ids` of the matching excerpts, and the complete
`section` (text, pages, footnotes). An Act that has not passed its source check is refused.
See the [retriever notes](legalos_rag/README.md#retrieverpy) for how results are merged and
what was measured.

## Filter retrieved sections by the evidence wanted

A search can return a section that only mentions the topic, such as a duties section for a
question about filing a complaint. This command retrieves sections as above and then asks
the local model, for each one, which sentence states the evidence wanted. It needs the same
inputs as the retriever, and Ollama running with the model in the config:

```bash
python -m chatbot.legalos_rag.evidenceFilter --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act posh_2013 --question "How do I file a complaint of sexual harassment?" --target "complaint filing process"
```

The command prints three lists: `kept` (each with the `support_quote`, the sentence that
states the evidence), `rejected` (with a `reason`) and `unchecked` (the model failed on it).
It does not decide an answer. See the
[filter notes](legalos_rag/README.md#evidencefilterpy) for how it works and what was measured.

## Gather evidence for each part of a question

This command splits a question into parts, then for each part names the evidence wanted,
retrieves sections of one checked Act, keeps those that state it, and returns them with their
sources. It needs the same inputs as the retriever and Ollama running with the model in the
config:

```bash
python -m chatbot.legalos_rag.evidenceAgent --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "What are the conditions for divorce by mutual consent, and what does the Act provide about custody of minor children?"
```

The output has one entry per question part with its `target`, a `status` of `found` or
`not_found`, the `evidence` (section, source ID, the `support_quote` sentence, pages and source
version), and the sections `rejected` or `unchecked`. A part that finds nothing names what was
not found in `missing`. A found part says `completeness: not_checked`: the command does not
decide whether the sections answer everything asked. It does not write an answer. See the
[agent notes](legalos_rag/README.md#evidenceagentpy) for what was measured.

## Check a section's conditions against stated facts

A rule applies only if its conditions are met and no exception applies. This command lists the
conditions and exceptions of one complete section and marks each `MET`, `NOT_MET` or `UNKNOWN`
against facts the user stated. It needs Ollama running with the model in the config. Run it
from the repository root; `--fact` is repeated for each fact, quoted as the user wrote it:

```bash
python -m chatbot.legalos_rag.conditionCheck --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --id "<section source_id>" --fact "The bride is 19 and the groom is 25"
```

Each condition shows its exact clause, its status and the facts it cites. A condition is `MET` or
`NOT_MET` only when it cites a stated fact; with no facts every condition is `UNKNOWN`, and the
model is not called. The `overall` result is `not_met`, `excepted`, `met` or `unknown`.

**Treat the statuses as an assessment, not a finding.** The local model sometimes judges a
condition wrongly even when it cites a real fact, for example on an age comparison; see the
[condition check notes](legalos_rag/README.md#conditioncheckpy) for what was measured. The result
should be reviewed by the evaluator before anything is said about a user's case.

## Assess what can be said about a situation

For a question about the user's own situation, this command reads the facts they stated,
gathers the evidence, checks the rule's conditions against those facts, and reports what can
and cannot be said. It needs the same inputs as the evidence agent and Ollama running:

```bash
python -m chatbot.legalos_rag.applicabilityAgent --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "We have been living separately for three months and we both want a divorce. Can we get a divorce by mutual consent?"
```

Each question part gets a `finding`: `needs_fact` (with an `ask` quoting an open condition of the
law, and a `conditional` note), `not_met_on_stated_facts`, `exception_applies`,
`conditions_met_on_stated_facts`, `cannot_assess`, `general` (a question about the law, not the
user's situation, so no facts are requested) or `no_evidence`. The conditions it rests on are
listed with their statuses and the facts they cite.

**No finding is a personal conclusion**, and the `ask` is not yet a polished question for the
user. See the [agent notes](legalos_rag/README.md#applicabilityagentpy).

## Compact the evidence for the answer step

The answer step needs the sections found for every question part, without repeating the same
text. This command gathers the evidence for a question as above and prints the compacted text,
with the source ID and pages of each section, and how many characters were saved. It needs the
same inputs as the evidence agent and Ollama running:

```bash
python -m chatbot.legalos_rag.contextCompactor --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "What are the conditions for divorce by mutual consent, and what does the Act provide about custody of minor children?"
```

A section used by several parts appears once. Short sections are kept whole; in a longer section only
whole sentences that carry no condition are removed, and each gap is marked `[...]`. The saving is
small: the sections of these Acts are mostly conditions, so almost nothing is removed. See the
[compactor notes](legalos_rag/README.md#contextcompactorpy) for what was measured.

## Answer a question from a verified Act

This command runs the whole new path for one question and one checked Act: it reads the facts,
gathers evidence for each part, checks the conditions, and writes an answer with its sources. It
needs the same inputs as the evidence agent and Ollama running:

```bash
python -m chatbot.legalos_rag.answerAgent --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "What are the conditions for divorce by mutual consent, and what does the Act provide about custody of minor children?"
```

The answer has one block for each question part, with an explanation of the law, a sentence on what
applies to the user's situation, the sources (Act, section and pages), and a note on its limits. A part
with nothing found says what was not found. The explanation is written by the local model from the
cited sections; every statement about the user's own case is a fixed sentence chosen in code from the
condition check, never model text. The answer is only as good as the sections found for it: see the
[answer agent notes](legalos_rag/README.md#answeragentpy) for what was measured.

## Evaluate an answer

A readable answer can still leave out a condition that changes it. This command answers a question as
above and then checks the answer against the sections it was written from. It needs the same inputs
as the answer agent and Ollama running (the evaluation itself uses no model):

```bash
python -m chatbot.legalos_rag.evaluator --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "What does the Act provide about custody of minor children?"
```

It prints the answer and then a verdict for the answer and for each question part: `pass`,
`incomplete` (something is missing or was not found) or `fail` (a citation or unsupported-claim
problem). Each finding names a specific detail and, for an omission, quotes the exact source text that
the answer left out. `missing_details` lists what to look for next. See the
[evaluator notes](legalos_rag/README.md#evaluatorpy) for exactly what is checked and what it cannot see.

## Answer with the orchestrator

This runs the whole new path as one LangGraph graph and decides what to do after each check. It needs
the same inputs as the answer agent and Ollama running:

```bash
python -m chatbot.legalos_rag.orchestrator --config ./config/poc_sources.json --rag-config ./config/rag_v1.json \
  --act hma_1955 --question "What are the conditions for divorce by mutual consent, and what does the Act provide about custody of minor children?"
```

It prints a `Status`, the jobs it ran and the path it took, then the answer. The status is one of:

| Status | Meaning |
| --- | --- |
| `answered` | The answer passed the evaluator. |
| `needs_fact` | A decisive fact is missing; the command prints what to ask the user. |
| `partial` | Some parts have no evidence or the evaluator still finds omissions; the missing details are printed. |
| `no_evidence` | Nothing was found in the Act for any part. |
| `failed_checks` | The evaluator found a citation or unsupported-claim problem; nothing was repaired. |
| `call_limit` | The job limit was reached before the answer passed. |

The settings are in the `orchestrator` object of `config/poc_sources.json`: `max_calls` (jobs per question, 8),
`max_repairs` (1), `allow_extra_search` (false) and `max_searches` (1). Extra search is off because it did not
add correct evidence in the measurements. See the [orchestrator notes](legalos_rag/README.md#orchestratorpy).

---

## Local SLM Setup (Ollama)

We use **qwen2.5:3b-instruct** as the local Small Language Model (SLM).

### Step 1: Install Ollama

**macOS**

```bash
brew install ollama
```

**Windows**  
Download from:  
https://ollama.com/download

---

### Step 2: Start Ollama server

```bash
ollama serve
```

---

### Step 3: Pull the model (run once)

```bash
ollama pull qwen2.5:3b-instruct
```

---

## Run the RAG System

Once everything is set up:

The RAG CLI reads one JSON config file and builds the LangGraph workflow once. Each question then moves through three nodes:

1. **Rewrite** — create legal search variants from the original question.
2. **Retrieve** — fetch and merge relevant chunks for those variants.
3. **Generate** — answer the original question using the retrieved chunks.

The graph is linear for now, so runtime behavior stays the same. Later, a query-classifier node can route direct, scenario-based, and multi-issue questions to different retrieval flows without turning `main.py` into a large set of conditions.

The interactive loop logs each completed answer through `runRag.log_rag_run`. Use **`run_rag(query, db_path, prompt_template, slm)`** directly for a single run without interactive input or logging; it returns **`(result, retrieved_chunks, final_prompt, queries)`**.

**Config (from `legalos/`):**

```text
config/
  └─ rag_v1.json
       ├─ vectordbpath       : \"./vectorDB\"
       ├─ promptTemplate     : { \"text\": \"You are a legal document reader...{format_instructions}...{facts}...{question}\" }
       ├─ model.model_name   : \"qwen2.5:3b-instruct\"
       └─ logging
            ├─ logfile            : \"chatbot/rag_runs.jsonl\"
            ├─ exclude_model_name : false
            └─ exclude_prompt     : true
```

**Run from the project root (`legalos/`):**

```bash
python -m chatbot.main --config ./config/rag_v1.json
```

---

## You're all set! ✅

**Ask legal questions. Get answers grounded strictly in law.**
