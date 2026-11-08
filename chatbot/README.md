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
