# Decision Index sources for Groundbeef v0.3

Reviewed 2026-10-11. This is a source-selection inventory, not a claim that
every source has passed a row-level ingestion audit. See the
[error analysis](BAZZA_TRANSFER_ERRORS.md) and [training plan](../plans/GROUNDBEEF_V03_PLAN.md).

## Scope and decisions

The snapshot combines the Space's benchmark catalogue, added benchmarks and
reproduction suite: **57 distinct text/catalogue entries**, including the eight
interactive environments and calibration extensions. SimpleBench appears in
the suite rather than the current Space catalogue; PhishNChips appears twice
and is counted once. These are not 57 equally weighted headline benchmarks.
ForecastBench was retired from suite edition 0.3. The separate vision board
has 11 entries, listed below and deferred for our text-only v0.3.

**Include:** suitable upstream training data exists; qualify its records first.
**Optional:** eligible material exists or is plausible, but lower priority or
an unresolved grouping/format gate remains. **Evaluation:** keep the benchmark
out of training; no verified disjoint training release. **Defer:** outside the
current scope. Published test subsets are not a licence to train on their
unused siblings.

## Complete text/catalogue inventory

| ID | Benchmark / primary source | Training availability and v0.3 decision |
|---:|---|---|
| 1 | [BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | **Evaluation.** Main release is an evaluation corpus. Train tool skills from separately qualified corpora below. |
| 2 | [ToolRet](https://github.com/mangopy/tool-retrieval-benchmark) | **Include separate training:** [ToolRet-Training-20w](https://huggingface.co/datasets/mangopy/ToolRet-Training-20w), 208,826 records. ToolRet-Queries is evaluation material. |
| 3 | [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank) | **Include:** separate training corpus; [Hub training-data](https://huggingface.co/datasets/liminghao1630/API-Bank/tree/main/training-data). Preserve dialogue families and time context. |
| 4 | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) | **Include modestly:** native train/test, 77 intent classes. Already represented in v0.2. |
| 5 | [CLINC150+OOS](https://github.com/clinc/oos-eval) | **Include:** native train/val/test plus OOS cases. Cover rejection as well as intent recognition. |
| 6 | [RouterBench](https://github.com/withmartian/routerbench) | **Defer:** routing targets depend on candidate model performance, cost and calibration profiles; not a universal correct-action label. |
| 7 | [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) | **Defer:** interactive CUA environment, needs trajectories/action observations. |
| 9 | [Home appliance simulator](https://github.com/apolinario/decision-index) | **Evaluation:** Index-authored cases. Retain our independent rules under a small quota; keep generators/template families separate. |
| 10 | [SGD/SGD-X](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | **Include modestly:** native train/dev/test; intent/schema diversity. Group dialogues and schema paraphrase variants. |
| 11 | [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | **Include if context fits:** native document splits, three labels. Never choose context using gold evidence spans. |
| 12 | [ANLI](https://github.com/facebookresearch/anli) | **Priority include:** 162,865 training examples across three rounds; preserve dev/test. Train all three semantic labels. |
| 13 | [Boxoban](https://github.com/google-deepmind/boxoban-levels) | **Defer:** interactive puzzle policy; level ownership and trajectories required. |
| 14 | [BabyAI/MiniGrid](https://github.com/Farama-Foundation/Minigrid) | **Defer:** environment observations/action trajectories, useful for a future agent phase. |
| 15 | [RTFM](https://github.com/facebookresearch/RTFM) | **Defer:** interactive instruction-following game. |
| 16 | [ScienceWorld](https://github.com/allenai/ScienceWorld) | **Defer:** interactive scientific task environment. |
| 17 | [Hanabi](https://github.com/google-deepmind/open_spiel) | **Defer:** hidden-state game, requires a policy/trajectory dataset. |
| 18 | [Leduc poker](https://github.com/google-deepmind/open_spiel) | **Defer:** stateful game policy, not static source labels. |
| 19 | [Codenames](https://github.com/clembench/clembench) | **Defer:** interactive game with episode-level evaluation. |
| 20 | [BPoMP](https://zenodo.org/records/7299879) | **Evaluation:** no verified independent training partition for the published poem pairs; split by poem/author before any future use. |
| 21 | [Humicroedit](https://github.com/n-hossain/semeval-2020-task-7-humicroedit) | **Optional:** native train/dev/test, human humour ratings/comparisons; group original headline and edits. |
| 22 | [POP909-CL](https://github.com/AndyWeasley2004/POP909-CL-Dataset) | **Defer:** symbolic chord labels, not raw audio; music-specific preprocessing and song-level ownership. |
| 23 | [cfcolor](https://www.dgp.toronto.edu/~donovan/cfcolor/) | **Defer:** personalized preference with per-user history; not a generic colour ground truth. |
| 24 | [MMLU](https://huggingface.co/datasets/cais/mmlu) | **Include auxiliary_train:** 99,842 items. Deduplicate against full MMLU-Pro and overlapping upstream science corpora. |
| 25 | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | **Evaluation:** Diamond is nested inside other GPQA configurations; changing config does not establish a training split. |
| 26 | [ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) | **Include:** 2,251 native training questions, subject to cross-corpus deduplication. |
| 27 | [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) | **Include:** 1,119 native training questions. Keep official validation/test protected. |
| 28 | [WinoGrande](https://huggingface.co/datasets/allenai/winogrande) | **Include modestly:** XL train 40,398; our labelled external evaluation uses validation. |
| 29 | [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | **Include modestly:** native train 39,905. Protect labelled validation used for public evaluation. |
| 30 | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | **Priority include:** main train 7,473; test 1,319 protected. Main/Socratic variants share underlying problems. |
| 31 | [ChessBench](https://github.com/google-deepmind/searchless_chess) | **Optional later:** separate training positions exist but are large; use tied-best-move labels and position-family ownership. |
| 32 | [MuSR](https://huggingface.co/datasets/TAUR-Lab/MuSR) | **Evaluation:** named splits are three task types, not train/test; released questions are benchmark material. |
| 33 | [SATA-Bench](https://github.com/sata-bench/sata-bench) | **Evaluation:** released select-all benchmark; no verified disjoint train. Requires multiple-correct-answer handling. |
| 34 | [SimpleBench](https://github.com/simple-bench/SimpleBench) | **Evaluation:** ten public benchmark items, not a useful training pool. |
| 36 | [BRIGHT](https://huggingface.co/datasets/xlangai/BRIGHT) | **Evaluation:** no verified upstream train for its reasoning retrieval questions. Use independent retrieval training instead. |
| 37 | [Amazon ESCI](https://github.com/amazon-science/esci-data) | **Optional include:** native train/test query-product labels: Exact, Substitute, Complement, Irrelevant. Own query groups; cap easy negatives. |
| 38 | [ACOS](https://github.com/NUSTM/ACOS) | **Optional:** native review train/dev/test; aspect/category/sentiment decisions. Resolve licence and label-expansion imbalance first. |
| 39 | [FinEntity](https://github.com/yixuantt/FinEntity) | **Evaluation:** the Index uses the unsplit 979-document release; making a new random split would reuse benchmark documents. |
| 40 | [iSarcasmEval](https://github.com/iabufarha/iSarcasmEval) | **Optional:** separate training and test files. Group original, rephrase and paired variants. |
| 41 | [VAST](https://github.com/emilyallaway/zero-shot-stance) | **Optional:** native train/dev/test stance, topic-based generalization; resolve unstated redistribution terms. |
| 42 | [NLI4CT](https://github.com/ai-systems/Task-2-SemEval-2024) | **Optional later:** separate training archive; clinical documents/sections require context and shared-trial grouping checks. |
| 43 | [CRUXEval](https://github.com/facebookresearch/cruxeval) | **Evaluation:** its 800 functions are the benchmark; use separately sourced executable code for training. |
| 44 | [CLadder](https://github.com/causalNLP/cladder) | **Evaluation:** no verified clean official training split. A fresh graph generator would need independent generator-family ownership. |
| 45 | [HLE](https://huggingface.co/datasets/cais/hle) | **Evaluation:** gated test release; do not use its public questions as training. |
| 48 | [ForecastBench](https://github.com/forecastingresearch/forecastbench-datasets) | **Defer:** retired from current suite; event/time ownership and changing resolutions need a separate forecast pipeline. |
| 50 | [Habermas Machine](https://github.com/google-deepmind/habermas_machine) | **Optional:** human candidate preferences; verify train/evaluation cohort ownership in the released comparisons before admitting any rows. |
| 51 | [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa) | **Include:** 4,957 native train questions; calibration-extension test remains protected. |
| 52 | [CommonsenseQA](https://huggingface.co/datasets/tau/commonsense_qa) | **Include modestly:** 9,741 train questions; labelled validation is used by calibration evaluation. |
| 53 | [Support-ticket calibration](https://github.com/scienthoon/jev-ood-calibration) | **Evaluation:** locally authored benchmark cases. Train intent/capability skills from CLINC, Banking and SGD instead. |
| 54 | [Phishing difficulty gradient](https://github.com/SamuelSacco/jev-exploration) | **Evaluation:** authored benchmark scenarios; cybersecurity expansion deferred. |
| 55 | [Email spam classification](https://github.com/bitnovus/jev-spam-eval) | **Evaluation/defer:** no verified disjoint train in benchmark release. A future raw source needs email/campaign-level grouping. |
| 56 | [PhishNChips](https://huggingface.co/datasets/AreLit/PhishNChips) | **Evaluation/defer:** core 2,000 are benchmarked; other named subsets are challenge sets, not an official train split. |
| 57 | [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | **Evaluation:** 70 validation and 12,032 test questions, no train. Protect all test rows, not just our 2,000. |
| 58 | [BBH](https://github.com/suzgunmirac/BIG-Bench-Hard) | **Evaluation:** fixed benchmark with a canary; never train on these files. |
| 59 | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) | **Priority include:** response records carry train/test ownership and human hallucination spans; derive response labels from spans. |
| 61 | [HoVer](https://github.com/hover-nlp/hover) | **Include if context fits:** native train vs dev. Index uses oracle documents; retain that framing and avoid answer-conditioned extraction. |
| 62 | [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | **Priority include:** 15K SFT + 9K preferences; test 3,652 protected. Actual train payloads need conversation/pair conversion. |
| 64 | [New Yorker caption matching](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest) | **Optional:** text-only matching has train/validation/test. Own contest/cartoon IDs and cross-fold variants together. |

## Complementary training sources

| Source | Why use it | Qualification requirement |
|---|---|---|
| [xLAM function calling 60K](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Tool schema + query + full native call targets; synthetic generation with verification | Access acknowledgement; audit labels and join parent IDs with When2Call/ToolRet. |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Alternative single/multiple function-call conversations | Preserve conversation/registry ownership; remove overlap with ToolRet and existing training. |
| [AQuA](https://github.com/google-deepmind/AQuA) | 97,467 native training MCQs, maths with five options | Validate numeric/letter targets; explanations excluded from model input. |
| [CodeIO raw PythonEdu](https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning-Raw) | Code, function arguments and executed outputs | Parse JSONL directly: heterogeneous outputs break the Arrow viewer. Some records lack I/O or adequate descriptions; code + input is the task input. |
| [SWE-QA](https://huggingface.co/datasets/lailaelkoussy/swe-qa) | Preserve qualified software-engineering decisions from v0.2 | Oracle/noisy-oracle variants share question families; keep our existing protected groups. |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Human ordinal quality ratings | Keep rating dimensions separate; preserve our previous heldouts and prompt families. |

No GPT-4 Prometheus rubric data, PrimeVul, or vision/audio rows are proposed
for the first v0.3 view. Human ratings do not by themselves supply every
arbitrary rubric-following task; keep that coverage gap explicit.

## Format and leakage findings that affect ingestion

1. **BFCL is not an SFT training split.** Do not convert the official benchmark
   into training just because its tool schemas and answers are public.
2. **When2Call's published train files are SFT and preference records**, not
   test-style four-choice rows. Avoid inferring policy classes from English
   refusal/clarification regexes. Prefer native preference pairs or audited
   schema-derived counterfactuals. Its current raw-to-SFT script has a
   suspicious cannot-answer branch selecting `answers.tool_call`; sampled
   shipped SFT rows look correct. This is a converter audit item, not evidence
   that the released corpus is corrupted.
3. **ToolRet can have multiple positives.** Use judged binary relevance or
   uniquely correct menus. Do not arbitrarily label one positive correct and
   another negative; unjudged candidates are not verified negatives.
4. **API-Bank training is separate from the small evaluation API catalogue.**
   A sampled training record has instruction/input/output, embedded API
   descriptions, dialogue and a time reference. Keep the full required schema
   and only the conversation prefix available before the target call.
5. **Shared lineage matters.** When2Call train derives from APIGen/xLAM;
   ToolRet aggregates many function corpora. MMLU auxiliary material can share
   questions with science corpora and revised MMLU-Pro. Group across sources.
6. A split name can describe a **task or challenge**, rather than ownership:
   MuSR and PhishNChips are examples. Verify the source before assigning train.
7. RAGTruth annotations and source contexts have different terms. Keep the
   source-level licence ledger; rebuild non-redistributable contexts locally.
   ANLI has a non-commercial licence, some sources have share-alike terms,
   and ACOS/VAST terms need resolution. A local research view and a public
   redistributable edition may have different membership.

## Vision board: explicitly deferred

| Entry | Future task family |
|---|---|
| CV-Bench | Spatial relations/counting |
| BLINK | Visual perception |
| RealWorldQA | Scene understanding |
| CharXiv | Scientific chart reading |
| InfographicVQA | Document/image question answering |
| Mind2Web | Browser action selection |
| Winoground | Text-image composition |
| KIE (CORD+FUNSD) | Document information extraction |
| Moderation (Hateful Memes) | Multimodal moderation |
| R-Bench-M | Visual relationship reasoning |
| MMMU-Pro vision | Multimodal knowledge/reasoning |

Only the board inventory/framing was reviewed here; upstream training-split
eligibility for these eleven is not yet audited. Native Gemma vision support
and a dedicated multimodal dataset plan precede their inclusion.

## Reproducibility

- [Decision Index Space](https://huggingface.co/spaces/multimodalart/jev-decision-index),
  revision `f4ecd9f76093e73560d25fc5a9bd316f17326790`.
- [Reproduction suite](https://github.com/apolinario/decision-index), revision
  `9eb2dbe2a358004c8782c66e40a83ac07b953fec`.
- Cached methodology/suite manifest:
  `build/evaluations/protocol-audit/2026-10-11/`.
- Source cards, split metadata, bounded examples, hashes and fetch failures:
  `build/research/groundbeef-v0.3/2026-10-11/download-manifest*.json`.
- Complete official source pins and evaluation framing also live in
  `datasets/evaluation-panel/sources/decision-index/docs/suite.md`.

An Include/Optional entry remains conditional on actual row quality, context
budget, grouping, preserved heldouts and licence/access checks. Failed metadata
requests are recorded rather than treated as proof that training data exists.
