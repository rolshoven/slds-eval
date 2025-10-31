# MIT License

# Copyright (c) 2025 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
This module contains task configurations and prompt functions for evaluating
LLM models on SLDS. It was extended from the swiss_legal_evals task initially
created by Joel Niklaus.

Authors: Joel Niklaus, Luca Rolshoven
"""

import logging
import os
import re
import statistics
from dataclasses import dataclass
from textwrap import dedent
from typing import Callable, Literal, Optional

import requests
import torch
from lighteval.metrics.imports.bert_scorer import BERTScorer
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import BertScore, JudgeLLM
from lighteval.metrics.normalizations import (remove_braces,
                                              remove_braces_and_strip)
from lighteval.metrics.utils.metric_utils import (SampleLevelMetricGrouping,
                                                  SamplingMethod)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----- PROMPTS ----- #

SLDS_JUDGE_SYSTEM_PROMPT = dedent(
    """
    You are a senior legal expert and quality assurance specialist with over 20 years of experience in Swiss law. You possess native-level proficiency in German, French, and Italian, enabling you to evaluate Swiss Federal Supreme Court headnotes with precision. Your task is to compare the **Official (Gold) Headnote** with a **Model-Generated Headnote** and provide a structured evaluation in five categories. You will carefully analyze each category and provide a short analysis before committing to a score. The categories are:

    1. Accuracy & Faithfulness: How well does the Model-Generated Headnote match the essential legal meaning and intent of the Official Headnote?
    2. Completeness & Relevance: Does the Model-Generated Headnote include all important points that the Official Headnote emphasizes, without adding irrelevant details?
    3. Clarity & Coherence: Is the text well-organized, easy to understand, and coherent in style and structure?
    4. Articles: Do the same legal articles (prefixed “Art.”) appear correctly and completely in the Model-Generated Headnote as in the Official Headnote?
    5. Considerations: Do the same considerations (prefixed “E.” in German or “consid.” in French/Italian) appear correctly and completely in the Model-Generated Headnote as in the Official Headnote?

    For each category, provide a short and concise explanation followed by a score on a scale from 1 to 3:

    1: Fails or is substantially flawed.
    Major omissions or inaccuracies that fundamentally alter the legal meaning.

    2: Largely correct but missing key element(s).
    Generally captures the substance, yet lacks one or more important details or references.

    3: Closely matches the Official Headnote.
    Covers all critical aspects and references with only minor wording variations that do not affect the legal content.

    Your output must follow the exact structure provided below to ensure consistency and ease of parsing.
    """
).strip()

SLDS_JUDGE_USER_PROMPT = dedent(
    """
    Below are two headnotes for the same leading decision from the Swiss Federal Supreme Court. Please compare the Model-Generated Headnote to the Official (Gold) Headnote according to the following five categories: Accuracy & Faithfulness, Completeness & Relevance, Clarity & Coherence, Articles, and Considerations.

    1. Analyze the Model-Generated Headnote in comparison to the Official Headnote for each category.
    2. Provide a short explanation for your evaluation in each category.
    3. Conclude each category with a score in the exact format: CATEGORYNAME_SCORE: [X], where X is an integer from 1 to 3.

    Required Output Format:

    ACCURACY_FAITHFULNESS:
    Analysis: [Your concise analysis here]
    ACCURACY_FAITHFULNESS_SCORE: [X]

    COMPLETENESS_RELEVANCE:
    Analysis: [Your concise analysis here]
    COMPLETENESS_RELEVANCE_SCORE: [X]

    CLARITY_COHERENCE:
    Analysis: [Your concise analysis here]
    CLARITY_COHERENCE_SCORE: [X]

    ARTICLES:
    Analysis: [Your concise analysis here]
    ARTICLES_SCORE: [X]

    CONSIDERATIONS:
    Analysis: [Your concise analysis here]
    CONSIDERATIONS_SCORE: [X]

    Please adhere strictly to this structure.

    Here is an example how your response could look like:

    <EXAMPLE>
    {one_shot_example}
    </EXAMPLE>

    Now evaluate the Model-Generated Headnote against the Official (Gold) Headnote on the headnotes below.

    Official (Gold) Headnote:
    {original_headnote}

    Model-Generated Headnote:
    {generated_headnote}
    """
).strip()

# Sample id: 60312; decision id: 148 IV 89
SLDS_JUDGE_ONE_SHOT_EXAMPLE_DE = dedent(
    """
    Official (Gold) Headnote:
    Art. 391 Abs. 2 StPO; die erstmalige Anordnung einer ambulanten Massnahme durch das Berufungsgericht verstösst gegen das Verschlechterungsverbot (reformatio in peius). Verzichtet das erstinstanzliche Gericht auf die Anordnung einer beantragten ambulanten Massnahme und hat die Staatsanwaltschaft in ihrer Anschlussberufung deren Anordnung nicht erneut beantragt, verletzt das Berufungsgericht das Verschlechterungsverbot, wenn es eine ambulante Massnahme anordnet (E. 4.1-4.4).

    Model-Generated Headnote:
    Art. 391 Abs. 2 StPO; Verschlechterungsverbot; ambulante Massnahme nach Art. 63 StGB. Die erstmalige Anordnung einer ambulanten Massnahme im Berufungsverfahren verletzt das Verschlechterungsverbot nach Art. 391 Abs. 2 StPO, wenn erstinstanzlich auf eine solche Massnahme verzichtet wurde und die Staatsanwaltschaft in ihrer Anschlussberufung keinen entsprechenden Antrag gestellt hat. Eine solche Anordnung stellt eine unzulässige reformatio in peius dar, da der Beschuldigte das Risiko einer nachträglichen Anpassung oder Umwandlung der Massnahme nur dann trägt, wenn bereits erstinstanzlich eine therapeutische Massnahme angeordnet wurde (E. 4.4). Die freiwillige Teilnahme an einer Therapie durch den Beschuldigten begründet keine ausreichende Grundlage für die erstmalige Anordnung einer ambulanten Massnahme im Berufungsverfahren (E. 4.3).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnotes captures the legal essence and intent of the official headnote accurately. It correctly explains the legal principle of reformatio in peius and the violation of the prohibition of worsening in the context of ordering outpatient measures.
    ACCURACY_FAITHFULNESS_SCORE: 3

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote inclundes all relevant aspects of the official headnote, such as the prohibition of worsening, the legal context of ordering outpatient measures, and the implications of the prosecution's appeal. However, it also adds additional details regarding the voluntary participation in therapy, which are not explicitly mentioned in the official headnote. These are very specific details that may not be necessary for a concise headnote.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is easy to understand and coherent. The clarity could be improved by not repeating the fact that the outpatient measure violates the prohibition of worsening twice. Other than that, the structure and style are clear and well-organized.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: While both the official and model-generated headnotes correctly reference Art. 391 Abs. 2 StPO, the model-generated headnote also includes Art. 63 StGB, which is not explicitly mentioned in the official headnote. This additional reference is relevant to the context of outpatient measures but may not be necessary for a concise headnote.
    ARTICLES_SCORE: 2

    CONSIDERATIONS:
    Analysis: The model-generated headnote only cites considerations E. 4.3 and E. 4.4, whereas the official headnote includes considerations E. 4.1 to E. 4.4. This omission reduces the completeness of the model-generated headnote.
    CONSIDERATIONS_SCORE: 2
    """
).strip()

# Sample id: 60247; decision id: 148 IV 292
SLDS_JUDGE_ONE_SHOT_EXAMPLE_FR = dedent(
    """
    Official (Gold) Headnote:
    Travail externe (art. 77a CP). Le condamné à une peine privative de liberté qui a passé une longue période en détention provisoire peut exécuter sa peine (restante) directement sous la forme de travail externe s'il réalise les conditions posées à l'art. 77a al. 1 CP. Il n'a pas nécessairement besoin d'être privé de sa liberté au moment où est prise la décision ordonnant le travail externe (consid. 2.5.2).

    Model-Generated Headnote:
    Art. 77a CP; art. 164 ss RSPC; conditions pour l'octroi du travail externe. Le travail externe, en tant que phase de l'élargissement progressif de l'exécution de la peine, ne nécessite pas impérativement un séjour préalable en milieu ouvert ou dans une section ouverte d'un établissement fermé. L'utilisation du terme "en principe" à l'art. 77a al. 2 CP indique que cette condition n'est pas absolue et peut être adaptée aux circonstances individuelles, notamment lorsque le condamné a déjà purgé une partie significative de sa peine en détention provisoire ou pour des motifs de sûreté. La cour cantonale a violé le droit fédéral en exigeant de manière rigide un séjour en milieu ouvert comme condition préalable à l'octroi du travail externe. Le recours est admis, l'arrêt cantonal est annulé, et la cause est renvoyée pour une nouvelle décision sur la base des conditions prévues à l'art. 77a CP (consid. 2.5.1 à 2.5.3).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnote accurately reflects the legal principle and conditions for granting external work under Art. 77a CP.
    ACCURACY_FAITHFULNESS_SCORE: 3

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote includes all relevant aspects of the official headnote. However, it adds additional details regarding the use of the term "en principe" and the violation of federal law by the cantonal court. While these details provide context, they are not essential for a concise headnote that shapes future legislation.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is clear and well-organized, but the inclusion of specific details may obscure the broader legal principle.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: The model-generated headnote includes extra legal articles (Art. 164 ff. RSPC) not cited in the official headnote, deviating from its intended focus. Besides this, the reference to Art. 77a CP aligns with the official headnote.
    ARTICLES_SCORE: 2

    CONSIDERATIONS:
    Analysis: The official headnote only cites consideration E 2.5.2, while the model-generated headnote references considerations E 2.5.1 to E 2.5.3. This makes the model-generated headnote less precise.
    CONSIDERATIONS_SCORE: 2
    """
).strip()

# Sample id: 59894; decision id: 148 I 97
SLDS_JUDGE_ONE_SHOT_EXAMPLE_IT = dedent(
    """
    Official (Gold) Headnote:
    Art. 24 Cost.; libertà di domicilio; residenza secondaria per motivi professionali; autorizzazione di soggiorno.  Contenuto e portata della libertà di domicilio garantita dall'art. 24 Cost. (consid. 3.2.1). In tale ambito le competenze dei Cantoni e dei Comuni sono circoscritte al controllo e alla registrazione degli abitanti (consid. 3.3.1). L'obbligo di annunciarsi, in caso di soggiorno in un Comune differente da quello del domicilio per motivi professionali, non limita l'esercizio della libertà di domicilio. Detto annuncio non può però essere sottoposto ad un regime autorizzativo (consid. 3.3.3).

    Model-Generated Headnote:
    Art. 24 Cost.; libertà di domicilio; regime autorizzativo per il soggiorno; competenze comunali. La libertà di domicilio garantita dall'art. 24 Cost. consente a ogni cittadino svizzero di eleggere liberamente il proprio luogo di residenza o soggiorno sul territorio nazionale, senza che ciò possa essere sottoposto a un regime autorizzativo di polizia. Le autorità comunali possono richiedere unicamente una notifica al momento dell'arrivo nel Comune, finalizzata alla registrazione e al controllo degli abitanti, ma non possono imporre un'autorizzazione di soggiorno con scadenza annuale, poiché ciò costituirebbe una violazione della libertà di domicilio (consid. 3.3.3). Il sistema instaurato dalle Direttive MOVPOP, che prevede il rilascio di un'"autorizzazione di soggiorno" con validità limitata, deve essere interpretato nel senso che l'autorità comunale può solo certificare formalmente la notifica del soggiorno, senza sottoporre quest'ultimo a un regime autorizzativo (consid. 3.3.2 e 3.3.3). La conferma di un tale regime da parte del Tribunale cantonale amministrativo viola pertanto l'art. 24 Cost. e deve essere annullata (consid. 3.4).

    ACCURACY_FAITHFULNESS:
    Analysis: The model-generated headnote aligns with the core legal meaning but includes additional details (e.g., MOVPOP directives) not in the official headnote. These do not conflict but shift the focus slightly.
    ACCURACY_FAITHFULNESS_SCORE: 2

    COMPLETENESS_RELEVANCE:
    Analysis: The model-generated headnote captures key points but omits emphasis on secondary residence for professional reasons and cantonal/communal roles. Irrelevant details (e.g., MOVPOP) add complexity.
    COMPLETENESS_RELEVANCE_SCORE: 2

    CLARITY_COHERENCE:
    Analysis: The model-generated headnote is clear and organized, but additional elements like MOVPOP reduce coherence by shifting focus away from the main points and making the text longer and more complex.
    CLARITY_COHERENCE_SCORE: 2

    ARTICLES:
    Analysis: References to Art. 24 Cost. are correct and complete.
    ARTICLES_SCORE: 3

    CONSIDERATIONS:
    Analysis: The model-generated headnote correctly references consid. 3.3.3 but adds consid. 3.3.2 and 3.4, which are beyond the official headnote's scope. Moreover, it leaves out consid 3.2.1 and 3.3.1, reducing precision. Instead, it mentiones consid. 3.3.3 twice, which is redundant.
    CONSIDERATIONS_SCORE: 1
    """
).strip()


# ----- CUSTOM METRICS ----- #

class BertScoreMultilingual(BertScore):
    def __init__(
        self, normalize_gold=None, normalize_pred=None, language=str, model_type=str, num_layers=int, device=str
    ):
        super().__init__(normalize_gold, normalize_pred)
        self.language = language
        self.model_type = model_type
        self.num_layers = num_layers
        self.device = device

    def compute(self, model_response: ModelResponse, doc: Doc, **kwargs) -> dict[str, float]:
        # Make sure we load the correct bert_scorer before the parent class does
        if self.bert_scorer is None:
            self._init_bert_scorer()

        result = super().compute(model_response=model_response, doc=doc, **kwargs)

        # Multiply output by 100 for consistency
        return {k: v * 100 for k, v in result.items()}

    def _init_bert_scorer(self):
        language = self.language
        if language == "rm":
            language = "it"
            logger.warning("There is no BERTScore baseline file for Rumantsch, using Italian instead.")

        if self.device == "mps":
            raise ValueError("MPS is not supported for BERTScore")
        logger.info(
            f"Loading BERTScore with lang={language}, num_layers={self.num_layers}, model_type={self.model_type}, and device={device}..."
        )

        self.bert_scorer = BERTScorer(
            model_type=self.model_type,
            lang=language,  # Needs to be set if rescale_with_baseline is True
            num_layers=self.num_layers,  # Needs to be set if rescale_with_baseline is True
            rescale_with_baseline=True,
            baseline_path=None,
            device=self.device,
        )

        # Create directory structure if it doesn't exist
        os.makedirs(os.path.dirname(self.bert_scorer.baseline_path), exist_ok=True)

        # Download the baseline file if it doesn't exist
        if not os.path.exists(self.bert_scorer.baseline_path):
            raw_url = f"https://raw.githubusercontent.com/Tiiiger/bert_score/master/bert_score/rescale_baseline/{language}/{self.model_type}.tsv"
            logger.info(f"Downloading BERTScore baseline file from {raw_url}")
            response = requests.get(raw_url)
            if response.status_code == 200:
                with open(self.bert_scorer.baseline_path, "wb") as f:
                    f.write(response.content)
            else:
                raise RuntimeError(f"Failed to download baseline file from {raw_url}")



class JudgeSwissLandmarkDecisionSummarization(JudgeLLM):
    SCORE_EXTRACTION_PATTERN = r"^\s*([A-Z_]+_SCORE):\s*(\d+)\s*$"
    RUBRIC_NAMES = (
        "ACCURACY_FAITHFULNESS_SCORE",
        "COMPLETENESS_RELEVANCE_SCORE",
        "CLARITY_COHERENCE_SCORE",
        "ARTICLES_SCORE",
        "CONSIDERATIONS_SCORE",
    )

    def __init__(
        self,
        language: Literal["de", "fr", "it"],
        **kwargs,
    ):
        self.language = language

        super().__init__(template=self._template, process_judge_response=self._process_judge_response, **kwargs)

    def _template(
        self,
        question: str,
        answer: str,
        options: Optional[list[str]] = None,
        gold: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Template for evaluating the Swiss Landmark Decision Summarization task based only on the original and the generated headnotes."""

        # Remove landmark and trailing whitespaces
        system_prompt = SLDS_JUDGE_SYSTEM_PROMPT.strip()
        user_prompt = SLDS_JUDGE_USER_PROMPT.strip()

        if self.language == "de":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_DE.strip()
        elif self.language == "fr":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_FR.strip()
        elif self.language == "it":
            one_shot_example = SLDS_JUDGE_ONE_SHOT_EXAMPLE_IT.strip()

        # Fill template with original and generated headnote
        user_prompt = user_prompt.format(
            original_headnote=gold,
            generated_headnote=answer,
            one_shot_example=one_shot_example,
        )

        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def _process_judge_response(self, response: str) -> float:
        """Process the judge responses and extract the scores for each category."""
        sample_scores = re.findall(pattern=self.SCORE_EXTRACTION_PATTERN, string=response, flags=re.MULTILINE)

        if len(sample_scores) != 5:
            logger.warning("Could only extract %d out of 5 scores from the response: %s", len(sample_scores), response)

        aggregated_score = 0
        for metric_name, score in sample_scores:
            if metric_name not in self.RUBRIC_NAMES:
                logger.warning("Invalid metric name: %s", metric_name)
                continue

            # Transform scale from 1-3 to 0-2
            aggregated_score += int(score) - 1

        # Divide be the maximum possible score
        aggregated_score /= len(self.RUBRIC_NAMES) * 2

        return aggregated_score

    def compute(
        self,
        responses: list[ModelResponse],
        docs: list[Doc],
        **kwargs,
    ) -> list[dict]:
        logger.info(f"Judging {len(docs)} samples with {self.short_judge_name}...")

        not_considered = [None for _ in docs]
        original_headnotes = [doc.get_golds()[0] for doc in docs]
        generated_headnotes = [response.text[0] for response in responses]

        # Exclude the messages (user prompt) because they are too long
        scores, _, judgements = self.judge.evaluate_answer_batch(
            questions=not_considered, answers=generated_headnotes, options=not_considered, golds=original_headnotes
        )
        return [
            {
                self.short_judge_name: score * 100,
                # TODO: find out how we can include the judgment text again without generating errors during aggregation
                # f"{self.short_judge_name}_judgment": judgment,
            }
            for score, judgment in zip(scores, judgements)
        ]


def get_bert_score(
    language: str,
    num_layers: int = 24,
    model_type: str = "xlm-roberta-large",
    device: str = "cpu",
    metric_category: SamplingMethod = SamplingMethod.GENERATIVE,
):
    return SampleLevelMetricGrouping(
        metric_name=["BERTScore-P", "BERTScore-R", "BERTScore-F"],
        higher_is_better={
            "BERTScore-P": True,
            "BERTScore-R": True,
            "BERTScore-F": True,
        },
        category=metric_category,
        sample_level_fn=BertScoreMultilingual(
            normalize_gold=remove_braces,
            normalize_pred=remove_braces_and_strip,
            language=language,
            model_type=model_type,
            num_layers=num_layers,
            device=device,
        ),
        corpus_level_fn={
            "BERTScore-P": statistics.mean,
            "BERTScore-R": statistics.mean,
            "BERTScore-F": statistics.mean,
        },
        batched_compute=False,
    )


def get_swiss_landmark_decision_summarization_judge(
    language: Literal["de", "fr", "it"],
    model_name: str = "openrouter/deepseek/deepseek-chat",
    short_judge_name: str = "slds_judge_deepseek_v3",
    backend: str = "litellm",
):
    judge = JudgeSwissLandmarkDecisionSummarization(
        judge_model_name=model_name,
        judge_backend=backend,
        short_judge_name=short_judge_name,
        language=language,
    )

    judge.judge.API_MAX_RETRY = 60

    return SampleLevelMetricGrouping(
        metric_name=[short_judge_name],
        higher_is_better={short_judge_name: True},
        category=SamplingMethod.GENERATIVE,
        sample_level_fn=judge,
        corpus_level_fn={short_judge_name: statistics.mean},
        batched_compute=True,
    )


def get_extractiveness(language: Literal["de", "fr", "it"]) -> SampleLevelMetricGrouping:
    if language == "de":
        return Metrics.extractiveness_de
    if language == "fr":
        return Metrics.extractiveness_fr
    if language == "it":
        return Metrics.extractiveness_it

    raise ValueError(f"Unsupported language for extractiveness metric: {language}")


# ----- DATASET CONFIGS AND HELPER FUNCTIONS ----- #

@dataclass
class LevelConfig:
    name: str
    text_col_name: str
    generation_size: int
    stop_sequence: list[str]  # just "\n" leads to problems for anthropic models, maybe we need a special case there
    metadata_cols: Optional[list[str]] = None
    custom_attributes: Optional[dict] = None
    dataset_filter: Optional[Callable[[dict], bool]] = None


@dataclass
class DatasetConfig:
    name: str
    hf_repo: str
    languages: list[str]
    task_type: Literal["translation", "summarization"]
    subsets: dict[str, LevelConfig]


# Headnote generation (summarization) for Swiss Landmark Decisions
slds_languages = ["de", "fr", "it"]


def get_slds_filter_fn(decision_language: str, headnote_language: str):
    def filter_dataset(example):
        return example["decision_language"] == decision_language and example["headnote_language"] == headnote_language

    return filter_dataset


SwissLandmarkDecisionHeadnotes = DatasetConfig(
    name="slds",
    hf_repo="ipst/slds",
    languages=slds_languages,
    task_type="summarization",
    subsets={
        **{
            f"{decision_lang}_{headnote_lang}": LevelConfig(
                name=f"{decision_lang}_{headnote_lang}",
                custom_attributes={
                    "decision_language": decision_lang,
                    "headnote_language": headnote_lang,
                },
                text_col_name="decision",
                generation_size=512,
                dataset_filter=get_slds_filter_fn(decision_lang, headnote_lang),
                stop_sequence=["</s>"],
            )
            for decision_lang in slds_languages
            for headnote_lang in slds_languages
        }
    },
)


def iso2lang(iso_code: str) -> str:
    """
    Convert an ISO 639-1 code to a language name.
    """
    assert iso_code in ["de", "fr", "it"], f"Invalid ISO code for SLDS dataset: {iso_code}"
    if iso_code == "de":
        return "German"
    if iso_code == "fr":
        return "French"
    if iso_code == "it":
        return "Italian"
    return None


def slds_prompt_fn(line: dict, task_name: str = None):
    """
    Create a prompt for the Swiss Legal Decision Summaries dataset.
    """
    template = (
        "Leading decision:\n```{decision}```\n\nGenerate a headnote in {language} for the leading decision above."
    )

    return Doc(
        task_name=task_name,
        query=template.format(language=iso2lang(line["headnote_language"]), decision=line["decision"]),
        choices=[str(line["headnote"])],
        gold_index=0,
        specific={
            "sample_id": line["sample_id"],
            "decision_id": line["decision_id"],
            "decision_language": line["decision_language"],
            "headnote_language": line["headnote_language"],
            "law_area": line["law_area"],
            "year": line["year"],
            "text": line["decision"],  # Needs to be called "text" for the extractiveness metric
            "headnote": line["headnote"],
        },
    )


# ----- LIGHTEVAL TASKS ----- #

class HeadnoteGenerationTask(LightevalTaskConfig):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        level_name: str,
    ):
        level_config = dataset_config.subsets[level_name]
        headnote_language = dataset_config.subsets[level_name].custom_attributes["headnote_language"]

        super().__init__(
            name=f"{dataset_config.name}:{level_name}",
            suite=["community"],
            prompt_function=slds_prompt_fn,
            hf_repo="ipst/slds",
            hf_subset=level_name,
            hf_filter=level_config.dataset_filter,
            hf_avail_splits=["train", "validation", "test", "one_shot_examples"],
            evaluation_splits=["test"],
            few_shots_split="one_shot_examples",
            few_shots_select="random",
            generation_size=level_config.generation_size,
            metrics=self._get_metrics(headnote_language),
            stop_sequence=level_config.stop_sequence,
        )

    def _get_metrics(self, headnote_language: Literal["de", "fr", "it"]) -> list[Metrics]:
        return [
            get_bert_score(
                language=headnote_language,
                model_type="xlm-roberta-large",
                device=device,
                metric_category=SamplingMethod.GENERATIVE,
            ),
            Metrics.bleu,
            Metrics.rouge1,
            Metrics.rouge2,
            Metrics.rougeL,
            get_swiss_landmark_decision_summarization_judge(
                language=headnote_language,
            ),
            get_extractiveness(language=headnote_language),
        ]


# ----- DATASETS AND TASKS TO EXPORT ----- #

DATASETS = [SwissLandmarkDecisionHeadnotes]

TASKS_TABLE = [
    HeadnoteGenerationTask(
        dataset_config=SwissLandmarkDecisionHeadnotes,
        level_name=subset,
    )
    for subset in SwissLandmarkDecisionHeadnotes.subsets
]