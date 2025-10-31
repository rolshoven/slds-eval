import os
from pathlib import Path
from textwrap import dedent

import typer
from dotenv import load_dotenv
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.endpoints.litellm_model import LiteLLMModelConfig
from lighteval.models.model_input import GenerationParameters
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

app = typer.Typer(pretty_exceptions_enable=False)


SLDS_GENERATION_SYSTEM_PROMPT = dedent(
    """
    You are a legal expert specializing in Swiss Federal Supreme Court decisions with extensive knowledge of legal terminology and conventions in German, French, and Italian. Your task is to generate a headnote for a provided leading decision. A headnote is a concise summary that captures the key legal points and significance of the decision. It is not merely a summary of the content but highlights the aspects that make the decision "leading" and important for future legislation.

    When generating the headnote:

    1. Focus on the core legal reasoning and key considerations that establish the decision's significance.
    2. Include any relevant references to legal articles (prefixed with "Art.") and considerations (prefixed with "E." in German or "consid." in French/Italian).
    3. Use precise legal terminology and adhere to the formal and professional style typical of Swiss Federal Supreme Court headnotes.
    4. Ensure clarity and coherence, so the headnote is logically structured and easy to understand in the specified language.
    
    Your response should consist solely of the headnote in the language specified by the user prompt.
    """
)


@app.command()
def main(
        model: str = typer.Option(default="openrouter/openai/gpt-4o", help="Model name or path"),
        one_shot: bool = typer.Option(default=True, help="Use one-shot prompting, true by default, set to false for trained models."),
        decision_language: str = typer.Option(default="de,fr,it", help="Language of the decision to summarize (de, fr, it), can be comma-separated for multiple languages. Default is de,fr,it"),
        headnote_language: str = typer.Option(default="de,fr,it", help="Language of the headnote to generate (de, fr, it), can be comma-separated for multiple languages. Default is de,fr,it"),
        debug: bool = typer.Option(default=False, help="Debug mode, uses only one sample per language pair")
    ):

    known_languages = {"de", "fr", "it"}
    decision_languages = [lang.strip() for lang in decision_language.split(",")]
    headnote_languages = [lang.strip() for lang in headnote_language.split(",")]

    if not all(lang in known_languages for lang in decision_languages):
        raise ValueError(f"Unknown decision language(s) provided. Known languages are: {', '.join(known_languages)}")

    if not all(lang in known_languages for lang in headnote_languages):
        raise ValueError(f"Unknown headnote language(s) provided. Known languages are: {', '.join(known_languages)}")

    evaluation_tracker = EvaluationTracker(
        output_dir="./results/",
    )

    task_file = Path(__file__).parent / "custom_task" / "slds.py"

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        custom_tasks_directory=str(task_file.resolve()),
        load_tasks_multilingual=True,
        max_samples=1 if debug else None,
    )

    generation_params = {
        "seed": 2025,
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
    }

    if any([s in model for s in ["openai", "gpt", "deepseek"]]):
        generation_params.pop("repetition_penalty")
        generation_params.pop("top_k")

    if "anthropic" in model:
        generation_params.pop("repetition_penalty")

    generation_config = GenerationParameters.from_dict(
        {
            "generation": generation_params,
        }
    )

    model_config = LiteLLMModelConfig(
        model_name=model,
        base_url="http://0.0.0.0:8000/v1" if "hosted_vllm" in model else None,  # Change this if using a different vLLM endpoint
        system_prompt=SLDS_GENERATION_SYSTEM_PROMPT.strip(),
        generation_parameters=generation_config,
        api_max_retry=999,
        api_retry_multiplier=1.75,
        api_retry_sleep=1,
    )

    tasks = [f"community|slds:{lang1}_{lang2}|{int(one_shot)}" for lang1 in decision_languages for lang2 in headnote_languages]
    tasks = ",".join(tasks)

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()

    pipeline.show_results()
    results = pipeline.get_results()
    pipeline.save_and_push_results()


if __name__ == "__main__":
    app()
