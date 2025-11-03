from deepeval.benchmarks import HumanEval
from deepeval.benchmarks.human_eval.task import HumanEvalTask

from utils.response_utils import CustomModel


# Define benchmark with specific tasks and number of code generations
benchmark = HumanEval(
    tasks=[
        # Statistical tasks
        HumanEvalTask.MEAN_ABSOLUTE_DEVIATION,
        HumanEvalTask.MEDIAN,
        HumanEvalTask.ROUNDED_AVG,
        HumanEvalTask.RESCALE_TO_UNIT,
        HumanEvalTask.HISTOGRAM,

        # Mathematical tasks
        HumanEvalTask.SUM_PRODUCT,
        HumanEvalTask.SUM_SQUARES,

        # Sorting tasks
        HumanEvalTask.SORT_NUMBERS,
        HumanEvalTask.SORT_ARRAY,
        HumanEvalTask.SORT_THIRD,
        HumanEvalTask.SORT_EVEN,
        HumanEvalTask.STRANGE_SORT_LIST,
        HumanEvalTask.SORTED_LIST_SUM,

        # Filtering tasks
        HumanEvalTask.FILTER_INTEGERS,
        HumanEvalTask.FILTER_BY_SUBSTRING,
        HumanEvalTask.FILTER_BY_PREFIX,
        HumanEvalTask.SPECIALFILTER,

        # Database tasks
        HumanEvalTask.ORDER_BY_POINTS,
        HumanEvalTask.UNIQUE,
        HumanEvalTask.UNIQUE_DIGITS,
        HumanEvalTask.REMOVE_DUPLICATES,

        HumanEvalTask.COUNT_DISTINCT_CHARACTERS,
        HumanEvalTask.COUNT_NUMS,
        HumanEvalTask.HOW_MANY_TIMES,
        HumanEvalTask.COUNT_UP_TO,
        HumanEvalTask.COUNT_UPPER,
        HumanEvalTask.ODD_COUNT,
        HumanEvalTask.EVEN_ODD_COUNT,

        HumanEvalTask.INTERSECTION,
        HumanEvalTask.SELECT_WORDS,
        # HumanEvalTask.SPLIT_WORDS,
        # HumanEvalTask.WORDS_IN_SENTENCE,
        # HumanEvalTask.FILE_NAME_CHECK,
    ],
    n = 5,
)

def _extract_code(response: str) -> str:
    """Extract code from a model response. If a fenced code block exists, return its contents; otherwise return stripped response."""
    if "```" in response:
        parts = response.split("```")
        if len(parts) >= 3:
            code = parts[1]
        else:
            code = parts[-1]
        if code.startswith("python"):
            code = code.lstrip("python").strip()
        return code.strip()
    return response.strip()


if __name__ == "__main__":
    benchmark.evaluate(model=CustomModel(apply_function=_extract_code), k=5)
    print('-'*40)
    print(benchmark.overall_score)
