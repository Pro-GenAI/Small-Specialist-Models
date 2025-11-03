from deepeval.benchmarks import MathQA, DROP, MMLU
from deepeval.benchmarks.math_qa.task import MathQATask
from deepeval.benchmarks.drop.task import DROPTask
from deepeval.benchmarks.mmlu.task import MMLUTask

from utils.response_utils import CustomModel


# Define benchmark with specific tasks and number of code generations
mathqa_benchmark = MathQA(
    tasks=[MathQATask.GENERAL, MathQATask.PROBABILITY, MathQATask.GAIN, MathQATask.OTHER],
    n_shots=1,
    n_problems_per_task=50,
)

if __name__ == "__main__":
    mathqa_benchmark.evaluate(model=CustomModel(), batch_size=1)
    print('-'*40)
    print(mathqa_benchmark.overall_score)



drop_tasks = [  # Get DROPTask.NFL_1, NFL_2, ..., DROPTask.NFL_249
    task for task in DROPTask if "NFL" in task.name and int(task.name.split('_')[-1]) < 500
]
drop_benchmark = DROP(
    tasks=drop_tasks,
    n_shots=1,
    n_problems_per_task=50,
)

if __name__ == "__main__":
    drop_benchmark.evaluate(model=CustomModel(), batch_size=3)
    print('-'*40)
    print(drop_benchmark.overall_score)



mmlu_benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_MATHEMATICS, MMLUTask.COLLEGE_COMPUTER_SCIENCE],
    n_shots=1,
    n_problems_per_task=50,
)

if __name__ == "__main__":
    mmlu_benchmark.evaluate(model=CustomModel(), batch_size=3)
    print('-'*40)
    print(mmlu_benchmark.overall_score)
