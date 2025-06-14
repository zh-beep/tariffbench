from inspect_ai import Task, task
from inspect_ai.dataset import example_dataset
from inspect_ai.solver import chain_of_thought, generate, self_critique
from inspect_ai.scorer import model_graded_fact


@task
def theory_of_mind():
    return Task(
        dataset=example_dataset("theory_of_mind"),
        solver=[
            chain_of_thought(),
            generate(),
            self_critique()
        ],
        scorer=model_graded_fact()
    ) 