



from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import chain_of_thought, generate
#from inspect_ai.tools import documents
from inspect_ai.scorer import model_graded_fact

#docs = documents.from_files("data/tariff_txt/**/*.txt")

@task
def tariff_qa_openbook():
    dataset = json_dataset("tariff_openbook.jsonl") 
    #dataset = open_dataset("tariff_openbook.jsonl")
    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=model_graded_fact()
    )