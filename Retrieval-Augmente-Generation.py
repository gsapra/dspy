"""
Retrieval-augmented generation (RAG) is an approach that allows LLMs to tap into a large corpus of knowledge from sources
and query its knowledge store to find relevant passages/content and produce a well-refined response.

RAG ensures LLMs can dynamically utilize real-time knowledge even if not originally trained on the subject and
give thoughtful answers. However, with this nuance comes greater complexities in setting up refined RAG pipelines.
To reduce these intricacies, we turn to DSPy, which offers a seamless approach to setting up prompting pipelines!

"""
import os
import pickle
import dspy
from dsp import LM
from dspy.datasets import HotPotQA
from openai import OpenAI
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
"""
 language model (LM) - To generate
 retrieval model (RM) - To retrieve - ColBERTv2 retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from this 2017 dump)
"""

class OpenRouter(LM):
    def __init__(self, model_name: str = 'anthropic/claude-3-opus', **kwargs):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key='',
        )
        self.kwargs = kwargs
        self.max_tokens = kwargs.get("max_tokens", 8196)
        self.temperature = kwargs.get("temperature", 0.5)

    def basic_request(self, prompt: str, **kwargs):
        # Perform the API call using the updated pattern
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
            {
                "role": "user",
                "content": prompt,
            },
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # import ipdb; ipdb.set_trace()
        response = completion.choices[0].message.content
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.basic_request(prompt, **kwargs)
        return [response]

openrouter_lm = OpenRouter( temperature=0.2, max_tokens=22937)

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=openrouter_lm, rm=colbertv2_wiki17_abstracts)
#We configure the LM and RM within DSPy, allowing DSPy to internally call the respective module when needed for generation or retrieval.


# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
#HotPotQA dataset, a collection of complex question-answer pairs typically answered in a multi-hop fashion

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]



# # Save trainset to a file
# with open('trainset.pkl', 'wb') as train_file:
#     pickle.dump(trainset, train_file)
#
# # Save devset to a file
# with open('devset.pkl', 'wb') as dev_file:
#     pickle.dump(devset, dev_file)



# # Load trainset from the file
# with open('trainset.pkl', 'rb') as train_file:
#     trainset = pickle.load(train_file)
#
# # Load devset from the file
# with open('devset.pkl', 'rb') as dev_file:
#     devset = pickle.load(dev_file)


"""
When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.
A signature is a declarative specification of input/output behavior of a DSPy module. 
Signatures allow you to tell the LM what it needs to do, rather than specify how we should ask the LM to do it.
"""
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


"""
 RAG pipeline as a DSPy module which will require two methods:

The __init__ method will simply declare the sub-modules it needs: dspy.Retrieve and dspy.ChainOfThought. 
The forward method will describe the control flow of answering the question using the modules we have: 
Given a question, we'll search for the top-3 relevant passages and then feed them as context for answer generation.
"""

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

#example to test the response
# rag=RAG()
#
# print(rag("What castle did David Gregory inherit?").answer)
"""
Compiling depends on three things:

A training set. We'll just use our 20 question–answer examples from trainset above.
A metric for validation. We'll define a simple validate_context_and_answer that checks that the predicted answer is correct and that the retrieved context actually contains the answer.
A specific teleprompter. The DSPy compiler includes a number of teleprompters that can optimize your programs.
"""


# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    print(answer_EM,answer_PM) # BOOLEAN VALUES
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset[:3])

# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
pred = compiled_rag(my_question)

# Print the contexts and the answer.
print(f"Question: {my_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Retrieved Contexts (truncated): {[c[:200] + '...' for c in pred.context]}")


#You can also easily inspect the learned objects themselves.
for name, parameter in compiled_rag.named_predictors():
    print(name) #generate_answer
    print(parameter)
    """ChainOfThought(GenerateAnswer(context, question -> answer
    instructions='Answer questions with short factoid answers.'
    context = Field(annotation=str required=True json_schema_extra={'desc': 'may contain relevant facts', '__dspy_field_type': 'input', 'prefix': 'Context:'})
    question = Field(annotation=str required=True json_schema_extra={'__dspy_field_type': 'input', 'prefix': 'Question:', 'desc': '${question}'})
    answer = Field(annotation=str required=True json_schema_extra={'desc': 'often between 1 and 5 words', '__dspy_field_type': 'output', 'prefix': 'Answer:'})
   )) """
    print(parameter.demos[0]) #Example({'question': 'Which of these publications was most recently published, Who Put the Bomp or Self?', 'answer': 'Self'}) (input_keys={'question'})


#evaluate the accuracy (exact match) of the predicted answer.
# Set up the `evaluate_on_hotpotqa` function. We'll use this many times below.
evaluate_on_hotpotqa = Evaluate(devset=devset, num_threads=1, display_progress=False, display_table=5)

# Evaluate the `compiled_rag` program with the `answer_exact_match` metric.
metric = dspy.evaluate.answer_exact_match
evaluate_on_hotpotqa(compiled_rag, metric=metric) # Average Metric: 54