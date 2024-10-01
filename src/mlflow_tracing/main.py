import time
from typing import List
import mlflow
import numpy as np

# Note: tracing only works when using a different experiment than Default
mlflow.set_experiment("My experiment")

# mlflow.tracing.disable()   # Disable tracing globally


def sleep_approx(sleep_time: float): 
    noise = np.min([np.random.normal(0, 0.2), sleep_time])
    time.sleep(sleep_time + noise)
    

@mlflow.trace(span_type="func")
def load_data() -> str:
    texts = [
        "Cats and dogs sometimes disagree",
        "Penguins probably don't care about cats",
        "Penguins must be cold all the time",
        "Dogs smell great, they can smell anything",
        "Cats are clumsy like a penguin",
        "Dogs and penguins cannot fly",
    ]
    sleep_approx(1)
    return np.random.choice(texts)

@mlflow.trace(span_type="func", attributes={"model_type": "small LLM", "model_version": "v4"})
def topic_helper(text: str, topics: List[str]) -> str:
    sleep_approx(1)
    topics = np.random.choice(topics, 2, replace=False)
    return topics

@mlflow.trace(span_type="func")
def extract_topic(text: str) -> str:
    
    mlflow.get_current_active_span().set_attributes(
        {"model_type": "big LLM", "model_version": "v1beta"}
    )
    
    topics = ["cats", "dogs", "pengiuns"]
    topics = topic_helper(text, topics)
    sleep_approx(1)
    topic = np.random.choice(topics)
    return topic


@mlflow.trace
def run_experiment():
    
    print("Running experiment...")
    
    text = load_data()
    topic = extract_topic(text)
    
    mlflow.log_param("text", text)
    mlflow.log_param("topic", topic)
    
    mlflow.log_metric("topic_accuracy", np.random.rand())


if __name__ == "__main__":
    
    with mlflow.start_run():
        run_experiment()
    
    print("Done!")

    