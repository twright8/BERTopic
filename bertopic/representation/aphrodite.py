import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
# NEW: Import Aphrodite components
from aphrodite import LLM, SamplingParams
from typing import Mapping, List, Tuple, Any, Union, Callable, Dict

# Keep BERTopic imports
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters

# --- Keep the default prompts ---
DEFAULT_PROMPT = """
This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title
---
Topic:
Sample texts from this topic:
- Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
- Meat, but especially beef, is the word food in terms of emissions.
- Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

Keywords: meat beef eat eating emissions steak food health processed chicken
Topic name: Environmental impacts of eating meat
---
Topic:
Sample texts from this topic:
- I have ordered the product weeks ago but it still has not arrived!
- The website mentions that it only takes a couple of days to deliver but I still have not received mine.
- I got a message stating that I received the monitor but that is not true!
- It took a month longer to deliver than was advised...

Keywords: deliver weeks product shipping long delivery received arrived arrive week
Topic name: Shipping and delivery issues
---
Topic:
Sample texts from this topic:
[DOCUMENTS]
Keywords: [KEYWORDS]
Topic name:"""

DEFAULT_SYSTEM_PROMPT = "You are an assistant that extracts high-level topics from texts." # Note: Aphrodite's generate might not directly use a system prompt like LlamaCPP's chat completion. It depends on the model and how the prompt is structured. We keep it for potential future use or inclusion in the main prompt.


# --- NEW Aphrodite Representation Class ---
class AphroditeRepresentation(BaseRepresentation):
    """ A aphrodite-engine implementation to use as a representation model.

    Arguments:
        model: Either a string representing the model name/path for Aphrodite
               or a pre-initialized `aphrodite.LLM` object.
        prompt: The prompt to be used in the model. If no prompt is given,
                `self.default_prompt_` is used instead.
                NOTE: Use `"[KEYWORDS]"` and `"[DOCUMENTS]"` in the prompt
                to decide where the keywords and documents need to be
                inserted.
        system_prompt: The system prompt (currently informational, might be
                       integrated into the main prompt if needed by the model).
        model_kwargs: Dictionary of keyword arguments to pass to the
                      `aphrodite.LLM` constructor when `model` is a string.
                      Defaults include basic settings like gpu_memory_utilization.
        sampling_kwargs: Dictionary of keyword arguments to pass to the
                         `aphrodite.SamplingParams` constructor during generation
                         (e.g., `max_tokens`, `temperature`, `top_p`).
        nr_docs: The number of documents to pass to the LLM if a prompt
                 with the `["DOCUMENTS"]` tag is used.
        diversity: The diversity of documents to pass to the LLM.
                   Accepts values between 0 and 1.
        doc_length: The maximum length of each document. If a document is longer,
                    it will be truncated. If None, the entire document is passed.
        tokenizer: The tokenizer used to calculate to split the document into segments
                   used to count the length of a document.

    Usage:

    ```python
    from bertopic import BERTopic
    # from bertopic.representation import AphroditeRepresentation # Assuming you save this class

    # Option 1: Initialize from model name/path
    # Make sure the model exists locally or on Hugging Face Hub
    model_name = "mistralai/Mistral-7B-Instruct-v0.1" # Example model
    representation_model = AphroditeRepresentation(
        model_name,
        model_kwargs={"trust_remote_code": True, "gpu_memory_utilization": 0.9},
        sampling_kwargs={"max_tokens": 100, "temperature": 0.7}
    )

    # Option 2: Initialize with a pre-configured Aphrodite LLM object
    # from aphrodite import LLM
    # llm_instance = LLM(model=model_name, trust_remote_code=True, gpu_memory_utilization=0.9)
    # representation_model = AphroditeRepresentation(
    #     llm_instance,
    #     sampling_kwargs={"max_tokens": 100, "temperature": 0.7}
    # )

    # Create BERTopic model
    topic_model = BERTopic(representation_model=representation_model, verbose=True)
    ```
    """

    def __init__(
        self,
        model: Union[str, LLM],
        prompt: str = None,
        system_prompt: str = None, # Keep for consistency, though direct use might vary
        model_kwargs: Dict[str, Any] = None,
        sampling_kwargs: Dict[str, Any] = None,
        nr_docs: int = 4,
        diversity: float = None,
        doc_length: int = None,
        tokenizer: Union[str, Callable] = None,
    ):
        if isinstance(model, str):
            _model_kwargs = {
                "gpu_memory_utilization": 0.90, # Sensible default
                "trust_remote_code": True,      # Common requirement
                # Add other defaults if needed, e.g., dtype? quantization?
            }
            if model_kwargs is not None:
                _model_kwargs.update(model_kwargs)
            self.model = LLM(model=model, **_model_kwargs)
        elif isinstance(model, LLM):
            self.model = model
            # If model_kwargs were provided with a pre-init model, maybe warn?
            if model_kwargs is not None:
                 print("Warning: `model_kwargs` are ignored when a pre-initialized `aphrodite.LLM` object is passed.")
        else:
            raise ValueError(
                "Make sure that the model that you pass is either a string "
                "referring to a model name/path for Aphrodite, or an "
                "`aphrodite.LLM` object."
            )

        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        # Store system prompt, but note its usage depends on how prompt is structured for `generate`
        self.system_prompt = system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
        self.default_prompt_ = DEFAULT_PROMPT
        self.default_system_prompt_ = DEFAULT_SYSTEM_PROMPT

        # Store sampling parameters
        self.sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else {}
        # Provide default sampling params if not specified
        self.sampling_kwargs.setdefault("max_tokens", 128) # Default max tokens
        self.sampling_kwargs.setdefault("temperature", 0.7) # Default temperature

        # BERTopic specific parameters
        self.nr_docs = nr_docs
        self.diversity = diversity
        self.doc_length = doc_length
        self.tokenizer = tokenizer
        validate_truncate_document_parameters(self.tokenizer, self.doc_length)

        self.prompts_ = [] # To store generated prompts for debugging/inspection

    def extract_topics(
        self,
        topic_model,
        documents: pd.DataFrame,
        c_tf_idf: csr_matrix,
        topics: Mapping[str, List[Tuple[str, float]]],
    ) -> Mapping[str, List[Tuple[str, float]]]:
        """Extract topic representations using Aphrodite engine.

        Arguments:
            topic_model: A BERTopic model
            documents: The documents DataFrame
            c_tf_idf: The c-TF-IDF matrix
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations with a single label per topic.
        """
        # Extract representative documents
        repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(
            c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity
        )

        updated_topics = {}
        topic_ids = list(repr_docs_mappings.keys())

        # Prepare all prompts first (Aphrodite is good at batching)
        prompts_to_generate = []
        for topic in topic_ids:
            docs = repr_docs_mappings[topic]
            # Prepare prompt
            truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            prompt = self._create_prompt(truncated_docs, topic, topics)
            # Note: If the model requires specific chat templating including the system prompt,
            # you might need to adjust prompt creation here using model.llm_engine.tokenizer.apply_chat_template
            # For now, we assume the DEFAULT_PROMPT structure is sufficient or the model is instruction-tuned.
            prompts_to_generate.append(prompt)
            self.prompts_.append(prompt) # Store for inspection

        # Create SamplingParams object from stored kwargs
        # Ensure no unsupported args are passed if Aphrodite's API changes
        # You might want to filter kwargs based on SamplingParams signature
        try:
            sampling_params = SamplingParams(**self.sampling_kwargs)
        except TypeError as e:
            print(f"Error creating SamplingParams. Check `sampling_kwargs`: {self.sampling_kwargs}. Error: {e}")
            # Fallback or re-raise
            sampling_params = SamplingParams(max_tokens=128, temperature=0.7) # Basic fallback


        # Generate labels in batch using Aphrodite
        if prompts_to_generate:
            outputs = self.model.generate(prompts=prompts_to_generate, sampling_params=sampling_params)
        else:
            outputs = []

        # Process outputs
        if len(outputs) != len(topic_ids):
             print(f"Warning: Number of Aphrodite outputs ({len(outputs)}) does not match number of topics ({len(topic_ids)}). Results might be misaligned.")
             # Handle misalignment or error? For now, proceed cautiously.

        for i, topic in enumerate(topic_ids):
            if i < len(outputs):
                output = outputs[i]
                if output.outputs:
                    # Get the first generated sequence's text, remove potential leading/trailing whitespace
                    label = output.outputs[0].text.strip()
                    # Remove potential "Topic name:" prefix if the model includes it
                    if label.lower().startswith("topic name:"):
                         label = label[len("topic name:"):].strip()
                    # Ensure label is not empty after stripping
                    if not label:
                        label = "Could not generate label"
                        print(f"Warning: Empty label generated for topic {topic}. Prompt was:\n{self.prompts_[i]}")
                else:
                    label = "Generation failed"
                    print(f"Warning: Aphrodite generation failed for topic {topic}. Prompt was:\n{self.prompts_[i]}")
            else:
                # Handle cases where output is missing for a topic
                label = "Missing output"
                print(f"Warning: Missing Aphrodite output for topic {topic}.")

            # Format according to BERTopic expectation (label, score) * 10
            updated_topics[topic] = [(label, 1.0)] + [("", 0.0) for _ in range(9)]

        return updated_topics

    # --- Helper methods (_create_prompt, _replace_documents) are identical to LlamaCPP ---
    def _create_prompt(self, docs, topic, topics):
        keywords = list(zip(*topics[topic]))[0]

        # Use the Default Chat Prompt
        # Note: We are not explicitly adding the system_prompt here, assuming the main prompt covers it
        # or the model is instruction-tuned. If needed, prepend self.system_prompt or use chat templates.
        if self.prompt == DEFAULT_PROMPT:
            prompt = self.prompt.replace("[KEYWORDS]", ", ".join(keywords))
            prompt = self._replace_documents(prompt, docs)

        # Use a custom prompt
        else:
            prompt = self.prompt
            if "[KEYWORDS]" in prompt:
                prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
            if "[DOCUMENTS]" in prompt:
                prompt = self._replace_documents(prompt, docs)

        return prompt

    @staticmethod
    def _replace_documents(prompt, docs):
        to_replace = ""
        for doc in docs:
            # Ensure doc is a string and escape potential issues if needed
            to_replace += f"- {str(doc)}\n"
        prompt = prompt.replace("[DOCUMENTS]", to_replace)
        return prompt
