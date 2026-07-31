*This project has been created as part of the 42 curriculum by jnantes.*

# Call Me Maybe

## Description

Call Me Maybe is a Python package that translates natural-language prompts into
structured function calls.

Instead of answering a prompt directly, the program selects the function that
should be called and extracts the arguments required by that function. The final
result is written as JSON with exactly these keys:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2, "b": 3}
}
```

The project uses the provided `llm_sdk.Small_LLM_Model` wrapper and implements
constrained decoding to make the generated output valid and schema-compliant.

## Instructions

Install the project dependencies:

```bash
uv sync
```

Run with the default input and output paths:

```bash
uv run python -m src
```

Run with explicit paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The same commands are available through the Makefile:

```bash
make install
make run
make debug
make lint
make clean
```

## Algorithm Explanation

The program performs function calling in two main steps.

First, it selects the function name. The model receives the user prompt and the
available function definitions. The decoder then restricts generation to the
known function names. At each token step, only tokens that can still complete one
of the allowed function names remain valid.

Second, it extracts the arguments. After the function is selected, the program
builds a partial JSON object:

```text
{"a":
```

or:

```text
{"source_string": "...", "regex":
```

The value is generated exactly at the position where it should appear in the JSON
object. Numbers are generated character by character with only numeric JSON
characters allowed. Strings are generated after an opening quote and stop when
the model would produce a closing quote.

This approach does not rely on the model spontaneously writing correct JSON. The
program controls the valid token space while the model chooses the highest-logit
valid token.

## Design Decisions

The project is split into small modules:

- `src/__main__.py`: command-line entry point and pipeline orchestration.
- `src/json_utils.py`: JSON loading, validation, and output writing.
- `src/models.py`: Pydantic models and schema validation.
- `src/decoding_utils.py`: token-level constrained decoding helpers.
- `src/interpretation_utils.py`: function-call interpretation logic.

The function is selected by the LLM, not by hardcoded keyword rules. Argument
generation is also driven by the LLM, but constrained by the expected parameter
type.

The current implementation supports the parameter types used by the provided
function set: `number` and `string`.

## Performance Analysis

The model is loaded once and reused for every prompt. Token groups used during
argument generation, such as digit tokens and quote tokens, are cached per model
instance to avoid recomputing them for every prompt.

The output writer always serializes Pydantic models with `json.dump`, producing a
valid JSON array. The result model forbids extra top-level keys, and each
function call is validated against the selected function definition before being
written.

Accuracy depends mostly on two tasks: selecting the correct function and
extracting strings such as regex arguments. The constrained JSON-based argument
extraction improves reliability because the model sees the already generated
arguments as part of the context.

## Challenges Faced

The hardest part was argument extraction. A previous approach asked the model to
choose from candidate values extracted from the prompt. This worked for simple
numbers and names, but failed for regex prompts because regex arguments often
need to be inferred from natural language.

The solution was to switch to incremental JSON generation. This lets the model
generate the value at the exact position where the argument belongs while the
decoder enforces the expected type.

Another challenge was avoiding prompt-only JSON generation. The implementation
uses logits and token restrictions directly instead of trusting the model to
format a JSON object on its own.

## Testing Strategy

The main program can be tested with:

```bash
uv run python -m src
```

For targeted manual testing, `interpretation_utils` can process one or more
prompts directly:

```bash
uv run python -m src.interpretation_utils \
  --prompt "Greet shrek" \
  --prompt "Reverse the word AMAZING" \
  --prompt 'Replace all numbers in "Hello 34 I am 233 years old" with NUMBERS'
```

The recommended checks are:

- the output file is valid JSON;
- each result has exactly `prompt`, `name`, and `parameters`;
- the selected function exists in the function definition file;
- all required parameters are present;
- parameter values match their declared types.

## Resources

- Hugging Face documentation: generation strategies.
- Hugging Face documentation: tokenizers and tokenization.
- Pydantic documentation: models and validation.
- Python documentation: `json`, `argparse`, and package execution with `-m`.
- Project subject: constrained decoding and function calling requirements.

AI assistance was used during development to discuss constrained decoding,
compare architectural options, review errors, and draft documentation. The final
implementation choices were tested and reviewed manually during development.
