from flask import Flask, request, jsonify
import requests
import os
import json
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import HumanMessage

# Load .env file
load_dotenv()

app = Flask(__name__)

# -------------------
# Judge0 API configuration
# -------------------
JUDGE0_URL = "https://ce.judge0.com"
HEADERS = {"Content-Type": "application/json"}

# Supported languages
LANGUAGES = {
    "python": 71,
    "javascript": 63,
    "c": 50,
    "cpp": 54,
    "java": 62
}

# -------------------
# Hugging Face LLM configuration
# -------------------
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

llm = HuggingFaceEndpoint(
    repo_id=model_name,
    task="text-generation"
)

generator = ChatHuggingFace(llm=llm)

# -------------------
# Default question
# -------------------
question = "Given an array of integers, find the length of the longest contiguous subarray where the absolute difference between any two elements is less than or equal to 1."

# -------------------
# Default fallback test cases
# -------------------
default_examples = [
    {"stdin": "5\n1 2 3 4 5", "expected_output": "5"},
    {"stdin": "3\n10 20 30", "expected_output": "3"},
    {"stdin": "4\n7 8 9 10", "expected_output": "4"},
    {"stdin": "6\n1 1 1 1 1 1", "expected_output": "1"}
]

# -------------------
# Hugging Face test case generation
# -------------------
def generate_test_cases(question, num_cases=5):
    """Generate test cases using Hugging Face LLM as proper JSON."""
    prompt = f"""
Generate {num_cases} test cases for the following programming problem.
Each test case should be valid, have correct output, and be designed for the optimal solution.
Use small, medium, and large values to cover edge cases and typical cases.
All numbers should be positive integers unless negative numbers are logically required by the problem.
Format the test cases strictly as a JSON array of objects:
[{{"stdin": "<input>", "expected_output": "<output>"}}, ...]

Additional strict rules:
1. Do NOT include any extra text, explanations, or newlines outside the JSON array.
2. Ensure that each "stdin" is exactly what would be provided via standard input for Judge0 (first line: n, second line: space-separated numbers if needed).
3. Ensure all "expected_output" values are correct for the optimal solution.
4. Ensure the JSON is syntactically valid (double quotes for strings, no trailing commas, proper brackets).
5. Use numeric values in strings only if required by Judge0 (otherwise just plain text numbers).
6. Cover edge cases: minimum input, maximum input, repeated numbers, sorted ascending/descending, random distribution.

Question: {question}
"""

    response = generator.invoke([HumanMessage(content=prompt)])
    if response is None:
        raise Exception("LLM returned None. Check model access and API key.")

    text = response.content  # Raw text from generator
    print("Generated Test Cases:\n", text, flush=True)

    # Parse JSON
    try:
        test_cases = json.loads(text)
        if not isinstance(test_cases, list):
            raise ValueError("LLM did not return a list of test cases")
    except Exception as e:
        print(f"Failed to parse LLM JSON: {e}. Using default examples.", flush=True)
        test_cases = []

    # Fill missing test cases from defaults
    if len(test_cases) < num_cases:
        test_cases.extend(default_examples[:num_cases - len(test_cases)])

    return test_cases[:num_cases], text

# -------------------
# Judge0 submission
# -------------------
def submit_code(source_code, language_id, stdin=""):
    data = {
        "source_code": source_code,
        "language_id": language_id,
        "stdin": stdin,
        "cpu_time_limit": 2,
        "memory_limit": 128000
    }
    response = requests.post(
        f"{JUDGE0_URL}/submissions?base64_encoded=false&wait=true",
        headers=HEADERS,
        json=data
    )
    if response.status_code in [200, 201]:
        return response.json()
    else:
        raise Exception(f"Submission failed: {response.text}")

# -------------------
# Flask route
# -------------------
# -------------------
# Flask route
# -------------------
@app.route('/analyze', methods=['POST'])
def analyze_code():
    response_payload = {"results": [], "big_o_estimate": None, "passed_test_cases": 0}
    data = request.json
    source_code = data.get('source_code')
    lang = data.get('language', 'python').lower()
    language_id = LANGUAGES.get(lang)

    if not language_id:
        return jsonify({"error": "Unsupported language"}), 400

    big_o_enabled = data.get('estimate_big_o', False)

    # Generate test cases
    test_cases = data.get('test_cases')
    if not test_cases:
        try:
            test_cases, _ = generate_test_cases(question, num_cases=5)
        except Exception as e:
            test_cases = []

    results = []
    passed_count = 0

    for case in test_cases:
        stdin = case.get("stdin", "")
        expected_output = case.get("expected_output")
        try:
            res = submit_code(source_code, language_id, stdin=stdin)
        except Exception as e:
            results.append({
                "stdin": stdin,
                "stdout": None,
                "stderr": None,
                "compile_output": None,
                "time": None,
                "memory": None,
                "status": "Submission failed",
                "errors": [{"type": "submission", "message": str(e)}]
            })
            continue

        result_data = {
            "stdin": stdin,
            "stdout": res.get("stdout"),
            "stderr": res.get("stderr"),
            "compile_output": res.get("compile_output"),
            "time": res.get("time"),
            "memory": res.get("memory"),
            "status": res.get("status", {}).get("description"),
            "errors": []
        }

        if res.get("status", {}).get("id") == 6:
            result_data["errors"].append({"type": "compilation", "message": res.get("compile_output")})
        elif res.get("stderr"):
            result_data["errors"].append({"type": "runtime", "message": res.get("stderr")})

        # Check output correctness
        is_correct = False
        if expected_output is not None and res.get("stdout") is not None:
            if res.get("stdout").strip() == expected_output.strip():
                is_correct = True
                passed_count += 1
            else:
                result_data["errors"].append({
                    "type": "wrong_output",
                    "message": f"Expected '{expected_output.strip()}', got '{res.get('stdout').strip()}'"
                })

        results.append(result_data)

    response_payload["results"] = results
    response_payload["passed_test_cases"] = passed_count

    # -------------------
    # Ask LLM for Big O estimation
    # -------------------
    if big_o_enabled:
        try:
            test_cases_str = json.dumps(test_cases, indent=2)
            big_o_prompt = f"""
Given the following source code in {lang} and sample test cases, estimate the time complexity (Big O notation) of the code.
only return the estimated complexity (e.g., O(n), O(n^2), O(n log n)) dont explain anything just give the estimated complexity in one word .

Source code:

Test cases:
{test_cases_str}
"""
            big_o_response = generator.invoke([HumanMessage(content=big_o_prompt)])
            # Keep only the first line if LLM returns multiple lines
            response_payload["big_o_estimate"] = big_o_response.content.strip().split("\n")[0]
        except Exception as e:
            response_payload["big_o_estimate"] = f"Big O estimation failed: {str(e)}"

    return jsonify(response_payload)
# Run Flask app
# -------------------
if __name__ == '__main__':
    app.run(debug=True)
