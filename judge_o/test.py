import requests

# Judge0 API configuration
JUDGE0_URL = "https://ce.judge0.com"
HEADERS = {
    "Content-Type": "application/json"
}

# Python 3 language ID
LANGUAGE_ID = 71

def submit_code(source_code, stdin=""):
    """Submit code to Judge0 and get the result immediately (wait=true)."""
    data = {
        "source_code": source_code,
        "language_id": LANGUAGE_ID,
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

if __name__ == "__main__":
    # Your Python code to test
    source_code = """n = int(input())
arr = list(map(int, input().split()))
print(sum(arr))"""

    # Input to feed into the program
    stdin = "5\n1 2 3 4 5"

    # Submit and get the result
    result = submit_code(source_code, stdin=stdin)

    # Print results
    print("Status:", result.get("status", {}).get("description"))
    print("Stdout:", result.get("stdout"))
    print("Stderr:", result.get("stderr"))
    print("Compile Output:", result.get("compile_output"))
    print("Time:", result.get("time"))
    print("Memory:", result.get("memory"))
