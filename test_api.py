import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_info():
    print("Testing info endpoint...")
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_query(question):
    print(f"Testing query endpoint with question: '{question}'")
    
    payload = {
        "question": question
    }
    
    response = requests.post(f"{BASE_URL}/query", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
    else:
        print(f"Error: {response.text}")
    print()

def main():
    print("Testing RAG API endpoints...")
    print("=" * 50)
    
    test_health()
    
    test_info()
    
    # Test query endpoint with sample questions
    sample_questions = [
        "What are the important deadlines for Medicare enrollment?",
        "What is Medicare?"
    ]
    
    for question in sample_questions:
        test_query(question)
        print("-" * 50)

if __name__ == "__main__":
    main() 