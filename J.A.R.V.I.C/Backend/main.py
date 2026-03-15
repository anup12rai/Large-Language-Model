import sys
import os
from llm.src.interface.pipeline import classify_query

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

print("Query Type Detector Started")
print("Type 'exit' to stop\n")

while True:

    user_input = input("User: ")

    if user_input.lower() == "exit":
        break

    result = classify_query(user_input)

    print("Bot:", result)

print("Query Type Detector Stopped")