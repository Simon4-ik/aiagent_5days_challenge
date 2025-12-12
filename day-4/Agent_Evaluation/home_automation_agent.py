import os
import json
import asyncio
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types


# -----------------------------
# LOAD ENV + SETUP RETRY CONFIG
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)


# -----------------------------
# TOOL: DEVICE CONTROL
# -----------------------------
def set_device_status(location: str, device_id: str, status: str) -> dict:
    print(f"Tool Call: Setting {device_id} in {location} to {status}")
    return {
        "success": True,
        "message": f"Successfully set the {device_id} in {location} to {status.lower()}."
    }


# -----------------------------
# AGENT DEFINITION
# -----------------------------
root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="home_automation_agent",
    description="An agent to control smart devices in a home.",
    instruction="""
        You are a home automation assistant. You control ALL smart devices in the house.
        You have access to lights, security systems, ovens, fireplaces, and any device the user mentions.
        Always try to be helpful and control whatever device the user asks for.
    """,
    tools=[set_device_status],
)


# -----------------------------
# CREATE FOLDERS
# -----------------------------
os.makedirs("Agent_Evaluation", exist_ok=True)


# -----------------------------
# TEST CONFIG
# -----------------------------
eval_config = {
    "criteria": {
        "tool_trajectory_avg_score": 1.0,
        "response_match_score": 0.8
    }
}

with open("Agent_Evaluation/test_config.json", "w") as f:
    json.dump(eval_config, f, indent=2)

print("✅ test_config.json created.")


# -----------------------------
# TEST CASES (EVALSET)
# -----------------------------
test_cases = {
    "eval_cases": [
        {
            "eval_id": "basic_device_control",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Turn on the kitchen lights."}
                        ]
                    }
                }
            ],
            "expected_response": "Setting lights in kitchen to on.",
            "expected_tool_calls": [
                {
                    "tool_name": "set_device_status",
                    "args": {
                        "location": "kitchen",
                        "device_id": "lights",
                        "status": "ON"
                    }
                }
            ]
        },
        {
            "eval_id": "wrong_tool_usage_test",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Switch off the bedroom heater."}
                        ]
                    }
                }
            ],
            "expected_response": "Setting heater in bedroom to off.",
            "expected_tool_calls": [
                {
                    "tool_name": "set_device_status",
                    "args": {
                        "location": "bedroom",
                        "device_id": "heater",
                        "status": "OFF"
                    }
                }
            ]
        },
        {
            "eval_id": "poor_response_quality_test",
            "conversation": [
                {
                    "user_content": {
                        "parts": [
                            {"text": "Turn off the living room TV."}
                        ]
                    }
                }
            ],
            "expected_response": "Setting tv in living room to off.",
            "expected_tool_calls": [
                {
                    "tool_name": "set_device_status",
                    "args": {
                        "location": "living room",
                        "device_id": "tv",
                        "status": "OFF"
                    }
                }
            ]
        }
    ]
}

with open("Agent_Evaluation/integration.evalset.json", "w") as f:
    json.dump(test_cases, f, indent=2)

print("✅ integration.evalset.json created.")


# -----------------------------
# OPTIONAL: AUTO-RUN EVALUATION
# -----------------------------
print("\n🚀 To run evaluation manually, use this command:")
print("adk eval home_automation_agent Agent_Evaluation/integration.evalset.json "
      "--config_file_path=Agent_Evaluation/test_config.json --print_detailed_results")


# Uncomment below ONLY if you want Python to run the eval directly:
"""
import subprocess

subprocess.run([
    "adk",
    "eval",
    "home_automation_agent",
    "Agent_Evaluation/integration.evalset.json",
    "--config_file_path=Agent_Evaluation/test_config.json",
    "--print_detailed_results"
], check=True)
"""
