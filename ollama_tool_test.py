from ollama import chat
from ollama import ChatResponse
import requests
import time
import statistics
from tabulate import tabulate
import csv
from typing import Dict, Any, List
import math
import random

def day_of_the_week() -> str:
    """
    Get the current day of the week

    Args:
        No args

    Returns:
        str: The current day of the week (e.g., 'Monday', 'Tuesday', etc.)
    """
    from datetime import datetime
    return datetime.now().strftime('%A')


def weather_tool(location: str) -> str:
    """
    Find the weather for a specific town or city (location)

    Args:
        location (str): The city or town name for the weather forecast

    Returns:
        str: The weather forecast for the specified location

    """
    # Convert angle to radians for math functions
    if location.lower() == 'paris':
        return "Pas Mal"
    else:
        return "Merd!"

def time_tool() -> str:
    """
    Return the current time in HH:MM format in the local timezone

    Args:
        No args

    Returns:
        str: The current time in HH:MM format

    """

    return "Now"

def location_tool() -> str:
    """
    Returns the current location in {longitude, latitude}

    Args:
        No args

    Returns:
        str: Returns the current location in {longitude, latitude} e.g. "-0.2W, 51.5N"

    """
    return "Here"

def distance_tool(location: str) -> int:
    """
    Calculated the distance to the city or town provided

    Args:
        location (str): The city or town name for the distance calculator

    Returns:
        int: The distance in km to the town or city provided

    """
    if location.lower() == 'new york':
        return 12345
    else:
        return 0

def get_available_models() -> list:
    try:
        response = requests.get('http://localhost:11434/api/tags')
        models = response.json()['models']
        return models
    except requests.RequestException as e:
        print(f"Error listing models: {e}")
        return []


def is_suitable_model(model: Dict[str, Any]) -> bool:
    """Filter for suitable models - adjust criteria as needed"""
    return (model["size"] <=4_000_000_000
            and "bert" not in model["details"]["family"]
            # and "qwen2.5:1.5b-instruct-fp16"  in model['name'].lower()
    )

def run_single_test(prompt: str, expected_result: str, model: str) -> tuple[Dict[str, Any], float]:
    start_time = time.time()
    results = {
        'model': model,
        'prompt': prompt,
        'success': False,
        'error': None,
        'raw_response': None
    }

    available_functions = {
        'day_of_the_week': day_of_the_week,
        'weather_tool': weather_tool,
        'time_tool': time_tool,
        'location_tool': location_tool,
        'distance_tool': distance_tool,
    }

    try:
        response: ChatResponse = chat(
            model,
            messages=[{'role': 'user', 'content': prompt}],
            tools=[
                day_of_the_week,
               weather_tool, time_tool,
               location_tool, distance_tool,
            ],
        )

        results['raw_response'] = response.message

        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    function_result = function_to_call(**tool.function.arguments)
                    results['success'] = function_result == expected_result
                    break
    except Exception as e:
        results['error'] = str(e)

    execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    return results, execution_time


def test_model(model: str, num_runs: int = 5) -> Dict[str, Any]:
    from datetime import datetime
    current_day = datetime.now().strftime('%A')

    # Initial test cases
    test_cases = [
        ('What is the day of the week today?', current_day),
        ('What is the weather in the capital of France?', "Pas Mal"),
        ('What is the time?', "Now"),
        ('What is my location?', "Here"),
        ('How far is it to New York?', 12345),
    ]

    # # Last three test cases to choose from
    # extra_cases = [
    #     ('What is the sine of 30 degrees?', 0.5000),  # sin(30°) ≈ 0.5
    #     ('What is the cosine of 60 degrees?', 0.5000),  # cos(60°) ≈ 0.5
    #     ('What is the tangent of 45 degrees?', 1.0000)  # tan(45°) = 1
    # ]
    #
    # # Randomly select one of the last three
    # test_cases.append(random.choice(extra_cases))

    print(f"\nTesting {model}...")

    # Warmup run
    print("  Performing warmup run...", end=' ', flush=True)
    _ = run_single_test(test_cases[0][0], test_cases[0][1], model)
    print("done")

    all_results = []
    execution_times = []

    for test_case in test_cases:
        prompt, expected = test_case
        print(f"\n  Testing prompt: {prompt}")

        for i in range(num_runs):
            print(f"    Run {i + 1}/{num_runs}...", end=' ', flush=True)
            result, execution_time = run_single_test(prompt, expected, model)
            all_results.append(result)
            execution_times.append(execution_time)
            print("done")

    # Calculate scores
    success_count = sum(1 for r in all_results if r['success'])
    total_possible = len(test_cases) * num_runs

    final_result = {
        'model': model,
        'execution_time': statistics.mean(execution_times),
        'execution_time_std': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
        'success_rate': success_count,
        'total_possible': total_possible,
        'error': all_results[-1].get('error', None),
        'raw_response': all_results[-1].get('raw_response', None)
    }

    return final_result


def format_results_table(results: List[Dict[str, Any]]) -> str:
    headers = ['Model', 'Size (GB)', 'Time (ms)', 'σ Time', 'Success Rate']
    table_data = []

    for result in results:
        model_size_gb = result.get('model_size', 0) / 1_000_000_000
        success_rate = f"{result['success_rate']}/{result['total_possible']}"
        row = [
            result['model'],
            f"{model_size_gb:.2f}",
            f"{result.get('execution_time', 0):.0f}",
            f"±{result.get('execution_time_std', 0):.0f}",
            success_rate
        ]
        table_data.append(row)

    return tabulate(table_data, headers=headers, tablefmt='grid')


def write_results_csv(results: List[Dict[str, Any]], filename: str = 'ollama_function_results.csv'):
    headers = ['Model', 'Size_GB', 'Time_ms', 'Time_StdDev', 'Success_Rate', 'Total_Possible']

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for result in results:
            model_size_gb = result.get('model_size', 0) / 1_000_000_000
            row = [
                result['model'],
                f"{model_size_gb:.2f}",
                f"{result.get('execution_time', 0):.0f}",
                f"{result.get('execution_time_std', 0):.0f}",
                result['success_rate'],
                result['total_possible']
            ]
            writer.writerow(row)


def format_final_report(results: List[Dict[str, Any]]) -> str:
    report = []

    report.append("\n" + "=" * 60)
    report.append("FUNCTION SUPPORT TEST RESULTS SUMMARY")
    report.append("=" * 60)
    report.append(format_results_table(results))

    report.append("\nDETAILED STATISTICS")
    report.append("=" * 60)

    total_models = len(results)
    perfect_models = []
    other_models = []

    for result in results:
        if result['success_rate'] == result['total_possible']:
            perfect_models.append(result['model'])
        else:
            success_rate = (result['success_rate'] / result['total_possible']) * 100
            other_models.append((result['model'], success_rate))

    report.append(f"\nTotal models tested: {total_models}")
    report.append(f"Perfect score models: {len(perfect_models)}")
    report.append(f"Other models: {len(other_models)}")

    if perfect_models:
        report.append("\nModels with perfect scores:")
        for model in perfect_models:
            report.append(f"  ✓ {model}")

    if other_models:
        report.append("\nOther models:")
        for model, rate in other_models:
            report.append(f"  ⚠ {model}: {rate:.1f}% success rate")

    return "\n".join(report)


def main():
    print("Starting function support test suite...\n")

    models = get_available_models()
    if not models:
        print("Error: No models found. Please ensure Ollama is running and models are installed.")
        return

    suitable_models = [m for m in models if is_suitable_model(m)]

    print(f"Found {len(suitable_models)} suitable models for testing.")
    print("Running tests (this may take a few minutes)...")
    print("=" * 60)

    results = []
    for model in suitable_models:
        result = test_model(model['name'])
        result['model_size'] = model['size']
        results.append(result)

    print("\n" * 2)
    report = format_final_report(results)
    print(report)

    # Write results to CSV
    write_results_csv(results)
    print(f"\nResults have been saved to ollama_function_results.csv")


if __name__ == "__main__":
    main()