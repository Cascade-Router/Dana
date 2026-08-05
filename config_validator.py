import os
import json

def validate_config(config_file_path):
    """
    Validates the configuration file.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing validation results.
    """

    # Check if the configuration file exists
    if not os.path.exists(config_file_path):
        return {"error": "Configuration file does not exist"}

    # Load the configuration file
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in configuration file"}

    # Check if required keys are present in the configuration
    required_keys = ["loader"]
    for key in required_keys:
        if key not in config:
            return {"error": f"Missing required key '{key}' in configuration"}

    # Validate the loader configuration
    loader_config = config["loader"]
    if "type" not in loader_config or "params" not in loader_config:
        return {"error": "Invalid loader configuration"}
    if loader_config["type"] not in ["file", "database"]:
        return {"error": "Unsupported loader type"}

    # Check if the loader configuration has valid parameters
    if "params" in loader_config and "host" not in loader_config["params"]:
        return {"error": "Missing host parameter in loader configuration"}
    if "params" in loader_config and "port" not in loader_config["params"]:
        return {"error": "Missing port parameter in loader configuration"}

    # If all checks pass, return a success message
    return {"success": "Configuration is valid"}
