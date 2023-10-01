import hashlib

def calculate_checksum(file_path, hash_algorithm="sha256"):
    """
    Calculate the checksum of a file using the specified hash algorithm.

    Args:
        file_path (str): The path to the file.
        hash_algorithm (str): The hash algorithm to use (e.g., "md5", "sha256").

    Returns:
        str: The checksum of the file.
    """
    try:
        # Create a hash object based on the specified algorithm
        hasher = hashlib.new(hash_algorithm)

        # Open the file in binary mode for reading
        with open(file_path, "rb") as file:
            # Read the file in chunks to avoid loading the entire file into memory
            while True:
                chunk = file.read(4096)  # 4KB chunks
                if not chunk:
                    break
                hasher.update(chunk)

        # Calculate the checksum
        checksum = hasher.hexdigest()
        return checksum
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    file_path = "path/to/your/file.txt"  # Replace with the path to your file
    hash_algorithm = "sha256"  # You can choose a different hash algorithm if needed

    checksum = calculate_checksum(file_path, hash_algorithm)
    if checksum:
        print(f"{hash_algorithm} checksum of '{file_path}': {checksum}")
    else:
        print(f"Error calculating checksum for '{file_path}'")