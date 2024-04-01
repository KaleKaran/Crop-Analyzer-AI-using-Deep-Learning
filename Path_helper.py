import pyperclip

def replace_backslash(input_string):
    return input_string.replace("\\", "/")

# Example usage:
input_str = input("Enter a string: ")
output_str = replace_backslash(input_str)
print("Output:", output_str)

# Copy the output to the clipboard
pyperclip.copy(output_str)
print("Output copied to clipboard.")