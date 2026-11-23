def count_words(text):
    """Count the number of words in a text."""
    return len(text.split())


def count_characters(text):
    """Count the number of characters."""
    return len(text)


def extract_words(text, minimum_length=1):
    """Extract words above a minimum length."""
    return [
        word for word in text.split()
        if len(word) >= minimum_length
    ]
