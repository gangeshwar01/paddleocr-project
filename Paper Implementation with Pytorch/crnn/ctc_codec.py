class CTCCodec:
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # Add blank character
        self.char_to_int = {char: i for i, char in enumerate(self.alphabet)}
        self.int_to_char = {i: char for i, char in enumerate(self.alphabet)}
        self.blank_idx = len(self.alphabet) - 1

    def encode(self, text):
        """Converts a string of text into a list of integers."""
        return [self.char_to_int[char] for char in text]

    def decode(self, t):
        """Decodes a sequence of integers into a string."""
        char_list = []
        for i in range(len(t)):
            # Ignore blank and duplicates
            if t[i] != self.blank_idx and (i == 0 or t[i] != t[i - 1]):
                char_list.append(self.int_to_char[t[i]])
        return ''.join(char_list)
