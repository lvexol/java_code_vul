import sys
import re
import argparse
import tensorflow as tf
import numpy as np
import javalang


# Tokenize Java code using javalang
def tokenize_java_code(code):
    try:
        tokens = list(javalang.tokenizer.tokenize(code, ignore_errors=True))
        return [str(token) for token in tokens]
    except javalang.tokenizer.LexerError as e:
        print(f"LexerError: {e} for code: {code}")
        return []

def preprocess_code(code):
    """Preprocess Java code (normalize quotes and whitespace, tokenize)."""
    if not isinstance(code, str):
        return ""
    # Normalize quotes and whitespace
    code = re.sub(r"(?<!\\)\'", '"', code.strip())
    return code

def load_model(model_path):
    """Load the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def evaluate_code(model, tokenizer, processed_code, maxlen=200):
    """Evaluate processed code for vulnerability using the model."""
    try:
        # Convert processed code into sequences using Keras Tokenizer
        seq = tokenizer.texts_to_sequences([processed_code])
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=maxlen)
        prediction = model.predict(padded_seq, verbose=0)[0][0]
        return prediction
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Analyze Java code for vulnerabilities")
    parser.add_argument("java_file", help="Path to the Java file")
    args = parser.parse_args()
    model_path="vul_detection_model.h5"
    model = load_model(model_path)
    
    # Load and preprocess the Java file
    try:
        with open(args.java_file, "r") as f:
            java_code = f.read()
        print("File loaded successfully")
    except Exception as e:
        print(f"Error reading Java file: {e}")
        exit(1)

    # Preprocess the entire file
    processed_code = preprocess_code(java_code)
    tokenized_code = ' '.join(tokenize_java_code(processed_code))
    
    # Initialize and fit tokenizer on the code
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([tokenized_code])
    
    # Evaluate the Java file's vulnerability
    vulnerability_score = evaluate_code(model, tokenizer, tokenized_code)

    if vulnerability_score is not None:
        print(f"Vulnerability score: {vulnerability_score:.2f}")
        if vulnerability_score > 0.5:
            print("⚠ This code is potentially vulnerable.")
        else:
            print("✓ This code is likely not vulnerable.")
    else:
        print("Error evaluating the code.")

if __name__ == "__main__":
    main()


