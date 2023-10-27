import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
from flask import Flask, render_template, request
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
import re


app = Flask(__name__)


# preprocess and OCR
def ocr_image(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    text = pytesseract.image_to_string(image)
    return text


# correct text
# OCR function to extract text from an image
# OCR function to extract text from an image

# OCR function to extract text from an image
def ocr_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    # Filter the extracted text to keep only alphabet letters
    alphabet_only_text = ''.join(filter(str.isalpha, text))

    return alphabet_only_text

# Correct text using LanguageTool
def correct_text(text):
    from language_tool_python import LanguageTool
    tool = LanguageTool('en-US')
    corrected_text = tool.correct(text)

    # Define custom word replacements


    return corrected_text


# Main correction function
# Main correction function
# Main correction function



def improve_text_prediction(text, image_path):
    # Define custom word replacements
    word_replacements = {
        "«1": "an",
        "docum >nts": "documents",
    }

    # Split the input text into words
    words = text.split()

    # Initialize variables to track sentence and current word
    sentence = []
    current_word = ""

    for word in words:
        # Remove non-alphabetic characters from the word
        cleaned_word = re.sub(r'[^a-zA-Z]', '', word)

        if cleaned_word:
            # Check if the word is in the custom replacements
            if cleaned_word.lower() in word_replacements:
                # Replace the word with the custom replacement
                corrected_word = word_replacements[cleaned_word.lower()]
            else:
                # If it contains non-alphabetic characters, extract the missing characters using OCR
                extracted_word = ocr_text_from_image(image_path)
                # Use LanguageTool to correct the extracted word
                corrected_word = correct_text(extracted_word)
        else:
            # If the word is already alphabetic, no need for correction
            corrected_word = word

        # Add a space after the word if it originally had a space
        if ' ' in word:
            corrected_word += ' '

        # Append the corrected word to the current word
        current_word += corrected_word

    # Append the current word to the sentence
    sentence.append(current_word)

    # Combine the corrected words into a single sentence
    corrected_sentence = ''.join(sentence)

    return corrected_sentence


# route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No image file provided."

        uploaded_image = request.files['image']

        if uploaded_image.filename == '':
            return "No selected image file."

        if uploaded_image:
            image_path = "uploaded_image.png"
            uploaded_image.save(image_path)
            ocr_result = ocr_image(image_path)
            corrected_result = correct_text(ocr_result)

            return render_template('index.html', ocr_result=ocr_result, corrected_result=corrected_result)

    return render_template('index.html', ocr_result=None, corrected_result=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3999, debug=True)