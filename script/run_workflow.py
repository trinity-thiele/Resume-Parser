import subprocess
import sys
from pathlib import Path
import time

def main():
    # Train Bag of Words model
    
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'models'/'bag_of_words'/'bag_of_words_model.py')],
        check=True
    )

    # Record start time
    start_time = time.time()
    # Read and process resumes
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'models'/'file_reading_application'/'file_reader.py')],
        check=True
    )

    # Test the trained model on processed resumes
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'script'/'test_model.py')],
        check=True
    )
    # Print time taken for processing and testing
    end_time = time.time()
    print(f"Total time for processing files and testing: {end_time - start_time:.2f} seconds")

    # # Evaluate SpaCy/Gemini model
    # subprocess.run(
    #     [sys.executable, str(Path(__file__).parent.parent/'nlp'/'spacy_feature_extraction.py')],
    #     check=True
    # )

    # Compare results between Bag of Words and SpaCy/Gemini
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'script'/'compare_bow_spacy_gemini.py')],
        check=True
    )

if __name__ == '__main__':
    main()