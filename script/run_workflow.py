import subprocess
import sys
from pathlib import Path

def main():
    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'models'/'file_reading_application'/'file_reader.py')],
        check=True
    )

    subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent/'script'/'test_model.py')],
        check=True
    )

if __name__ == '__main__':
    main()