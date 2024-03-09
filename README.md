# Multiple Choice Question Answering

This script performs multiple choice question answering using a pre-trained language model. It evaluates the model's performance on a set of test questions and saves the results to a CSV file.

## Requirements

- Python 3.x
- numpy
- pandas
- transformers

## Usage

### Arguments

- `--ntrain NTRAIN`, `-k NTRAIN`: Number of training examples to use (default: 5).
- `--data_dir DATA_DIR`, `-d DATA_DIR`: Directory containing the MMLU data files (default: "data").
- `--save_dir SAVE_DIR`, `-s SAVE_DIR`: Directory to save the results (default: "results").
- `--model MODEL`, `-m MODEL`: Name or path of the pre-trained model to use (default: "mistralai/Mistral-7B-v0.1").

### Example

This command runs the script with the following settings:
- Number of training examples: 10
- Data directory: "data"
- Save directory: "results"
- Pre-trained model: "mistralai/Mistral-7B-v0.1"

## Data Format

The script expects the data files to be in the following format:
- Training data: `{subject}_dev.csv` (e.g., "history_dev.csv")
- Test data: `{subject}_test.csv` (e.g., "history_test.csv")

The data files should be located in the specified `data_dir` directory, with the training data in the "dev" subdirectory and the test data in the "test" subdirectory.

Each data file should be a CSV file with the following columns:
- Question
- Option A
- Option B
- Option C
- Option D
- Answer

## Results

The script evaluates the model's performance on each subject's test questions and saves the results to a CSV file in the specified `save_dir` directory. The CSV file will have the same name as the subject (e.g., "history.csv").

The CSV file will contain the following columns:
- Question
- Option A
- Option B
- Option C
- Option D
- Answer
- Correct (indicating whether the model's prediction is correct or not)

The script also prints the average accuracy for each subject and the overall weighted accuracy across all subjects.

## License

This script is released under the [MIT License](https://opensource.org/licenses/MIT).