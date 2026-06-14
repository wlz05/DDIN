# DDIN: Domain-aware Disentanglement Interaction Network for Multimodal Fake News Detection

import pandas as pd
import re


def process_reasoning_data(input_file, output_file):
    """Split text_reasoning column into text/image/cross_modal reasoning fields."""
    df = pd.read_csv(input_file)

    # Ensure required columns exist
    required_columns = ['text_reasoning', 'image_reasoning', 'cross_modal_reasoning']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    # Match entries starting with 1), 2), 3)
    pattern = r'(\d\)\s.*?)(?=\s*\d\)|$)'

    for index, row in df.iterrows():
        reasoning_text = str(row['text_reasoning']).strip()
        matches = re.findall(pattern, reasoning_text, re.DOTALL)

        text_content = ""
        image_content = ""
        cross_content = ""

        for match in matches:
            if match.startswith('1)'):
                text_content = match[2:].strip()
            elif match.startswith('2)'):
                image_content = match[2:].strip()
            elif match.startswith('3)'):
                cross_content = match[2:].strip()

        df.at[index, 'text_reasoning'] = text_content
        df.at[index, 'image_reasoning'] = image_content
        df.at[index, 'cross_modal_reasoning'] = cross_content

    df.to_csv(output_file, index=False)
    print(f"Processing complete. Saved to {output_file}")


if __name__ == "__main__":
    input_csv = "./weibo_with_reasoning.csv"
    output_csv = "./processed_weibo_reasoning.csv"
    process_reasoning_data(input_csv, output_csv)
