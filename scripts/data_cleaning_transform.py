import pandas as pd
import logging
import re
import os
import emoji
from sklearn.preprocessing import StandardScaler

# Ensure logs folder exists
os.makedirs("../logs", exist_ok=True)

# Configure logging to write to file & display in Jupyter Notebook
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/data_cleaning_transform.log"),  # Log to file
        logging.StreamHandler()  # Log to Jupyter Notebook output
        ]
)

def remove_duplicates(data):
  """
  Removes duplicate elements from a list while preserving the order.
  """
  seen = set()
  result = []
  for item in data:
    if item not in seen:
      result.append(item)
      seen.add(item)
  return result


def extract_emojis(text):
    """ Extract emojis from text, return 'No emoji' if none found. """
    emojis = ''.join(c for c in text if c in emoji.EMOJI_DATA)
    return emojis if emojis else "No emoji"


def remove_emojis(text):
  """
  Removes emojis from a given text.
  """
  return emoji.replace_emoji(text, replace='')

def standardize_dataset(df, column_name):
  """
  Standardizes specified columns in a pandas DataFrame.
  """
  df_standardized = df.copy()
  df_standardized[column_name] = df_standardized[column_name].str.replace('\n', ' ', regex=True) 
  df_standardized[column_name] = df_standardized[column_name].str.strip() 

  return df_standardized

def validate_and_transform_data(df):
    """
    Validates and transforms a DataFrame by performing several operations:
    - Remove duplicates
    - Convert date columns to datetime format, replacing NaT with None
    - Convert 'ID' column to integer for PostgreSQL BIGINT compatibility
    - Extract emojis and store them in a new column
    - Remove emojis from message text
    - Extract YouTube links into a separate column
    - Remove YouTube links from message text
    - Rename columns to match PostgreSQL schema
    - Standardize text columns (strip and lowercasing)
    """

    def remove_duplicates(dataframe):
        return dataframe.drop_duplicates()

    def convert_dates(dataframe):
        for col in dataframe.select_dtypes(include=['object']).columns:
            try:
                dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')
            except Exception:
                pass
        return dataframe

    def replace_nat_with_none(dataframe):
        return dataframe.where(pd.notnull(dataframe), None)

    def convert_id_to_integer(dataframe):
        if 'ID' in dataframe.columns:
            dataframe['ID'] = pd.to_numeric(dataframe['ID'], errors='coerce').fillna(0).astype(int)
        return dataframe

    def extract_emojis(dataframe):
        def emoji_extractor(text):
            return ''.join(char for char in text if char in emoji.UNICODE_EMOJI['en'])
        
        dataframe['extracted_emojis'] = dataframe['Message'].apply(emoji_extractor)
        return dataframe

    def remove_emojis(dataframe):
        dataframe['message'] = dataframe['Message'].str.replace(r'[^\w\s,]', '', regex=True)
        return dataframe

    def extract_youtube_links(dataframe):
        def youtube_extractor(text):
            return re.findall(r'(https?://[^\s]+)', text)
        
        dataframe['youtube_links'] = dataframe['Message'].apply(youtube_extractor)
        return dataframe

    def remove_youtube_links(dataframe):
        dataframe['message'] = dataframe['Message'].str.replace(r'https?://[^\s]+', '', regex=True)
        return dataframe

    def rename_columns(dataframe):
        dataframe.rename(columns={
           "Channel Title": "channel_title",
            "Channel Username": "channel_username",
            "ID": "message_id",
            "Message": "Message",
            "Date": "message_date",
            "Media Path": "media_path",
            "emoji_used": "emoji_used",
            "youtube_links": "youtube_links"
        }, inplace=True)
        return dataframe

    def standardize_text_columns(dataframe):
        for col in dataframe.select_dtypes(include=['object']).columns:
            dataframe[col] = dataframe[col].str.strip().str.lower()
        return dataframe

    # Perform all transformations
    df = remove_duplicates(df)
    df = convert_dates(df)
    df = replace_nat_with_none(df)
    df = convert_id_to_integer(df)
    df = extract_emojis(df)
    df = remove_emojis(df)
    df = extract_youtube_links(df)
    df = remove_youtube_links(df)
    df = rename_columns(df)
    df = standardize_text_columns(df)

    return df

def cleaned_data(df, output_path):
    """ Save cleaned data to a new CSV file. """
    try:
        df.to_csv(output_path, index=False)
        logging.info(f"✅ Cleaned data saved successfully to '{output_path}'.")
        print(f"✅ Cleaned data saved successfully to '{output_path}'.")
    except Exception as e:
        logging.error(f"❌ Error saving cleaned data: {e}")
        raise
