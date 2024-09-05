import pandas as pd

# Ensure the file path is correct
file = '/Users/mjg/Desktop/5. Coding/NLP excercisses/5. Sentiment Analysis/training.1600000.processed.noemoticon.csv'

# Read the CSV file with the specified encoding and delimiter without headers
df = pd.read_csv(file, encoding='latin-1', delimiter=',', header=None)  # No headers

# Assign new column names
df.columns = ["sentiment", "id", "date", "query", "author", "text"]

# Adjust Pandas settings to display all columns
pd.set_option('display.max_columns', None)

print(df.shape)
print(df['sentiment'].unique())

# Replace sentiment value 4 with 1
df['sentiment'] = df['sentiment'].replace(4, 1)

def get_balanced_df(num_rows):
    # Filter rows with sentiment 0 and sentiment 1
    df_sentiment_0 = df[df['sentiment'] == 0].head(num_rows)
    df_sentiment_1 = df[df['sentiment'] == 1].head(num_rows)
    
    # Concatenate the two DataFrames
    new_df = pd.concat([df_sentiment_0, df_sentiment_1], ignore_index=True)
    
    # Select only the 'sentiment' and 'text' columns
    new_df = new_df[['sentiment', 'text']]
    
    return new_df

number_of_rows = 5000
new_df = get_balanced_df(number_of_rows)
print(new_df.shape)
print(new_df.tail(10))

# Save the DataFrame to a new CSV file
output_file = f'/Users/mjg/Desktop/5. Coding/NLP excercisses/5. Sentiment Analysis/balanced_sentiment {(number_of_rows/1000)*2}k.csv'
new_df.to_csv(output_file, index=False)

print(f"New DataFrame saved to {output_file}")
