import csv

# Replace 'your_file.csv' with your actual CSV file path
file_path = '/mnt/D/BDA/python/Anime Recommendation System/dataset/music_with_lyrics.csv'

with open(file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the first line
    print("Header:", header)
