import os
from collections import Counter
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lyricsgenius import Genius
from spotify import Song, search_playlist_interactive
from PIL import Image, ImageDraw, ImageFont
import random
import re
from textblob import TextBlob
from collections import defaultdict
import seaborn as sns



GENIUS_ACCESS_TOKEN = "your_genius_access_token"


def read_songs_from_csv(csv_file: str) -> List[Song]:
    df = pd.read_csv(csv_file)
    return [Song(**row._asdict()) for row in df.itertuples(index=False)]


def create_spider_plot(songs: List[Song], output_file: str):
    audio_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ]

    songs_df = pd.DataFrame([song.__dict__ for song in songs])
    audio_features_df = songs_df[audio_features]

    # Calculate the mean value of each audio feature
    mean_audio_features = audio_features_df.mean().values

    # Create a spider plot
    angles = np.linspace(0, 2 * np.pi, len(audio_features), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))[:-1]
    mean_audio_features = np.concatenate((mean_audio_features, [mean_audio_features[0]]))[:-1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, mean_audio_features, "o-", linewidth=2)
    ax.fill(angles, mean_audio_features, alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, audio_features)
    ax.set_title("Playlist Audio Features")
    plt.savefig(output_file)


def get_lyrics(song: Song, genius: Genius) -> str:
    song_info = genius.search_song(song.name, song.artist)
    return song_info.lyrics if song_info else ""


def get_lyrics_filename(song: Song) -> str:
    return f"lyrics/{song.artist}_{song.name}.txt"

def download_lyrics(song: Song, genius: Genius) -> str:
    song_info = genius.search_song(song.name, song.artist)
    return song_info.lyrics if song_info else ""

def load_lyrics_from_file(filename: str) -> str:
    with open(filename, "r") as file:
        return file.read()

def save_lyrics_to_file(filename: str, lyrics: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(lyrics)
    except UnicodeEncodeError as e:
        print(f"Error saving lyrics for '{filename}': {e}")

def get_lyrics_for_song(song: Song, genius: Genius) -> str:
    print(f"Getting lyrics for '{song}'")
    lyrics_filename = get_lyrics_filename(song)
    if os.path.isfile(lyrics_filename):
        print(f"Lyrics found for '{song.name}' by '{song.artist}'.")
        lyrics = load_lyrics_from_file(lyrics_filename)
    else:
        print(f"Lyrics not found for '{song.name}' by '{song.artist}'.")
        lyrics = download_lyrics(song, genius)
        save_lyrics_to_file(lyrics_filename, lyrics)
    return lyrics

def get_sentiment_score(text: str) -> float:
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def create_sentiment_histogram(sentiment_scores: List[float], output_file: str):
    sentiment_histogram_file = output_file
    plt.hist(sentiment_scores, bins=20, color='blue', edgecolor='black')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Sentiment Histogram')
    plt.savefig(sentiment_histogram_file)
    # clear plt 
    plt.clf()



def create_popularity_histogram(songs: List[Song], output_file: str):
    songs_df = pd.DataFrame([song.__dict__ for song in songs])

    plt.figure(figsize=(10, 5))
    sns.histplot(data=songs_df, x="popularity", bins=10)
    plt.title("Song Popularity Distribution")
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    plt.savefig(output_file)
    # clear plt 
    plt.clf()


def average_sentiment_by_artist(songs: List[Song], sentiment_scores: List[float]) -> dict[str, float]:
    artist_sentiment = defaultdict(list)
    for song, sentiment in zip(songs, sentiment_scores):
        artist_sentiment[song.artist].append(sentiment)

    artist_sentiment_avg = {artist: sum(sentiments) / len(sentiments) for artist, sentiments in artist_sentiment.items()}
    return artist_sentiment_avg

def create_artist_sentiment_bar_chart(artist_sentiment: dict[str, float], output_file: str):
    sorted_artists = sorted(artist_sentiment.items(), key=lambda x: x[1], reverse=True)
    artists = [artist[0] for artist in sorted_artists]
    avg_sentiments = [artist[1] for artist in sorted_artists]

    plt.figure(figsize=(10, 5))
    plt.bar(artists, avg_sentiments, color='#5DA5DA', edgecolor='black')
    plt.xlabel('Artists')
    plt.ylabel('Average Sentiment Score')
    plt.title('Average Sentiment Score per Artist')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, v in enumerate(avg_sentiments):
        plt.text(i, v+0.05, str(round(v, 2)), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file)
    # clear plt 
    plt.clf()


def create_artist_distribution_pie_chart(songs: List[Song], output_file: str):
    artist_count = Counter(song.artist for song in songs)
    sorted_artists = sorted(artist_count.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top 5 artists and their counts
    top_artists = sorted_artists[:5]
    top_artists_names = [artist[0] for artist in top_artists]
    top_artists_counts = [artist[1] for artist in top_artists]
    
    # Get the count of all remaining artists
    other_artists_count = sum(artist[1] for artist in sorted_artists[5:])
    
    # Add the "Other" category to the lists
    top_artists_names.append("Other")
    top_artists_counts.append(other_artists_count)
    
    # Create the pie chart
    plt.pie(top_artists_counts, labels=top_artists_names, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Track Distribution by Artist')
    plt.savefig(output_file)
    # clear plt 
    plt.clf()




def main():
    playlist_id = search_playlist_interactive()
    csv_file = f"output/{playlist_id}.csv"  # Replace with your CSV file path
    songs = read_songs_from_csv(csv_file)
    print(f"Number of songs: {len(songs)}")

    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Number of songs: {len(songs)}")

    # Create a spider plot for audio features
    spider_plot_file = os.path.join(output_dir, "spider_plot.png")
    create_spider_plot(songs, spider_plot_file)
    print(f"Saved spider plot to: {spider_plot_file}")
    print(f"Number of songs: {len(songs)}")
    # Create a word cloud for lyrics
    word_cloud_file = os.path.join(output_dir, "word_cloud.png")
    print(f"Saved word cloud to: {word_cloud_file}")
    print(f"Number of songs: {len(songs)}")
    # Create a popularity histogram
    popularity_histogram_file = os.path.join(output_dir, "popularity_histogram.png")
    create_popularity_histogram(songs, popularity_histogram_file)
    print(f"Saved popularity histogram to: {popularity_histogram_file}")
    print(f"Number of songs: {len(songs)}")
    genius = Genius(GENIUS_ACCESS_TOKEN)
    sentiment_scores = []
    for song in songs:
        print("Getting lyrics for song:", song)
        lyrics = get_lyrics_for_song(song, genius)
        
        sentiment_score = get_sentiment_score(lyrics)
        sentiment_scores.append(sentiment_score)
        print("Sentiment score:", sentiment_score)

    sentiment_histogram_file = os.path.join(output_dir,"sentiment_histogram.png")
    create_sentiment_histogram(sentiment_scores, sentiment_histogram_file)

    artist_sentiment = average_sentiment_by_artist(songs, sentiment_scores)
    artist_sentiment_bar_chart_file = os.path.join(output_dir,"artist_sentiment_bar_chart.png")
    create_artist_sentiment_bar_chart(artist_sentiment, artist_sentiment_bar_chart_file)

    artist_distribution_pie_chart_file = os.path.join(output_dir,"artist_distribution_pie_chart.png")
    create_artist_distribution_pie_chart(songs, artist_distribution_pie_chart_file)


if __name__ == "__main__":
    main()
